# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Context parallel tester mixin for model tests."""

import copy
from abc import ABC, abstractmethod

from transformers import is_torch_available
from transformers.testing_utils import SINGLE_PROCESS_DEVICES, is_fsdp_test, torch_device

from .distributed_testing_utils import _deterministic_init_model_dir, run_distributed
from .test_fsdp_mixin import _get_available_fsdp_workers


if is_torch_available():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from transformers.distributed.context_parallel import apply_context_parallelism
    from transformers.distributed.parallel_utils import _get_parameter_plan


# Set to None to run the context parallel tests for every causal language model.
CONTEXT_PARALLEL_TEST_MODEL_TYPES = {
    "gemma3_text",
    "glm4_moe",
    "llama",
    "mistral",
    "phi3",
    "qwen3",
    "qwen3_moe",
}

# The plan of a decoder-only base model: its layers run on this rank's slice, their attention on the whole sequence.
CP_PLAN = {"layers.*": "cp_split_sequence", "layers.*.self_attn": "cp_ulysses"}
CP_RTOL = 1e-5
CP_ATOL = 1e-5


class _MeshStub:
    """A `cp` mesh that only has to report its size and the caller's rank."""

    def __init__(self, size, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def _resolve_cp_plan(model):
    """`model.cp_plan` with `CP_PLAN` on the modules of its base model, `*` replaced by the layer numbers."""
    base_model_name = "" if model.base_model is model else model.base_model_prefix
    plan = dict(model.cp_plan)
    for name, _ in model.base_model.named_modules():
        if (style := _get_parameter_plan(name, CP_PLAN, is_weight=False)) is not None:
            plan[f"{base_model_name}.{name}" if base_model_name else name] = style
    # both rules have to match: layers without the attention style would run attention on this rank's slice only
    for style in CP_PLAN.values():
        if style not in plan.values():
            raise ValueError(f"the `{style}` rule of `{CP_PLAN}` names no module of {type(model).__name__}")
    return plan


def _refusal(config, model_class, cp_size):
    """
    The error that stops `model_class` from running with context parallelism, or `None`. The model is built from the
    config a checkpoint reloads, as the workers do.
    """
    try:
        with torch.device("meta"):
            model = model_class(type(config).from_dict(config.to_diff_dict()))
        apply_context_parallelism(model, _MeshStub(cp_size), _resolve_cp_plan(model))
    except Exception as error:
        return error
    return None


def _tiny_batch(config, batch_size, seq_len, packed=False, seed=0):
    """A batch every rank builds alike."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator)
    if not packed:
        return input_ids, None
    # two documents back to back, which `position_ids` has to describe for the whole sequence
    first = seq_len // 2
    position_ids = torch.cat([torch.arange(first), torch.arange(seq_len - first)]).expand(batch_size, -1)
    return input_ids, position_ids


def _compare_with_unsharded(init_dir, batch, model_class):
    """
    Run the full `batch` through an unsharded model and through the same weights with context parallelism installed,
    then compare the loss, this rank's slice of the logits and every gradient.
    """
    reference = model_class.from_pretrained(init_dir)
    reference_output = reference(**batch)
    reference_output.loss.backward()
    reference_grads = {
        name: param.grad.clone() for name, param in reference.named_parameters() if param.grad is not None
    }
    reference_loss, reference_logits = reference_output.loss.detach(), reference_output.logits.detach()
    del reference

    cp_mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    cp_rank, cp_size = cp_mesh.get_local_rank(), cp_mesh.size()
    model = model_class.from_pretrained(init_dir)
    apply_context_parallelism(model, cp_mesh, _resolve_cp_plan(model))
    output = model(**batch)
    output.loss.backward()

    # every rank computed the loss of the whole sequences, and its own slice of the logits
    expected_logits = reference_logits.chunk(cp_size, dim=1)[cp_rank]
    torch.testing.assert_close(output.logits, expected_logits, rtol=CP_RTOL, atol=CP_ATOL, msg="Logits mismatch")
    torch.testing.assert_close(output.loss, reference_loss, rtol=CP_RTOL, atol=CP_ATOL, msg="Loss mismatch")

    # FSDP2 averages the gradients over the cp ranks
    for name, param in model.named_parameters():
        expected = reference_grads.get(name)
        if param.grad is None or expected is None:
            assert param.grad is None and expected is None, f"Only one of the two runs has a gradient for {name}"
            continue
        grad = param.grad.clone()
        dist.all_reduce(grad, group=cp_mesh.get_group())
        torch.testing.assert_close(
            grad / cp_size, expected, rtol=CP_RTOL, atol=CP_ATOL, msg=f"Gradient mismatch for {name}"
        )


def _test_cp_matches_unsharded(rank, config, model_class, packed):
    """The loss, this rank's logits and every gradient of a context parallel model match the run on whole sequences."""
    with _deterministic_init_model_dir(rank, config, torch.float32, model_class) as init_dir:
        input_ids, position_ids = _tiny_batch(config, batch_size=2, seq_len=8, packed=packed)
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "use_cache": False}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        _compare_with_unsharded(init_dir, batch, model_class)
        # `init_dir` is rank 0's temporary directory: wait for the other ranks before it is removed
        dist.barrier()


class ContextParallelTesterMixin(ABC):
    """Context parallel tests on 2 ranks, added to the causal language model test classes."""

    cp_world_size: int = 2

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    def _run_cp_matches_unsharded(self, packed):
        if torch._C._get_accelerator().type == "mps":
            self.skipTest("The device mesh is built on the default accelerator, which MPS cannot back")
        if torch_device in SINGLE_PROCESS_DEVICES:
            self.skipTest(f"Distributed tests spawn one process per device, which {torch_device!r} doesn't allow")
        if _get_available_fsdp_workers() < self.cp_world_size:
            self.skipTest(f"Need at least {self.cp_world_size} workers")
        config = copy.deepcopy(self.model_tester.get_config())
        if (
            CONTEXT_PARALLEL_TEST_MODEL_TYPES is not None
            and config.model_type not in CONTEXT_PARALLEL_TEST_MODEL_TYPES
        ):
            self.skipTest(f"Context parallel tests are not enabled for model_type={config.model_type!r}")

        # the auxiliary load-balancing loss of the MoE models does not sum over the tokens of a sequence
        if hasattr(config.get_text_config(), "output_router_logits"):
            config.get_text_config().output_router_logits = False
        model_class = self.model_tester.causal_lm_class
        if (refusal := _refusal(config, model_class, self.cp_world_size)) is not None:
            self.skipTest(f"Context parallelism cannot run this model: {refusal}")
        run_distributed(world_size=self.cp_world_size, backend="gloo")(_test_cp_matches_unsharded)(
            config, model_class, packed
        )

    @is_fsdp_test
    def test_cp_matches_unsharded(self):
        """The loss, this rank's logits and every gradient match the run on whole sequences."""
        self._run_cp_matches_unsharded(packed=False)

    @is_fsdp_test
    def test_cp_matches_unsharded_packed(self):
        """Two documents in every sequence, so `position_ids` restart mid-sequence."""
        self._run_cp_matches_unsharded(packed=True)
