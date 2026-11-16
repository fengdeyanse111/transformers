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
import types
from abc import ABC, abstractmethod

from transformers import is_torch_available
from transformers.testing_utils import SINGLE_PROCESS_DEVICES, is_context_parallel_test, torch_device

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


def _cp_plan(model):
    """`model.cp_plan` with `CP_PLAN` on its base model, `*` replaced by the layer numbers."""
    prefix = "" if model.base_model is model else f"{model.base_model_prefix}."
    plan = dict(model.cp_plan)
    for name, _ in model.base_model.named_modules():
        if style := _get_parameter_plan(name, CP_PLAN, is_weight=False):
            plan[prefix + name] = style
    # a layer without the attention style would run its attention on this rank's slice only
    if missing := set(CP_PLAN.values()) - set(plan.values()):
        raise ValueError(f"{sorted(missing)} of `{CP_PLAN}` name no module of {type(model).__name__}")
    return plan


def _refusal(config, model_class, cp_size):
    """The error that stops `model_class` from running with context parallelism, or `None`."""
    mesh = types.SimpleNamespace(size=lambda: cp_size, get_local_rank=lambda: 0)
    try:
        with torch.device("meta"):  # from the config the workers reload from the checkpoint
            model = model_class(type(config).from_dict(config.to_diff_dict()))
        apply_context_parallelism(model, mesh, _cp_plan(model))
    except Exception as error:
        return error
    return None


def _load_cp_and_reference_models(model_dir, model_class):
    """The checkpoint with the context parallel plan installed, the same weights unsharded, and the `cp` mesh."""
    cp_mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    model_cp = model_class.from_pretrained(model_dir)
    apply_context_parallelism(model_cp, cp_mesh, _cp_plan(model_cp))
    model_ref = model_class.from_pretrained(model_dir)
    return model_cp, model_ref, cp_mesh


def _inputs_and_labels(config, seq_len=8, packed=False, padded=False):
    """
    Two rows of `seq_len` tokens, with themselves as labels. `packed` puts two documents in every row, so
    `position_ids` restart mid-sequence; `padded` masks the end of the first row and the start of the second.
    """
    input_ids = torch.randint(0, config.vocab_size, (2, seq_len), generator=torch.Generator().manual_seed(0))
    inputs, labels = {"input_ids": input_ids}, input_ids.clone()
    if packed:
        first = seq_len // 2
        inputs["position_ids"] = torch.cat([torch.arange(first), torch.arange(seq_len - first)]).expand(2, -1)
    if padded:
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, -3:] = 0
        attention_mask[1, :2] = 0
        inputs["attention_mask"] = attention_mask
        labels = labels.masked_fill(attention_mask == 0, -100)
    return inputs, labels


def _test_cp_forward_impl(rank, config, model_class, atol, rtol, batch_options):
    """This rank's slice of the logits matches the logits of the whole sequences."""
    with _deterministic_init_model_dir(rank, config, torch.float32, model_class) as model_dir:
        model_cp, model_ref, cp_mesh = _load_cp_and_reference_models(model_dir, model_class)
        inputs, _ = _inputs_and_labels(config, **batch_options)

        with torch.no_grad():
            logits = model_ref(**inputs, use_cache=False).logits
            logits_cp = model_cp(**inputs).logits

        # this rank's slice of the sequences padded to a multiple of `cp_size`: the padding at the end of the last
        # slice has no counterpart in the unsharded logits
        start = cp_mesh.get_local_rank() * logits_cp.shape[1]
        expected = logits[:, start : start + logits_cp.shape[1]]
        logits_cp = logits_cp[:, : expected.shape[1]]
        torch.testing.assert_close(logits_cp, expected, atol=atol, rtol=rtol, msg="CP and non-CP logits differ")
        dist.barrier()


def _test_cp_backward_impl(rank, config, model_class, atol, rtol, batch_options):
    """Every rank reports the loss of the whole sequences, and the gradients averaged over the cp ranks match."""
    with _deterministic_init_model_dir(rank, config, torch.float32, model_class) as model_dir:
        model_cp, model_ref, cp_mesh = _load_cp_and_reference_models(model_dir, model_class)
        inputs, labels = _inputs_and_labels(config, **batch_options)

        loss = model_ref(**inputs, labels=labels, use_cache=False).loss
        loss.backward()
        loss_cp = model_cp(**inputs, labels=labels).loss
        loss_cp.backward()
        torch.testing.assert_close(loss_cp, loss, atol=atol, rtol=rtol, msg="CP and non-CP losses differ")

        # the gradient of this rank's slice, averaged over the cp ranks as FSDP2 does: a parameter the slice does not
        # reach (an expert no token of it is routed to) contributes nothing
        for (name, param), (_, param_cp) in zip(
            model_ref.named_parameters(), model_cp.named_parameters(), strict=True
        ):
            grad = torch.zeros_like(param) if param.grad is None else param.grad
            grad_cp = torch.zeros_like(param_cp) if param_cp.grad is None else param_cp.grad.clone()
            dist.all_reduce(grad_cp, group=cp_mesh.get_group())
            torch.testing.assert_close(grad_cp / cp_mesh.size(), grad, atol=atol, rtol=rtol, msg=f"{name} differs")
        dist.barrier()


class ContextParallelTesterMixin(ABC):
    """
    Mixin for context parallel tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides `get_config()` and `causal_lm_class`; every test
    loads that tiny model on `context_parallel_size` CPU ranks, each holding a slice of every sequence, and compares
    it with the same weights on the whole sequences.
    """

    context_parallel_size: int = 2
    context_parallel_atol: float = 1e-5
    context_parallel_rtol: float = 1e-5

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    def _skip_if_cp_not_supported(self):
        if torch._C._get_accelerator().type == "mps":
            self.skipTest("The device mesh is built on the default accelerator, which MPS cannot back")
        if torch_device in SINGLE_PROCESS_DEVICES:
            self.skipTest(f"Distributed tests spawn one process per device, which {torch_device!r} doesn't allow")
        if _get_available_fsdp_workers() < self.context_parallel_size:
            self.skipTest(f"Need at least {self.context_parallel_size} workers")
        model_type = self.model_tester.get_config().model_type
        if CONTEXT_PARALLEL_TEST_MODEL_TYPES is not None and model_type not in CONTEXT_PARALLEL_TEST_MODEL_TYPES:
            self.skipTest(f"Context parallel tests are not enabled for model_type={model_type!r}")

    def _get_cp_config(self):
        config = copy.deepcopy(self.model_tester.get_config())
        # the auxiliary load-balancing loss of the MoE models does not sum over the tokens of a sequence
        if hasattr(config.get_text_config(), "output_router_logits"):
            config.get_text_config().output_router_logits = False
        return config

    def _run_cp_test(self, test_impl, **batch_options):
        self._skip_if_cp_not_supported()
        config, model_class = self._get_cp_config(), self.model_tester.causal_lm_class
        if (refusal := _refusal(config, model_class, self.context_parallel_size)) is not None:
            self.skipTest(f"Context parallelism cannot run this model: {refusal}")
        run_distributed(world_size=self.context_parallel_size, backend="gloo")(test_impl)(
            config, model_class, self.context_parallel_atol, self.context_parallel_rtol, batch_options
        )

    @is_context_parallel_test
    def test_cp_forward(self):
        self._run_cp_test(_test_cp_forward_impl)

    @is_context_parallel_test
    def test_cp_forward_packed(self):
        """Two documents in every sequence: `position_ids` restart mid-sequence, and so does the mask."""
        self._run_cp_test(_test_cp_forward_impl, packed=True)

    @is_context_parallel_test
    def test_cp_forward_padded(self):
        """7 tokens over 2 ranks, so the model pads the sequences, and a 2D `attention_mask` of padded rows."""
        self._run_cp_test(_test_cp_forward_impl, seq_len=7, padded=True)

    @is_context_parallel_test
    def test_cp_backward(self):
        self._run_cp_test(_test_cp_backward_impl)

    @is_context_parallel_test
    def test_cp_backward_packed(self):
        self._run_cp_test(_test_cp_backward_impl, packed=True)

    @is_context_parallel_test
    def test_cp_backward_padded(self):
        """The padding the model appends and the masked positions are not trained on."""
        self._run_cp_test(_test_cp_backward_impl, seq_len=7, padded=True)
