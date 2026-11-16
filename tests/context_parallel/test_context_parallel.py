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
"""
The context parallel styles, each on its own: `cp_ulysses` against plain attention and the loss of `cp_model` on 2
ranks, then `cp_split_sequence` on a decoder layer, `cp_model` on the inputs of a model, and
`apply_context_parallelism` on a model. The models themselves are tested by `ContextParallelTesterMixin`.
"""

import inspect
import types
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch.distributed.device_mesh import init_device_mesh

from transformers import (
    DynamicCache,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    MistralConfig,
)
from transformers.distributed.context_parallel import (
    ALL_CP_STYLES,
    ContextParallelAttention,
    SplitSequenceParallel,
    UlyssesAttention,
    apply_context_parallelism,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import is_context_parallel_test

from ..distributed_testing_utils import run_distributed
from ..test_context_parallel_mixin import _cp_plan


def _mesh(size=2, rank=0):
    """A `cp` mesh that only has to report its size and the caller's rank."""
    return types.SimpleNamespace(size=lambda: size, get_local_rank=lambda: rank)


def _config(**kwargs):
    """A `LlamaConfig` small enough for CPU tests, with `sdpa` attention."""
    defaults = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 99,
        "attn_implementation": "sdpa",
    }
    return LlamaConfig(**{**defaults, **kwargs})


def _cp_mesh():
    return init_device_mesh("cpu", (dist.get_world_size(),))


class _Attention(torch.nn.Module):
    """What the attention functions read from a model's attention module, and the call a model makes."""

    def __init__(self, config=None, num_key_value_groups=1):
        super().__init__()
        self.config, self.is_causal, self.num_key_value_groups = config or _config(), True, num_key_value_groups

    def forward(self, query, key, value, attention_mask=None, **kwargs):
        attention_function = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        return attention_function(self, query, key, value, attention_mask, **kwargs)


class _Layer(torch.nn.Module):
    """A decoder layer stand-in that records what its forward receives."""

    def __init__(self):
        super().__init__()
        self.seen = {}

    def forward(self, hidden_states, **kwargs):
        self.seen = {"hidden_states": hidden_states, **kwargs}
        return hidden_states


class _SpyAttention(ContextParallelAttention):
    """An attention style that records the modules it runs, then runs them as they are."""

    def __init__(self):
        self.seen = []

    def attention(self, cp_mesh, attention_function, module, *args, **kwargs):
        self.seen.append(module)
        return attention_function(module, *args, **kwargs)


# =============================================================================
# On 2 ranks
# =============================================================================


def _mask(config, batch, seq, name):
    """The mask a model builds over the whole sequence: causal, or from the `position_ids` of two packed documents."""
    if name == "causal":
        return torch.ones(seq, seq, dtype=torch.bool).tril().expand(batch, 1, seq, seq)
    position_ids = torch.cat([torch.arange(seq // 2), torch.arange(seq // 2)]).expand(batch, -1)
    mask_function = create_sliding_window_causal_mask if name == "sliding" else create_causal_mask
    return mask_function(config, torch.zeros(batch, seq, config.hidden_size), None, None, position_ids=position_ids)


def _test_ulysses(rank, kv_heads, mask_name):
    """The output and the gradients of this rank's slice match plain attention on the whole sequence."""
    mesh, cp_size = _cp_mesh(), dist.get_world_size()
    batch, heads, seq, head_dim = 2, 4, 8, 4
    config = MistralConfig(
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        sliding_window=4,
        attn_implementation="sdpa",
    )
    module, mask = _Attention(config, heads // kv_heads), _mask(config, batch, seq, mask_name)
    generator = torch.Generator().manual_seed(0)
    query, key, value = (
        torch.randn(batch, n, seq, head_dim, generator=generator, requires_grad=True)
        for n in (heads, kv_heads, kv_heads)
    )
    upstream = torch.randn(batch, seq, heads, head_dim, generator=generator)

    # this rank holds a contiguous slice of the sequence with every head, and the mask of the whole sequence
    local = [t.detach().chunk(cp_size, dim=2)[rank].clone().requires_grad_() for t in (query, key, value)]
    output, _ = UlyssesAttention().attention(mesh, sdpa_attention_forward, module, *local, mask)
    (output * upstream.chunk(cp_size, dim=1)[rank]).sum().backward()
    reference, _ = sdpa_attention_forward(module, query, key, value, mask)
    (reference * upstream).sum().backward()

    torch.testing.assert_close(output, reference.chunk(cp_size, dim=1)[rank])
    for local_tensor, tensor in zip(local, (query, key, value)):
        torch.testing.assert_close(local_tensor.grad, tensor.grad.chunk(cp_size, dim=2)[rank])


def _test_ulysses_kwargs(rank):
    """The attention function gets the model's kwargs unchanged, and its share of the attention sinks."""
    mesh, cp_size = _cp_mesh(), dist.get_world_size()
    batch, heads, seq, head_dim = 1, 4, 8, 2
    seen = {}

    def recording_attention(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs, attention_mask=attention_mask)
        return torch.zeros(batch, seq, heads // cp_size, head_dim), None

    local = [torch.randn(batch, heads, seq // cp_size, head_dim) for _ in range(3)]
    mask = torch.ones(batch, 1, seq, seq).tril()
    position_ids, s_aux = torch.arange(seq).unsqueeze(0), torch.randn(heads)
    UlyssesAttention().attention(
        mesh, recording_attention, _Attention(), *local, mask, position_ids=position_ids, s_aux=s_aux, dropout=0.0
    )

    assert seen["attention_mask"] is mask and seen["position_ids"] is position_ids and seen["dropout"] == 0.0
    # attention sinks: one logit per head, so this rank's share of the heads
    torch.testing.assert_close(seen["s_aux"], s_aux.chunk(cp_size)[rank])


def _test_loss(rank):
    """Every rank reports the loss of the whole sequences, with the gradient of its own slice times `cp_size`."""
    mesh, cp_size = _cp_mesh(), dist.get_world_size()
    style, model = ALL_CP_STYLES["cp_model"], types.SimpleNamespace(config=types.SimpleNamespace(pad_token_id=0))
    batch, seq, vocab_size = 2, 8, 11
    logits = torch.randn(batch, seq, vocab_size, generator=torch.Generator().manual_seed(0), requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch, seq), generator=torch.Generator().manual_seed(1))
    labels[0, -3:] = -100

    def cp_loss(local_logits, **kwargs):
        """What the model does between the two transforms of `cp_model`: its own loss, on its slice of the logits."""
        _, kwargs = style.transform_inputs_pre_forward(
            model, (), {"input_ids": labels, "labels": labels, **kwargs}, mesh
        )
        loss = ForCausalLMLoss(
            local_logits, labels, vocab_size, kwargs["num_items_in_batch"], shift_labels=kwargs["shift_labels"]
        )
        return style.transform_output_post_forward(model, types.SimpleNamespace(loss=loss), mesh).loss

    local_logits = logits.detach().chunk(cp_size, dim=1)[rank].clone().requires_grad_()
    loss = cp_loss(local_logits)
    loss.backward()
    ForCausalLMLoss(logits, labels, vocab_size).backward()
    torch.testing.assert_close(loss, ForCausalLMLoss(logits, labels, vocab_size))
    torch.testing.assert_close(local_logits.grad, logits.grad.chunk(cp_size, dim=1)[rank] * cp_size)

    # a caller's `num_items_in_batch` scales the loss instead of the token count
    expected = ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch=7)
    torch.testing.assert_close(cp_loss(local_logits, num_items_in_batch=7), expected)
    # a slice without a single label contributes nothing, and reports no NaN
    labels[:, : seq // cp_size + 1] = -100
    torch.testing.assert_close(cp_loss(local_logits), ForCausalLMLoss(logits, labels, vocab_size))


@is_context_parallel_test
class TestContextParallelOnTwoRanks(unittest.TestCase):
    @parameterized.expand([(4, "causal"), (2, "packed"), (2, "sliding")])
    def test_ulysses_matches_full_attention(self, kv_heads, mask_name):
        run_distributed(world_size=2, backend="gloo")(_test_ulysses)(kv_heads, mask_name)

    def test_ulysses_passes_the_models_kwargs_through(self):
        run_distributed(world_size=2, backend="gloo")(_test_ulysses_kwargs)()

    def test_loss_matches_full_sequence_loss(self):
        run_distributed(world_size=2, backend="gloo")(_test_loss)()


# =============================================================================
# The styles, one at a time
# =============================================================================


class TestContextParallelAttention(unittest.TestCase):
    """The attention style swaps the module's entry of `ALL_ATTENTION_FUNCTIONS` while the module runs."""

    def setUp(self):
        self.sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.addCleanup(ALL_ATTENTION_FUNCTIONS.register, "sdpa", self.sdpa)
        self.spy, self.routed, self.plain = _SpyAttention(), _Attention(), _Attention()
        self.spy.install_forward(self.routed, _mesh())
        self.qkv = [torch.randn(2, 4, 8, 4) for _ in range(3)]

    def test_routes_the_installed_module_only(self):
        output, _ = self.routed(*self.qkv)
        torch.testing.assert_close(output, self.sdpa(self.routed, *self.qkv, None)[0])
        self.plain(*self.qkv)
        self.assertEqual(self.spy.seen, [self.routed])
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], self.sdpa)

    def test_runs_the_function_registered_at_call_time(self):
        # a hub kernel loaded after the model was built registers another function under the key
        kernel = mock.Mock(wraps=self.sdpa)
        ALL_ATTENTION_FUNCTIONS.register("sdpa", kernel)
        self.routed(*self.qkv)
        self.assertEqual(kernel.call_count, 1)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], kernel)

    def test_restores_the_entry_when_the_attention_raises(self):
        failing = mock.Mock(side_effect=RuntimeError("kernel"))
        ALL_ATTENTION_FUNCTIONS.register("sdpa", failing)
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            self.routed(*self.qkv)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], failing)


class TestUlyssesCheck(unittest.TestCase):
    def _check(self, config, cp_size=2):
        UlyssesAttention().check(_Attention(config), _mesh(cp_size))

    def test_accepts_attention_that_takes_the_mask(self):
        self._check(_config())
        # the mask the model builds has the sliding window in it, and flash kernels build it themselves
        self._check(MistralConfig(attn_implementation="sdpa", sliding_window=5))
        self._check(MistralConfig(attn_implementation="flash_attention_2", sliding_window=5))

    def test_refuses_attention_that_takes_no_mask(self):
        for attn_implementation in ("eager", "flex_attention"):
            with self.assertRaisesRegex(ValueError, "takes no attention mask"):
                self._check(_config(attn_implementation=attn_implementation))

    def test_refuses_heads_the_ranks_cannot_split(self):
        # 6 heads, then 8 heads with 2 key/value heads, over 4 ranks
        for heads, kv_heads in ((6, 6), (8, 2)):
            with self.assertRaisesRegex(ValueError, "must divide both the number of attention heads"):
                self._check(_config(hidden_size=8 * heads, num_attention_heads=heads, num_key_value_heads=kv_heads), 4)


class TestSplitSequenceParallel(unittest.TestCase):
    def test_cuts_the_hidden_states_once_and_the_rotary_embeddings_everywhere(self):
        first, second = _Layer(), _Layer()
        SplitSequenceParallel().install_forward(first, _mesh(rank=1), split_hidden_states=True)
        SplitSequenceParallel().install_forward(second, _mesh(rank=1))
        hidden_states, cos, sin = (torch.randn(2, 8, 4) for _ in range(3))
        mask, position_ids = torch.ones(2, 1, 8, 8, dtype=torch.bool).tril(), torch.arange(8).expand(2, -1)

        local = first(hidden_states, attention_mask=mask, position_ids=position_ids, position_embeddings=(cos, sin))
        torch.testing.assert_close(local, hidden_states.chunk(2, dim=1)[1])
        torch.testing.assert_close(first.seen["position_embeddings"], (cos.chunk(2, dim=1)[1], sin.chunk(2, dim=1)[1]))
        # the mask and the positions stay those of the whole sequence, which the attention style needs
        self.assertIs(first.seen["attention_mask"], mask)
        self.assertIs(first.seen["position_ids"], position_ids)
        # some models pass the hidden states by keyword
        torch.testing.assert_close(first(hidden_states=hidden_states, position_embeddings=cos), local)
        # a later layer gets this rank's slice already, and cuts the embeddings the model computed for all of it
        second(local, position_embeddings=cos)
        torch.testing.assert_close(second.seen["hidden_states"], local)
        torch.testing.assert_close(second.seen["position_embeddings"], cos.chunk(2, dim=1)[1])

    def test_refuses_a_sequence_the_ranks_cannot_split(self):
        layer = _Layer()
        SplitSequenceParallel().install_forward(layer, _mesh(), split_hidden_states=True)
        with self.assertRaisesRegex(ValueError, "must be divisible by `cp_size`"):
            layer(torch.randn(1, 7, 4))


class TestModelContextParallel(unittest.TestCase):
    """`cp_model` on a model's inputs: no cache, this rank's shifted labels, padding to a multiple of `cp_size`."""

    def _inputs(self, rank=0, *args, **kwargs):
        model = LlamaForCausalLM(_config(pad_token_id=7))
        args, kwargs = ALL_CP_STYLES["cp_model"].transform_inputs_pre_forward(model, args, kwargs, _mesh(rank=rank))
        self.assertEqual(args, ())  # a positional input is passed on by name
        return kwargs

    def test_refuses_a_cache_and_turns_it_off(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "`past_key_values` are not allowed"):
            self._inputs(0, input_ids=input_ids, past_key_values=DynamicCache())
        kwargs = self._inputs(0, input_ids, use_cache=True)
        self.assertIs(kwargs["input_ids"], input_ids)
        self.assertIs(kwargs["use_cache"], False)

    def test_pads_the_sequences_to_a_multiple_of_cp_size(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        kwargs = self._inputs(
            0,
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=torch.tensor([[0, 1, 2, 0, 1]]),
        )
        # the pad token, with no label and no attention, at a position continuing the last document
        torch.testing.assert_close(kwargs["input_ids"], torch.tensor([[1, 2, 3, 4, 5, 7]]))
        torch.testing.assert_close(kwargs["labels"], torch.tensor([[1, 2, 3, 4, 5, -100]]))
        torch.testing.assert_close(kwargs["attention_mask"], torch.tensor([[1, 1, 1, 1, 1, 0]]))
        torch.testing.assert_close(kwargs["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2]]))
        # a divisible sequence is left alone
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        self.assertIs(self._inputs(0, input_ids=input_ids)["input_ids"], input_ids)


class TestApplyContextParallelism(unittest.TestCase):
    def test_accepts_a_base_model_and_keeps_the_forward_signature(self):
        # a base model computes no loss of its own, so the caller's loss is theirs to shard
        model = LlamaModel(_config())
        apply_context_parallelism(model, _mesh(), _cp_plan(model))
        # `Trainer` reads the signature to drop the dataset columns the model does not take
        model = LlamaForCausalLM(_config())
        apply_context_parallelism(model, _mesh(), _cp_plan(model))
        self.assertIn("labels", inspect.signature(model.forward).parameters)

    def test_refusals(self):
        class LlamaWithOwnAttention(LlamaForCausalLM):
            _supports_attention_backend = False

        for model, message in (
            (LlamaForSequenceClassification(_config()), "only supports the causal language modeling loss"),
            (LlamaWithOwnAttention(_config()), "`_supports_attention_backend`"),
            (LlamaForCausalLM(_config(layer_types=["full_attention", "linear_attention"])), "linear_attention"),
        ):
            # refused before the plan is even read
            with self.subTest(type(model).__name__), self.assertRaisesRegex(ValueError, message):
                apply_context_parallelism(model, _mesh(), {})

    def test_every_model_is_the_root_of_its_plan(self):
        config = _config()
        config.base_model_cp_plan = {"layers.*": "cp_split_sequence", "layers.*.self_attn": "cp_ulysses"}
        model = LlamaForCausalLM(config)
        self.assertEqual(model.model.cp_plan, {"": "cp_model", **config.base_model_cp_plan})
        # the head, not its base model, is the `""` of the head's plan
        self.assertEqual(
            model.cp_plan,
            {"": "cp_model", "model.layers.*": "cp_split_sequence", "model.layers.*.self_attn": "cp_ulysses"},
        )
        self.assertNotIn("base_model_cp_plan", config.to_dict())
