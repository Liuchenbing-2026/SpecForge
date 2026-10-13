"""Regression tests for DFlash target-layer validation (issues #501 / #520).

When the captured target hidden states do not line up with the draft's
``target_layer_ids`` (for example the target only returned one layer while the
draft expects several), DFlash used to fail with an opaque ``IndexError`` or a
raw ``RuntimeError: mat1 and mat2 shapes cannot be multiplied`` from the
projection layer. These tests pin the clearer up-front validation instead.
"""

import unittest

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
)


def _tiny_config(num_draft_layers: int, num_target_layers: int) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_draft_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * num_draft_layers,
    )
    config.block_size = 2
    config.num_target_layers = num_target_layers
    return config


class TestExtractContextFeature(unittest.TestCase):
    def test_selects_and_concatenates_requested_layers(self):
        hidden_states = [torch.full((1, 3, 4), float(i)) for i in range(6)]
        # layer_ids are zero-based decoder layers; index 0 is the embedding
        # output, so layer 0 -> hidden_states[1] and layer 2 -> hidden_states[3].
        out = extract_context_feature(hidden_states, [0, 2])
        self.assertEqual(tuple(out.shape), (1, 3, 8))
        self.assertTrue(torch.all(out[..., :4] == 1.0))
        self.assertTrue(torch.all(out[..., 4:] == 3.0))

    def test_empty_layer_ids_raise(self):
        with self.assertRaises(ValueError):
            extract_context_feature([torch.zeros(1, 2, 4)], [])

    def test_too_few_hidden_states_raise_clear_error(self):
        # Only one captured layer (plus the embedding output) but layer 10 was
        # requested: must raise a descriptive ValueError, not an IndexError.
        hidden_states = [torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)]
        with self.assertRaises(ValueError) as ctx:
            extract_context_feature(hidden_states, [10])
        self.assertIn("target_layer_ids", str(ctx.exception))


class TestForwardTargetHiddenWidth(unittest.TestCase):
    def test_mismatched_target_hidden_width_raises_clear_error(self):
        config = _tiny_config(num_draft_layers=2, num_target_layers=4)
        model = DFlashDraftModel(config)
        self.assertEqual(model.fc.in_features, 2 * config.hidden_size)

        position_ids = torch.arange(3).unsqueeze(0)
        noise_embedding = torch.zeros(1, 3, config.hidden_size)
        # Wrong width: the capture returned a single layer instead of the two
        # the draft expects.
        with self.assertRaises(ValueError) as ctx:
            model(
                position_ids=position_ids,
                noise_embedding=noise_embedding,
                target_hidden=torch.zeros(1, 3, config.hidden_size),
            )
        self.assertIn("target_hidden", str(ctx.exception))


class TestBuildTargetLayerIds(unittest.TestCase):
    def test_single_draft_layer_uses_middle_target_layer(self):
        self.assertEqual(build_target_layer_ids(32, 1), [16])

    def test_multi_layer_spans_expected_range(self):
        ids = build_target_layer_ids(32, 5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], 1)
        self.assertEqual(ids[-1], 29)


if __name__ == "__main__":
    unittest.main()
