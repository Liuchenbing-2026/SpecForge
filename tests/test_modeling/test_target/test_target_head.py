import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from specforge.modeling.target.target_head import (
    TargetHead,
    _resolve_target_head_key,
)


class TargetHeadPreprocessTest(unittest.TestCase):
    def test_shifts_loss_mask_with_inputs_and_target(self):
        input_ids = torch.tensor([[10, 11, 12, 13]])
        target = torch.tensor([[[20], [21], [22], [23]]])
        loss_mask = torch.tensor([[0, 0, 1, 1]])

        shifted_ids, shifted_target, shifted_mask = TargetHead.preprocess(
            None,
            input_ids,
            target,
            loss_mask,
        )

        self.assertEqual(shifted_ids.tolist(), [[11, 12, 13, 0]])
        self.assertEqual(shifted_target.squeeze(-1).tolist(), [[21, 22, 23, 0]])
        self.assertEqual(shifted_mask.squeeze(-1).tolist(), [[0, 1, 1, 0]])

    def test_shifts_batched_disjoint_and_boundary_masks(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        target = torch.arange(20).reshape(2, 5, 2)
        loss_mask = torch.tensor(
            [[True, False, True, True, False], [False, False, False, False, True]]
        )

        _, _, shifted_mask = TargetHead.preprocess(
            None,
            input_ids,
            target,
            loss_mask,
        )

        self.assertEqual(shifted_mask.dtype, torch.bool)
        self.assertEqual(
            shifted_mask.squeeze(-1).tolist(),
            [
                [False, True, True, False, False],
                [False, False, False, True, False],
            ],
        )

    def test_single_token_sequence_has_no_trainable_eagle_row(self):
        _, _, shifted_mask = TargetHead.preprocess(
            None,
            torch.tensor([[7]]),
            torch.tensor([[[11, 12]]]),
            torch.tensor([[1]]),
        )

        self.assertEqual(shifted_mask.squeeze(-1).tolist(), [[0]])


class _FakeSafeOpen:
    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def get_tensor(self, key):
        return self._tensors[key]


def _write_tied_checkpoint(directory, hidden_size=4, vocab_size=6):
    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": hidden_size,
        "intermediate_size": 8,
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "vocab_size": vocab_size,
        "tie_word_embeddings": True,
    }
    (Path(directory) / "config.json").write_text(json.dumps(config), encoding="utf-8")
    embedding = torch.arange(vocab_size * hidden_size, dtype=torch.float32).reshape(
        vocab_size, hidden_size
    )
    (Path(directory) / "model.safetensors").touch()
    (Path(directory) / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    return embedding


class TargetHeadTiedEmbeddingsTest(unittest.TestCase):
    def test_resolves_missing_head_key_to_tied_embedding(self):
        weight_map = {"model.embed_tokens.weight": "model.safetensors"}
        config = type("Config", (), {"tie_word_embeddings": True})()
        self.assertEqual(
            _resolve_target_head_key(weight_map, config, "lm_head.weight"),
            ("model.embed_tokens.weight", True),
        )

    def test_prefers_explicit_head_key_when_present(self):
        weight_map = {
            "lm_head.weight": "model.safetensors",
            "model.embed_tokens.weight": "model.safetensors",
        }
        config = type("Config", (), {"tie_word_embeddings": True})()
        self.assertEqual(
            _resolve_target_head_key(weight_map, config, "lm_head.weight"),
            ("lm_head.weight", False),
        )

    def test_resolves_manual_workaround_embedding_key(self):
        weight_map = {"model.language_model.embed_tokens.weight": "model.safetensors"}
        config = type("Config", (), {"tie_word_embeddings": True})()
        self.assertEqual(
            _resolve_target_head_key(
                weight_map, config, "model.language_model.embed_tokens.weight"
            ),
            ("model.language_model.embed_tokens.weight", False),
        )

    def test_raises_when_head_key_missing_and_embeddings_not_tied(self):
        weight_map = {"model.embed_tokens.weight": "model.safetensors"}
        config = type("Config", (), {"tie_word_embeddings": False})()
        with self.assertRaises(FileNotFoundError) as ctx:
            _resolve_target_head_key(weight_map, config, "lm_head.weight")
        self.assertIn("tie_word_embeddings", str(ctx.exception))

    def test_load_weights_uses_tied_embedding_tensor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding = _write_tied_checkpoint(tmp_dir)
            tensors = {"model.embed_tokens.weight": embedding}
            with patch(
                "specforge.modeling.target.target_head.safe_open",
                side_effect=lambda *_args, **_kwargs: _FakeSafeOpen(tensors),
            ):
                head = TargetHead.from_pretrained(tmp_dir)
            torch.testing.assert_close(
                head.fc.weight.detach().float().to("cpu"), embedding
            )
            self.assertTrue(head.tied_word_embeddings)

    def test_load_weights_raises_clear_error_for_untied_missing_head(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _write_tied_checkpoint(tmp_dir)
            config_path = Path(tmp_dir) / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["tie_word_embeddings"] = False
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as ctx:
                TargetHead.from_pretrained(tmp_dir)
            self.assertIn("lm_head.weight", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
