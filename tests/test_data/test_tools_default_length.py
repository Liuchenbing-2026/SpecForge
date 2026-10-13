"""Regression tests for the ``tools`` argument of ``preprocess_conversations``.

Previously ``tools`` defaulted to a single-element list ``[[]]``. Because the
body zips ``conversations`` with ``tools``, calling the function with more than
one conversation and no ``tools`` argument silently kept only the first
conversation. These tests pin the corrected behaviour without loading a real
tokenizer.
"""

import types
import unittest
from unittest.mock import patch

import torch

from specforge.data import preprocessing


class _StubParser:
    """Minimal stand-in for the real conversation parsers."""

    def __init__(self, tokenizer, chat_template):
        self.num_parses = 0

    def parse(self, source, max_length, **kwargs):
        self.num_parses += 1
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        loss_mask = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        return input_ids, loss_mask


def _chat_template():
    return types.SimpleNamespace(parser_type="general")


class TestToolsDefaultLength(unittest.TestCase):
    def test_multiple_conversations_without_tools_are_all_kept(self):
        conversations = [
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
            [{"role": "user", "content": "e"}, {"role": "assistant", "content": "f"}],
        ]
        with patch.object(preprocessing, "GeneralParser", _StubParser):
            results = preprocessing.preprocess_conversations(
                tokenizer=object(),
                conversations=conversations,
                chat_template=_chat_template(),
                max_length=16,
            )

        self.assertEqual(len(results["input_ids"]), 3)
        self.assertEqual(len(results["loss_mask"]), 3)
        self.assertEqual(len(results["attention_mask"]), 3)

    def test_empty_string_entries_are_skipped_not_truncated(self):
        conversations = [
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            None,
            [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
        ]
        with patch.object(preprocessing, "GeneralParser", _StubParser):
            results = preprocessing.preprocess_conversations(
                tokenizer=object(),
                conversations=conversations,
                chat_template=_chat_template(),
                max_length=16,
            )

        # ``None`` entries are skipped, the two real conversations survive.
        self.assertEqual(len(results["input_ids"]), 2)

    def test_provided_tools_must_match_conversation_count(self):
        conversations = [
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
        ]
        with patch.object(preprocessing, "GeneralParser", _StubParser):
            with self.assertRaises(ValueError):
                preprocessing.preprocess_conversations(
                    tokenizer=object(),
                    conversations=conversations,
                    chat_template=_chat_template(),
                    max_length=16,
                    tools=[[]],
                )


if __name__ == "__main__":
    unittest.main()
