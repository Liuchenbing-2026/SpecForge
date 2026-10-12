import unittest
import warnings

import torch

from specforge.data.parse import (
    EmptyConversationError,
    GeneralParser,
)
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


class _CharTokenizer:
    """Deterministic tokenizer where one character maps to one token.

    ``encode`` returns ``len(text)`` ids so character offsets stay aligned
    with token indices, avoiding any dependency on a remote tokenizer.
    """

    pad_token_id = 0
    unk_token_id = 0
    bos_token = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        parts = []
        for message in messages:
            parts.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )
        return "".join(parts)

    def _ids(self, text, max_length=None):
        ids = [ord(char) for char in text]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        return self._ids(text, max_length)

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
    ):
        return _Encoding(self._ids(text, max_length))


class _Encoding:
    def __init__(self, ids):
        self.input_ids = torch.tensor([ids], dtype=torch.long)


class EmptyConversationGuardTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _CharTokenizer()
        self.chat_template = TEMPLATE_REGISTRY.get("qwen")

    def test_parse_raises_dedicated_error_for_assistant_only_row(self):
        parser = GeneralParser(self.tokenizer, self.chat_template)
        conversation = [{"role": "assistant", "content": "no user turn here"}]
        with self.assertRaises(EmptyConversationError):
            parser.parse(conversation, max_length=128)

    def test_preprocessing_skips_row_without_user_turn(self):
        conversations = [
            [{"role": "assistant", "content": "leading assistant only"}],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "It is 4."},
            ],
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = preprocess_conversations(
                tokenizer=self.tokenizer,
                conversations=conversations,
                chat_template=self.chat_template,
                max_length=256,
                tools=[[] for _ in conversations],
            )

        # The malformed row is dropped; the valid row survives.
        self.assertEqual(len(results["input_ids"]), 1)
        self.assertEqual(len(results["loss_mask"]), 1)
        self.assertEqual(len(results["attention_mask"]), 1)
        self.assertTrue(
            any("Skipping conversation" in str(item.message) for item in caught)
        )
        # The surviving row has the assistant content supervised.
        self.assertEqual(int(results["loss_mask"][0].sum().item()) > 0, True)


if __name__ == "__main__":
    unittest.main()
