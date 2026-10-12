import unittest

from specforge.data.parse import ThinkingParser
from specforge.data.template import TEMPLATE_REGISTRY


class _CharTokenizer:
    """Minimal deterministic tokenizer: each character is one token.

    ``encode`` therefore returns exactly ``len(text)`` ids, which keeps the
    character offsets computed by the parser aligned with token indices
    without depending on any remote tokenizer.
    """

    pad_token_id = 0
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        return [ord(char) for char in text]

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
    ):
        ids = [ord(char) for char in text]
        return _Encoding(ids)

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _Encoding:
    def __init__(self, ids):
        self.input_ids = _Tensor([ids])


class _Tensor(list):
    def __getitem__(self, index):
        return super().__getitem__(index)


class ThinkingHistoryLossMaskTest(unittest.TestCase):
    def _rendered_history(self):
        # Mirrors how qwen3.5 / qwen3-next-thinking render a multi-turn think
        # conversation: the thinking opener appears only on the turn after the
        # final user message, while the earlier assistant turn uses the plain
        # "<|im_start|>assistant\n" header.
        return (
            "<|im_start|>user\nWhat is 1+1?<|im_end|>\n"
            "<|im_start|>assistant\nIt is 2.<|im_end|>\n"
            "<|im_start|>user\nAnd 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nTwo plus two equals four."
            "\n</think>\n\nIt is 4.<|im_end|>\n"
        )

    def _loss_mask(self, template_key):
        chat_template = TEMPLATE_REGISTRY.get(template_key)
        parser = ThinkingParser(_CharTokenizer(), chat_template)
        parser.apply_chat_template = lambda messages, tool=None, **kwargs: (
            self._rendered_history()
        )
        conversation = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "It is 2."},
            {"role": "user", "content": "And 2+2?"},
            {"role": "assistant", "content": "It is 4."},
        ]
        _, loss_mask = parser.parse(conversation, max_length=4096)
        return loss_mask.tolist()

    def test_plain_history_turn_is_supervised(self):
        for template_key in ("qwen3.5", "qwen3-next-thinking"):
            with self.subTest(template=template_key):
                mask = self._loss_mask(template_key)
                rendered = self._rendered_history()
                first = rendered.index("It is 2.") + len("It is 2.")
                last = rendered.index("It is 4.") + len("It is 4.")
                # Character index == token index for the fake tokenizer.
                self.assertEqual(mask[rendered.index("It is 2.")], 1)
                self.assertEqual(mask[first - 1], 1)
                self.assertEqual(mask[rendered.index("It is 4.")], 1)
                self.assertEqual(mask[last - 1], 1)
                spans = 0
                previous = 0
                for value in mask:
                    if value == 1 and previous == 0:
                        spans += 1
                    previous = value
                self.assertEqual(spans, 2)


if __name__ == "__main__":
    unittest.main()
