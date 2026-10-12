import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open

from specforge.modeling.target.target_utils import (
    load_target_config,
    target_hidden_size,
    target_vocab_size,
)
from specforge.utils import get_local_device, padding


def _tied_embedding_key_candidates(head_key: str) -> list[str]:
    """Embedding keys that may stand in for a tied ``lm_head``.

    Targets with ``tie_word_embeddings: true`` store a single embedding matrix
    and reuse it as the output projection at load time, so their checkpoints
    never contain ``lm_head.weight``.  Probe the conventional embedding key
    derived from the requested head key plus the common transformer layouts.
    """

    candidates: list[str] = []
    prefix = head_key.split(".")[0]
    if prefix:
        candidates.append(f"{prefix}.embed_tokens.weight")
    candidates.extend(
        [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ]
    )
    seen: set[str] = set()
    return [key for key in candidates if not (key in seen or seen.add(key))]


def _resolve_target_head_key(
    weight_map: dict[str, str], config, lm_head_key: str
) -> tuple[str, bool]:
    """Resolve the checkpoint key for the frozen target output head.

    Returns ``(key, tied)`` where ``tied`` indicates the resolved key is the
    input embedding matrix reused as a tied output head.
    """

    if lm_head_key in weight_map:
        return lm_head_key, False

    if getattr(config, "tie_word_embeddings", False):
        for candidate in _tied_embedding_key_candidates(lm_head_key):
            if candidate in weight_map:
                return candidate, True
        raise FileNotFoundError(
            f"tie_word_embeddings is enabled but none of the expected "
            f"embedding tensors {_tied_embedding_key_candidates(lm_head_key)} "
            f"are present in the target checkpoint"
        )

    raise FileNotFoundError(
        f"{lm_head_key!r} not found in the target checkpoint weight map. "
        f"For a target with tied word embeddings set "
        f"tie_word_embeddings: true, or point model.lm_head_key at the output "
        f"head tensor."
    )


class TargetHead(nn.Module):
    def __init__(
        self,
        model_path,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = load_target_config(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.hidden_size = target_hidden_size(self.config)
        self.vocab_size = target_vocab_size(self.config)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "TargetHead":
        target_head = cls(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            cache_dir=cache_dir,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().to(
            device=get_local_device(), dtype=torch.bfloat16
        )
        return target_head

    @torch.no_grad()
    def load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        cache_dir: Optional[str] = None,
    ):
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            self.model_path = snapshot_download(repo_id=model_path, cache_dir=cache_dir)

        # model_path is a local directory
        # check if there is file ending with index.json
        glob_path = os.path.join(self.model_path, "*.index.json")
        index_json_path = glob.glob(glob_path)

        if len(index_json_path) == 0:
            raise FileNotFoundError(f"No index.json file found in {self.model_path}")
        if len(index_json_path) > 1:
            raise FileNotFoundError(
                f"Multiple index.json files found in {self.model_path}"
            )
        index_json_path = index_json_path[0]

        with open(index_json_path, "r") as f:
            index_json = json.load(f)
        weight_map = index_json["weight_map"]
        resolved_key, tied = _resolve_target_head_key(
            weight_map, self.config, lm_head_key
        )
        ckpt_file = weight_map[resolved_key]

        if ckpt_file.endswith(".safetensors"):
            with safe_open(
                os.path.join(self.model_path, ckpt_file), framework="pt"
            ) as f:
                lm_head = f.get_tensor(resolved_key)
        else:
            state_dict = torch.load(os.path.join(self.model_path, ckpt_file))
            lm_head = state_dict[resolved_key]
        self.fc.weight.copy_(lm_head)
        self.tied_word_embeddings = tied

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)

    def preprocess(self, input_ids, target, loss_mask):
        # Shift loss_mask with input_ids and final target states; auxiliary hidden
        # states remain unshifted because the draft model consumes them at row p.
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = padding(loss_mask, left=False)[..., None]
        return input_ids, target, loss_mask
