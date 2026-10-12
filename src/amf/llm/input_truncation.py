from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

class HeadTailMessageTruncator:
    """Limit local-model inputs while preserving instructions and the latest task.

    AMF currently sends one assembled system message: stable instructions are
    near its beginning, while the current question and latest observations are
    near its end. Keeping both ends is safer than one-sided truncation when an
    abnormal trajectory makes a prompt excessively long.
    """

    marker = "\n\n[... middle of input truncated by the local model client ...]\n\n"

    def __init__(
        self,
        *,
        tokenizer_file: str | Path,
        max_input_tokens: int,
        template_reserve_tokens: int = 512,
        preserve_head_ratio: float = 0.5,
    ) -> None:
        if max_input_tokens <= template_reserve_tokens:
            raise ValueError(
                "max_input_tokens must exceed template_reserve_tokens."
            )
        if not 0.0 < preserve_head_ratio < 1.0:
            raise ValueError("preserve_head_ratio must be between zero and one.")
        self.tokenizer_file = Path(tokenizer_file)
        self.max_input_tokens = int(max_input_tokens)
        self.template_reserve_tokens = int(template_reserve_tokens)
        self.preserve_head_ratio = float(preserve_head_ratio)
        self._tokenizer: Any | None = None
        self._is_tiktoken = False

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            if self.tokenizer_file.exists():

                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(self.tokenizer_file))
            else:

                import tiktoken

                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                self._is_tiktoken = True
        return self._tokenizer

    def _encode(self, text: str) -> list[int]:
        tokenizer = self.tokenizer
        if self._is_tiktoken:
            return tokenizer.encode(text)
        return list(tokenizer.encode(text, add_special_tokens=False).ids)

    def _decode(self, token_ids: list[int]) -> str:
        tokenizer = self.tokenizer
        if self._is_tiktoken:
            return tokenizer.decode(token_ids)
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))

    def truncate(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        result = copy.deepcopy(messages)
        encoded: dict[int, list[int]] = {}
        for index, message in enumerate(result):
            content = message.get("content")
            if isinstance(content, str):
                encoded[index] = self._encode(content)

        original_tokens = sum(len(ids) for ids in encoded.values())

        content_budget = self.max_input_tokens - self.template_reserve_tokens
        if original_tokens <= content_budget:
            return result, None

        marker_ids = self._encode(self.marker)
        while sum(len(ids) for ids in encoded.values()) > content_budget:
            if not encoded:
                break
            index = max(encoded, key=lambda item: len(encoded[item]))
            ids = encoded[index]
            excess = sum(len(value) for value in encoded.values()) - content_budget
            keep_count = max(0, len(ids) - excess - len(marker_ids))
            head_count = int(keep_count * self.preserve_head_ratio)
            tail_count = keep_count - head_count
            replacement = ids[:head_count] + marker_ids
            if tail_count:
                replacement += ids[-tail_count:]
            if len(replacement) >= len(ids):
                replacement = []
            encoded[index] = replacement
            result[index]["content"] = self._decode(replacement)

        final_tokens = sum(len(ids) for ids in encoded.values())
        return result, {
            "strategy": "head_tail_middle_removal",
            "max_input_tokens": self.max_input_tokens,
            "template_reserve_tokens": self.template_reserve_tokens,
            "original_content_tokens": original_tokens,
            "final_content_tokens": final_tokens,
            "removed_content_tokens": original_tokens - final_tokens,
        }
