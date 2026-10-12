from __future__ import annotations

from typing import Any

from amf.datasets.base import DatasetExample
from amf.datasets.jsonl_dataset import JSONLDataset

class HealthBenchDataset(JSONLDataset):
    """Adapter for OpenAI HealthBench JSONL rows.

    HealthBench rows include rubrics and ideal completions. Those fields are
    evaluation-only and must not be placed in `case`, because ContextAssembler
    sends `case` to the model.
    """

    dataset_name = "healthbench"

    def normalize_row(self, raw: dict[str, Any], row_index: int) -> DatasetExample:
        prompt_id = str(raw.get("prompt_id") or f"healthbench-{row_index}")
        messages = list(raw.get("prompt") or [])
        question = self._messages_to_text(messages)
        safe_metadata = {
            "row_index": row_index,
            "prompt_id": prompt_id,
            "example_tags": list(raw.get("example_tags") or []),
        }
        case = {
            "case_id": prompt_id,
            "question": question,
            "messages": messages,
            "dataset": self.dataset_name,
            "metadata": safe_metadata,
        }
        evaluation = {
            "rubrics": raw.get("rubrics") or [],
            "ideal_completions_data": raw.get("ideal_completions_data") or {},
            "canary": raw.get("canary"),
        }
        return DatasetExample(
            row_index=row_index,
            case=case,
            metadata=safe_metadata,
            evaluation=evaluation,
        )

    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
