from __future__ import annotations

import json
from typing import Any, Iterable

from amf.datasets.base import BaseDataset, DatasetExample

class JSONLDataset(BaseDataset):
    """Generic streaming JSONL dataset.

    Subclasses override `normalize_row` to map dataset-specific schemas into
    the framework's standard runtime case.
    """

    dataset_name = "jsonl"

    def iter_examples(
        self,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> Iterable[DatasetExample]:
        if start_index < 0:
            raise ValueError("start_index must be >= 0.")
        if end_index is not None and end_index < start_index:
            raise ValueError("end_index must be >= start_index.")

        with self.path.open("r", encoding="utf-8") as file:
            for row_index, line in enumerate(file):
                if row_index < start_index:
                    continue
                if end_index is not None and row_index >= end_index:
                    break
                if not line.strip():
                    continue
                raw = json.loads(line)
                yield self.normalize_row(raw, row_index)

    def normalize_row(self, raw: dict[str, Any], row_index: int) -> DatasetExample:
        case_id = str(raw.get("case_id") or raw.get("id") or raw.get("prompt_id") or row_index)
        question = raw.get("question") or raw.get("input") or raw.get("query") or str(raw)
        case = {
            "case_id": case_id,
            "question": str(question),
            "dataset": self.dataset_name,
            "metadata": {"row_index": row_index},
        }
        return DatasetExample(row_index=row_index, case=case, metadata={"case_id": case_id})
