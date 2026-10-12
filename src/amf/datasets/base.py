from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

@dataclass(slots=True)
class DatasetExample:
    """One dataset row split into runtime case and evaluation-only data."""

    row_index: int
    case: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)

class BaseDataset(ABC):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def iter_examples(
        self,
        start_index: int = 0,
        end_index: int | None = None,
    ) -> Iterable[DatasetExample]:
        """Yield rows in the 0-based half-open interval [start_index, end_index)."""
