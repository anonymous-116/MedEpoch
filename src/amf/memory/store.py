from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from amf.memory.item import KnowledgeItem

class BaseKnowledgeStore(ABC):
    @abstractmethod
    async def add(self, item: KnowledgeItem) -> KnowledgeItem:
        ...

    @abstractmethod
    async def get(self, item_id: str) -> KnowledgeItem | None:
        ...

    @abstractmethod
    async def list(self, filters: dict[str, Any] | None = None) -> list[KnowledgeItem]:
        ...

    @abstractmethod
    async def update(self, item_id: str, patch: dict[str, Any]) -> KnowledgeItem | None:
        ...
