from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass(slots=True)
class ProviderRequest:
    provider: str
    operation: str
    query: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class ProviderResponse:
    ok: bool
    data: dict[str, Any] | list[Any] | str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def call(self, request: ProviderRequest) -> ProviderResponse:
        """Call a provider operation."""
