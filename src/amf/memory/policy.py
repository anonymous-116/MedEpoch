from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from amf.providers.base import ProviderRequest
from amf.state import EpisodeState

class BaseRetrievalPolicy(ABC):
    """Decision point for memory/skill retrieval.

    Implementations decide when to retrieve, how to build a query, and which
    filters and sources to use.
    """

    @abstractmethod
    async def should_retrieve(self, state: EpisodeState) -> bool:
        ...

    @abstractmethod
    async def build_query(self, state: EpisodeState) -> str:
        ...

    async def build_filters(self, state: EpisodeState) -> dict[str, Any]:
        return {}

    async def build_request(
        self,
        state: EpisodeState,
        provider: str,
        top_k: int = 5,
        timeout: int = 30,
    ) -> ProviderRequest:

        return ProviderRequest(
            provider=provider,
            operation="retrieve",
            query=await self.build_query(state),
            payload={"top_k": top_k, "filters": await self.build_filters(state)},
            timeout=timeout,
        )

class NeverRetrievalPolicy(BaseRetrievalPolicy):
    async def should_retrieve(self, state: EpisodeState) -> bool:
        return False

    async def build_query(self, state: EpisodeState) -> str:
        return ""

class FirstTurnRetrievalPolicy(BaseRetrievalPolicy):
    """Simple default: retrieve once at the beginning of an episode."""

    def __init__(
        self,
        filters: dict[str, Any] | None = None,
        *,
        request_timeout: int = 30,
    ) -> None:
        self.filters = dict(filters or {})
        self.request_timeout = int(request_timeout)
        if self.request_timeout <= 0:
            raise ValueError("Retrieval request_timeout must be positive.")

    async def should_retrieve(self, state: EpisodeState) -> bool:
        return state.turn_id == 0

    async def build_query(self, state: EpisodeState) -> str:
        for key in ("question", "input", "query", "task"):
            value = state.case.get(key)
            if value:
                return str(value)
        return str(state.case)

    async def build_filters(self, state: EpisodeState) -> dict[str, Any]:
        return dict(self.filters)

    async def build_request(
        self,
        state: EpisodeState,
        provider: str,
        top_k: int = 5,
        timeout: int | None = None,
    ) -> ProviderRequest:

        return await super().build_request(
            state,
            provider,
            top_k,
            timeout=self.request_timeout if timeout is None else timeout,
        )
