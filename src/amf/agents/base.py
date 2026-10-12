from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from amf.llm.parser import AgentDecision
from amf.state import EpisodeState

class BaseAgent(ABC):
    @abstractmethod
    def fork(self) -> "BaseAgent":
        """Return an episode-local Agent for concurrent execution."""
        ...

    @abstractmethod
    async def step(
        self,
        prompt: str,
        state: EpisodeState,
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> AgentDecision:
        ...
