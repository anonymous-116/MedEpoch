from __future__ import annotations

from typing import Any

from amf.agents.base import BaseAgent
from amf.llm.client import BaseLLMClient
from amf.llm.parser import AgentDecision, OutputParser
from amf.state import EpisodeState

DATA_INSPECTION_CONTEXT = (
    "This request is part of a controlled research evaluation of a medical "
    "question-answering system using a benchmark dataset. It is not intended "
    "to guide real-world medical care or carry out any real-world action. "
    "Please provide a safe, clinically appropriate, educational response. "
    "If the question contains sensitive or potentially harmful content, "
    "address it at an appropriate level, avoid actionable instructions that "
    "could enable harm, and prioritize safety guidance."
)

class ReactAgent(BaseAgent):
    """A tool-using agent that emits compact ReAct decisions.

    The outer protocol is labeled text; only Action Input is strict JSON.
    """

    def __init__(self, llm: BaseLLMClient, parser: OutputParser | None = None) -> None:
        self.llm = llm
        self.parser = parser or OutputParser()

    def fork(self) -> "ReactAgent":
        """Isolate per-request LLM audit state for one concurrent episode."""

        return ReactAgent(self.llm.fork(), parser=self.parser)

    _RUNTIME_ONLY_STOP = ["\nObservation:"]

    async def step(
        self,
        prompt: str,
        state: EpisodeState,
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> AgentDecision:
        if messages:
            raw_output = await self.llm.generate(
                messages=messages,
                stop=self._RUNTIME_ONLY_STOP,

                data_inspection_context=DATA_INSPECTION_CONTEXT,
            )
        else:
            raw_output = await self.llm.generate(
                prompt,
                stop=self._RUNTIME_ONLY_STOP,
                data_inspection_context=DATA_INSPECTION_CONTEXT,
            )
        decision = self.parser.parse(raw_output)

        if self.llm.last_usage is not None:
            decision.metadata["llm_usage"] = self.llm.last_usage
        if self.llm.last_request_recovery is not None:
            decision.metadata["llm_request_recovery"] = dict(
                self.llm.last_request_recovery
            )
        return decision
