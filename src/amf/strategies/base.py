from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from amf.agents.base import BaseAgent
from amf.memory.policy import BaseRetrievalPolicy
from amf.memory.provenance import source_id_for_case
from amf.prompt.assembler import ContextAssembler
from amf.prompt.protocol_repair import build_protocol_repair_messages
from amf.providers.base import ProviderRequest
from amf.providers.registry import ProviderRegistry
from amf.providers.tool_provider import ToolProvider
from amf.state import EpisodeState
from amf.trajectory.recorder import TrajectoryRecorder

@dataclass(slots=True)
class RuntimeContext:
    """Shared runtime services used by method strategies."""

    agent: BaseAgent
    providers: ProviderRegistry
    assembler: ContextAssembler
    recorder: TrajectoryRecorder
    max_turns: int
    memory_policy: BaseRetrievalPolicy
    skill_policy: BaseRetrievalPolicy
    memory_top_k: int
    skill_top_k: int
    mode: str
    trace_actions: bool = False
    protocol_repair_enabled: bool = True
    protocol_repair_max_attempts: int = 1
    last_model_messages: list[dict[str, Any]] | None = None

    accepted_final_messages: list[dict[str, Any]] | None = None
    accepted_final_raw_output: str | None = None

    def trace_action(self, state: EpisodeState, action: str, **fields: Any) -> None:
        """Print one compact progress line without leaking prompts or arguments."""

        if not self.trace_actions:
            return
        row_index = state.metadata.get("row_index", "-")
        suffix = " ".join(f"{key}={value}" for key, value in fields.items() if value)
        message = f"[amf] row={row_index} turn={state.turn_id} action={action}"
        if suffix:
            message += f" {suffix}"
        print(message, flush=True)

    async def prepare_context(self, state: EpisodeState) -> None:
        await self.retrieve_context(state, "memory", self.memory_policy, self.memory_top_k)
        await self.retrieve_context(state, "skill", self.skill_policy, self.skill_top_k)

    async def retrieve_context(
        self,
        state: EpisodeState,
        provider_name: str,
        policy: BaseRetrievalPolicy,
        top_k: int,
    ) -> None:
        if not self.providers.has(provider_name):
            return
        if not await policy.should_retrieve(state):
            return

        request = await policy.build_request(state, provider=provider_name, top_k=top_k)
        if provider_name == "memory":

            source_id = state.metadata.get("source_id") or source_id_for_case(state.case)
            if source_id:
                filters = dict(request.payload.get("filters") or {})
                filters.setdefault("exclude_content_source_id", str(source_id))
                request.payload["filters"] = filters
            if str(state.metadata.get("run_phase") or "") in {
                "validation",
                "test",
            }:

                filters = dict(request.payload.get("filters") or {})
                filters.setdefault(
                    "lifecycle_statuses",
                    ["active", "provisional"],
                )
                request.payload["filters"] = filters
        self.trace_action(state, f"retrieve.{provider_name}")
        self.recorder.record(state.episode_id, state.turn_id, "retrieve_context", request.to_dict())
        response = await self.providers.call(request)
        self.recorder.record(
            state.episode_id,
            state.turn_id,
            "context_returned",
            {"provider": provider_name, "response": response.to_dict()},
        )
        if response.ok and isinstance(response.data, list):
            if provider_name == "memory":
                state.metadata["same_source_excluded_count"] = int(
                    response.metadata.get("same_source_excluded_count") or 0
                )
            state.append_retrieved_context(
                source=provider_name,
                query=request.query or "",
                results=response.data,
            )

    def build_prompt(
        self,
        state: EpisodeState,
        mode: str | None = None,
        *,
        force_final: bool = False,
    ) -> str:
        state.set_status("BUILD_PROMPT")

        active_mode = mode or self.mode

        tool_specs = [] if force_final else self.tool_specs()
        prompt = self.assembler.build_prompt(
            state,
            tool_specs=tool_specs,
            mode=active_mode,
            force_final=force_final,
        )

        self.last_model_messages = self.assembler.build_messages_from_prompt(prompt)
        self.recorder.record(
            state.episode_id,
            state.turn_id,
            "prompt_built",
            {"prompt": prompt, "messages": self.last_model_messages},
        )
        return prompt

    def is_final_turn(self, turn_id: int) -> bool:
        """Return whether this is the answer-only turn reserved by max_turns."""

        return turn_id == self.max_turns - 1

    async def agent_step(
        self,
        state: EpisodeState,
        prompt: str,
        *,
        force_final: bool = False,
    ):
        state.set_status("MODEL_STEP")
        messages = self.last_model_messages
        self.last_model_messages = None
        decision = await self.agent.step(prompt, state, messages=messages)
        accepted_request_messages = self._actual_agent_messages(messages)

        needs_repair = decision.type == "invalid" or (
            force_final and decision.type == "tool_call"
        )
        self.recorder.record(
            state.episode_id,
            state.turn_id,
            "model_output_rejected" if needs_repair else "model_output",
            decision.to_dict(),
        )
        if needs_repair:
            self.trace_action(state, "model.invalid_output")

        if self.protocol_repair_enabled and needs_repair:
            for attempt in range(1, max(0, self.protocol_repair_max_attempts) + 1):
                repair_messages = build_protocol_repair_messages(
                    original_prompt=prompt,
                    invalid_output=decision.raw_output,
                    force_final=force_final,
                )
                self.trace_action(state, "model.protocol_repair", attempt=attempt)
                self.recorder.record(
                    state.episode_id,
                    state.turn_id,
                    "protocol_repair_requested",
                    {
                        "attempt": attempt,
                        "force_final": force_final,
                        "messages": repair_messages,
                    },
                )
                decision = await self.agent.step(
                    prompt,
                    state,
                    messages=repair_messages,
                )
                accepted_request_messages = self._actual_agent_messages(
                    repair_messages
                )
                self.recorder.record(
                    state.episode_id,
                    state.turn_id,
                    "protocol_repair_output",
                    {"attempt": attempt, "decision": decision.to_dict()},
                )
                needs_repair = decision.type == "invalid" or (
                    force_final and decision.type == "tool_call"
                )
                if not needs_repair:

                    self.recorder.record(
                        state.episode_id,
                        state.turn_id,
                        "model_output",
                        decision.to_dict(),
                    )
                    self.recorder.record(
                        state.episode_id,
                        state.turn_id,
                        "protocol_repair_succeeded",
                        {"attempt": attempt, "decision_type": decision.type},
                    )
                    break
            if needs_repair:
                self.recorder.record(
                    state.episode_id,
                    state.turn_id,
                    "protocol_repair_failed",
                    {
                        "attempts": max(0, self.protocol_repair_max_attempts),
                        "decision": decision.to_dict(),
                    },
                )

        if decision.type == "tool_call":
            tool_name = str((decision.tool_call or {}).get("name") or "unknown")
            self.trace_action(state, "model.tool_call", tool=tool_name)
        elif decision.type == "final":
            self.trace_action(state, "model.final_answer")
            if not needs_repair:
                self.accepted_final_messages = accepted_request_messages
                self.accepted_final_raw_output = decision.raw_output
        else:
            self.trace_action(state, "model.invalid_output")

        thought = decision.metadata.get("thought")
        if thought and not needs_repair:
            state.append_trajectory("thought", str(thought))
        return decision

    def _actual_agent_messages(
        self,
        fallback: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Return the post-normalization messages actually sent by the LLM client."""

        llm = getattr(self.agent, "llm", None)
        messages = getattr(llm, "last_request_messages", None)
        source = messages if messages is not None else fallback
        if source is None:
            return None
        return [dict(message) for message in source]

    def learning_events(self) -> list[dict[str, Any]]:
        """Return the accepted trajectory view used by learning components."""

        audit_only = {
            "model_output_rejected",
            "protocol_repair_requested",
            "protocol_repair_output",
            "protocol_repair_succeeded",
            "protocol_repair_failed",
        }
        return [
            event.to_dict()
            for event in self.recorder.events
            if event.type not in audit_only
        ]

    async def call_tool(self, state: EpisodeState, tool_call: dict[str, Any]) -> None:
        state.set_status("CALL_EXTERNAL")
        state.append_action(tool_call)
        state.append_trajectory("action", tool_call.get("name") or "")
        state.append_trajectory("action_input", dict(tool_call.get("arguments") or {}))
        self.recorder.record(state.episode_id, state.turn_id, "action_called", {"tool_call": tool_call})

        request = ProviderRequest(provider="tool", operation="call", payload={"tool_call": tool_call})
        response = await self.providers.call(request)
        observation = {
            "ok": response.ok,
            "provider": "tool",
            "data": response.data,
            "error": response.error,
            "metadata": response.metadata,
        }
        state.append_observation(observation)
        state.append_trajectory("observation", observation)
        self.recorder.record(state.episode_id, state.turn_id, "observation_returned", observation)
        state.set_status("UPDATE_STATE")
        self.recorder.record(state.episode_id, state.turn_id, "state_updated", {"turn_id": state.turn_id})

    def tool_specs(self) -> list[dict[str, Any]]:
        if not self.providers.has("tool"):
            return []
        provider = self.providers.get("tool")
        if isinstance(provider, ToolProvider):
            return provider.list_specs()
        return []

    def finish(self, state: EpisodeState, answer: str) -> None:
        state.set_final_answer(answer)
        state.append_trajectory("final_answer", answer)
        self.recorder.record(state.episode_id, state.turn_id, "final_answer", {"answer": answer})

    def question_key(self, state: EpisodeState) -> str:
        for key in ("question", "input", "query", "task"):
            if state.case.get(key):
                return str(state.case[key])
        return str(state.case)

    def record_error(self, state: EpisodeState, error: str, **payload: Any) -> None:
        state.set_status("ERROR")
        self.recorder.record(
            state.episode_id,
            state.turn_id,
            "error",
            {"error": error, **payload},
        )

class BaseStrategy(ABC):
    @abstractmethod
    async def run(self, ctx: RuntimeContext, state: EpisodeState) -> None:
        ...

    async def flush(self, ctx: RuntimeContext) -> None:
        return None

    async def on_evaluation(
        self,
        ctx: RuntimeContext,
        state: EpisodeState,
        feedback: dict[str, Any],
    ) -> None:
        """Optionally learn after an external evaluator has scored the answer."""

        return None

    def checkpoint_state(self) -> dict[str, Any]:
        """Return small in-memory strategy state needed for an exact resume."""

        return {}

    async def restore_checkpoint(
        self,
        ctx: RuntimeContext,
        payload: dict[str, Any],
    ) -> None:
        return None
