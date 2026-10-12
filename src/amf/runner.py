from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amf.agents.base import BaseAgent
from amf.answer_extraction import AnswerExtractionResult, LLMAnswerExtractor
from amf.llm import LLMUsageTracker
from amf.memory.policy import BaseRetrievalPolicy, NeverRetrievalPolicy
from amf.prompt.assembler import ContextAssembler
from amf.providers.registry import ProviderRegistry
from amf.state import EpisodeState
from amf.strategies import BaseStrategy, ReactStrategy, RuntimeContext
from amf.trajectory.recorder import TrajectoryRecorder
from amf.trajectory.serializer import TrajectorySerializer

@dataclass(slots=True)
class EpisodeResult:
    episode_id: str
    status: str
    final_answer: str | None
    trajectory_path: Path
    state: EpisodeState
    events: list[dict[str, Any]]

class Runner:
    """Episode-level runtime for the MedEpoch answering loop."""

    def __init__(
        self,
        agent: BaseAgent,
        providers: ProviderRegistry,
        assembler: ContextAssembler,
        recorder: TrajectoryRecorder,
        max_turns: int = 8,
        memory_policy: BaseRetrievalPolicy | None = None,
        skill_policy: BaseRetrievalPolicy | None = None,
        memory_top_k: int = 5,
        skill_top_k: int = 5,
        mode: str = "react",
        strategy: BaseStrategy | None = None,
        usage_tracker: LLMUsageTracker | None = None,
        trace_actions: bool = False,
        protocol_repair_enabled: bool = True,
        protocol_repair_max_attempts: int = 1,
        global_skill: str = "",
        answer_extractor: LLMAnswerExtractor | None = None,
    ) -> None:
        self.agent = agent
        self.providers = providers
        self.assembler = assembler
        self.recorder = recorder
        self.max_turns = max_turns
        self.memory_policy = memory_policy or NeverRetrievalPolicy()
        self.skill_policy = skill_policy or NeverRetrievalPolicy()
        self.memory_top_k = memory_top_k
        self.skill_top_k = skill_top_k
        self.mode = mode
        self.strategy = strategy or self._default_strategy(mode)
        self.usage_tracker = usage_tracker or LLMUsageTracker()
        self.trace_actions = trace_actions
        self.protocol_repair_enabled = protocol_repair_enabled
        self.protocol_repair_max_attempts = protocol_repair_max_attempts
        self.global_skill = global_skill
        self.answer_extractor = answer_extractor

    def set_global_skill(self, skill: str | None) -> None:
        """Replace the global skill used by subsequent episodes."""

        self.global_skill = str(skill or "").strip()

    def trace_action(
        self,
        state: EpisodeState,
        action: str,
        **fields: Any,
    ) -> None:
        """Print one compact non-prompt progress event for outer run stages.

        ReAct turn actions are emitted by ``RuntimeContext``. Evaluation runs
        outside that turn loop, so it uses this matching formatter instead.
        """

        if not self.trace_actions:
            return
        row_index = state.metadata.get("row_index", "-")
        suffix = " ".join(
            f"{key}={value}"
            for key, value in fields.items()
            if value is not None and value != ""
        )
        message = (
            f"[amf] row={row_index} turn={state.turn_id} action={action}"
        )
        if suffix:
            message += f" {suffix}"
        print(message, flush=True)

    def fork_for_episode(self) -> "Runner":
        """Create isolated mutable runtime state for a concurrent episode.

        Providers and the assembler remain shared: answer-time memory access is
        read-only inside a batch, and tool implementations own their own locks.
        Recorder, Agent/LLM request state, and Strategy objects are local.
        """

        recorder = TrajectoryRecorder(
            TrajectorySerializer(self.recorder.serializer.trajectory_dir)
        )
        return Runner(
            agent=self.agent.fork(),
            providers=self.providers,
            assembler=self.assembler,
            recorder=recorder,
            max_turns=self.max_turns,
            memory_policy=self.memory_policy,
            skill_policy=self.skill_policy,
            memory_top_k=self.memory_top_k,
            skill_top_k=self.skill_top_k,
            mode=self.mode,
            strategy=copy.copy(self.strategy),
            usage_tracker=self.usage_tracker,
            trace_actions=self.trace_actions,
            protocol_repair_enabled=self.protocol_repair_enabled,
            protocol_repair_max_attempts=self.protocol_repair_max_attempts,
            global_skill=self.global_skill,
            answer_extractor=(
                self.answer_extractor.fork() if self.answer_extractor else None
            ),
        )

    async def run_episode(
        self,
        case: dict[str, Any],
        hidden_context: dict[str, Any] | None = None,
        learning_feedback: dict[str, Any] | None = None,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        self.recorder.events = []
        state = EpisodeState.create(case)
        if hidden_context:

            state.metadata["hidden_context"] = hidden_context
        if learning_feedback:
            state.metadata["learning_feedback"] = learning_feedback
        if runtime_metadata:

            state.metadata.update(runtime_metadata)
        if self.global_skill:

            state.metadata["global_skill"] = self.global_skill

        self.trace_action(
            state,
            "episode.start",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.recorder.record(
            state.episode_id,
            state.turn_id,
            "episode_start",
            {"case": case, "runtime_metadata": runtime_metadata or {}},
        )

        ctx = RuntimeContext(
            agent=self.agent,
            providers=self.providers,
            assembler=self.assembler,
            recorder=self.recorder,
            max_turns=self.max_turns,
            memory_policy=self.memory_policy,
            skill_policy=self.skill_policy,
            memory_top_k=self.memory_top_k,
            skill_top_k=self.skill_top_k,
            mode=self.mode,
            trace_actions=self.trace_actions,
            protocol_repair_enabled=self.protocol_repair_enabled,
            protocol_repair_max_attempts=self.protocol_repair_max_attempts,
        )
        await self.strategy.run(ctx, state)

        if self.answer_extractor is not None:
            await self._extract_structured_answer(ctx, state)

        self.recorder.record(state.episode_id, state.turn_id, "episode_end", {"status": state.status})
        self.trace_action(
            state,
            "episode.end",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=state.status,
        )
        path = self.recorder.save(state)
        return EpisodeResult(
            episode_id=state.episode_id,
            status=state.status,
            final_answer=state.final_answer,
            trajectory_path=path,
            state=state,
            events=[event.to_dict() for event in self.recorder.events],
        )

    async def flush(self) -> None:
        ctx = self._runtime_context()
        await self.strategy.flush(ctx)

    async def apply_evaluation(
        self,
        result: EpisodeResult,
        feedback: dict[str, Any],
    ) -> EpisodeResult:
        """Persist an evaluation signal, then let the strategy learn from it."""

        score = feedback.get("score")
        details = dict(feedback.get("details") or {})
        evaluator_type = str(details.get("evaluator_type") or "").strip()
        explicit_correctness = details.get("is_correct")
        binary_judge_result = details.get("binary_judge")

        if isinstance(binary_judge_result, dict) and isinstance(
            binary_judge_result.get("passed"), bool
        ):
            success = bool(binary_judge_result["passed"])
        elif evaluator_type == "multiple_choice" and isinstance(
            explicit_correctness, bool
        ):
            success = explicit_correctness
        else:
            success = (
                bool(float(score) > 0.0)
                if isinstance(score, int | float)
                else None
            )
        compact_feedback = {
            "score": score,
            "metrics": dict(feedback.get("metrics") or {}),

            "success": success,
            "evaluator_type": evaluator_type or None,
        }
        result.state.metadata["learning_feedback"] = compact_feedback
        self.recorder.record(
            result.episode_id,
            result.state.turn_id,
            "evaluation_completed",
            compact_feedback,
        )
        if score is None:

            self.recorder.record(
                result.episode_id,
                result.state.turn_id,
                "learning_skipped",
                {"reason": "evaluation did not produce a valid score"},
            )
            result.trajectory_path = self.recorder.save(result.state)
            result.events = [event.to_dict() for event in self.recorder.events]
            return result

        await self.strategy.on_evaluation(
            self._runtime_context(),
            result.state,
            feedback,
        )

        result.trajectory_path = self.recorder.save(result.state)
        result.events = [event.to_dict() for event in self.recorder.events]
        return result

    def strategy_checkpoint_state(self) -> dict[str, Any]:
        """Expose optional strategy-local state to the shared run checkpoint."""

        return self.strategy.checkpoint_state()

    async def restore_strategy_checkpoint(self, payload: dict[str, Any]) -> None:
        await self.strategy.restore_checkpoint(self._runtime_context(), payload)

    def _runtime_context(self) -> RuntimeContext:
        return RuntimeContext(
            agent=self.agent,
            providers=self.providers,
            assembler=self.assembler,
            recorder=self.recorder,
            max_turns=self.max_turns,
            memory_policy=self.memory_policy,
            skill_policy=self.skill_policy,
            memory_top_k=self.memory_top_k,
            skill_top_k=self.skill_top_k,
            mode=self.mode,
            trace_actions=self.trace_actions,
            protocol_repair_enabled=self.protocol_repair_enabled,
            protocol_repair_max_attempts=self.protocol_repair_max_attempts,
        )

    async def _extract_structured_answer(
        self,
        ctx: RuntimeContext,
        state: EpisodeState,
    ) -> None:
        """Extract an MCQ label while preserving the solver's original answer."""

        if state.final_answer is None:
            extraction = AnswerExtractionResult(
                status="ok",
                answer=None,
                error="Episode produced no final answer.",
            )
        else:
            self.trace_action(state, "answer_extract.start")
            case_metadata = dict(state.case.get("metadata") or {})
            allowed_labels = case_metadata.get("allowed_labels")
            extraction = await self.answer_extractor.extract(
                question=ctx.question_key(state),
                candidate_response=state.final_answer,
                allowed_labels=allowed_labels,
            )

        payload = extraction.to_dict()

        state.metadata["answer_extraction"] = payload
        state.metadata["structured_answer"] = extraction.answer
        self.recorder.record(
            state.episode_id,
            state.turn_id,
            "answer_extracted",
            payload,
        )
        self.trace_action(
            state,
            "answer_extract.done",
            status=extraction.status,
            answer=extraction.answer,
        )

    def _default_strategy(self, mode: str) -> BaseStrategy:
        return ReactStrategy()
