from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from amf.events import utc_now_iso
from amf.llm import BaseLLMClient
from amf.memory.item import KnowledgeItem
from amf.memory.provenance import source_id_for_case
from amf.prompt.epoch_memory import EpochPromptBundle, replace_prompt_placeholders
from amf.state import EpisodeState
from amf.memory.window_validation import validate_window, combination_error
from amf.memory.output_schema import response_format

@dataclass(slots=True)
class MemoryWriteAction:
    """One validated database operation returned by the relation judge.

    ``model_target_ids`` are the anonymous labels shown to the model.  The
    runner/store only receives ``target_ids`` after they have been mapped back
    to real version IDs.
    """

    action: str
    items: list[KnowledgeItem] = field(default_factory=list)
    target_ids: list[str] = field(default_factory=list)
    model_target_ids: list[str] = field(default_factory=list)

    candidate_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "items": [item.to_dict() for item in self.items],
            "target_ids": list(self.target_ids),
            "model_target_ids": list(self.model_target_ids),
            "candidate_ids": list(self.candidate_ids),
            "reason": self.reason,
        }

@dataclass(slots=True)
class MemoryWriteDecision:
    action: str
    items: list[KnowledgeItem] = field(default_factory=list)
    target_id: str | None = None
    target_ids: list[str] = field(default_factory=list)

    model_target_ids: list[str] = field(default_factory=list)
    target_id_map: dict[str, str] = field(default_factory=dict)
    relation: str | None = None
    reason: str | None = None
    candidate_memory: dict[str, Any] | None = None
    candidate_raw_output: str = ""
    raw_output: str = ""
    error: str | None = None
    actions: list[MemoryWriteAction] = field(default_factory=list)
    window_prompt: str = ""
    window_attempts: list[dict[str, Any]] = field(default_factory=list)
    action_prompt: str = ""
    action_attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "action": self.action,
            "items": [item.to_dict() for item in self.items],
            "target_id": self.target_id,
            "target_ids": self.target_ids,
            "model_target_ids": self.model_target_ids,
            "target_id_map": self.target_id_map,
            "reason": self.reason,
            "candidate_memory": self.candidate_memory,
            "candidate_raw_output": self.candidate_raw_output,
            "raw_output": self.raw_output,
            "error": self.error,
            "actions": [action.to_dict() for action in self.actions],
        }

        if self.relation is not None:
            payload["relation"] = self.relation
        if self.window_prompt:
            payload["window_prompt"] = self.window_prompt
            payload["window_attempts"] = self.window_attempts
        if self.action_prompt:
            payload["action_prompt"] = self.action_prompt
            payload["action_attempts"] = self.action_attempts
        return payload

class LLMMemoryWriter:
    """Base interface for the Epoch memory writer."""

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        method: str = "unknown",
        **_: Any,
    ) -> None:
        self.method = method

    async def write(self, *_: Any, **__: Any) -> MemoryWriteDecision:
        raise NotImplementedError(f"Memory writer is not implemented for {self.method}.")

class EpochMemoryWriter(LLMMemoryWriter):
    """Epoch-specific writer used after rubric evaluation at a batch boundary."""

    def __init__(
        self,
        llm: BaseLLMClient,
        method: str,
        max_retries: int = 3,
        strict_json: bool = False,
        use_json_response_format: bool = False,
        dataset_name: str = "healthbench",
        prompt_root: str | None = None,
        prompt_bundle: EpochPromptBundle | None = None,
        prompt_dataset: str | None = None,
    ) -> None:
        super().__init__(llm=llm, method=method)
        self.llm = llm
        self.method = method
        self.max_retries = max_retries
        self.strict_json = strict_json
        self.use_json_response_format = use_json_response_format

        self.prompt_bundle = prompt_bundle or EpochPromptBundle.for_dataset(
            prompt_dataset or dataset_name,
            prompt_root=prompt_root,
        )

    def fork(self) -> "EpochMemoryWriter":
        """Give one concurrent candidate call request-local LLM audit state."""

        return EpochMemoryWriter(
            self.llm.fork(),
            method=self.method,
            max_retries=self.max_retries,
            strict_json=self.strict_json,
            use_json_response_format=self.use_json_response_format,
            prompt_bundle=self.prompt_bundle,
        )

    async def write(
        self,
        state: EpisodeState,
        events: list[dict[str, Any]],
        feedback: dict[str, Any] | None = None,
        candidate_memories: list[dict[str, Any]] | None = None,
        hidden_context: dict[str, Any] | None = None,
        information: list[dict[str, Any]] | None = None,
    ) -> MemoryWriteDecision:
        del events, hidden_context
        feedback = feedback or {}

        candidate_memory, candidate_raw_output, candidate_error = (
            await self.generate_candidate(state, feedback, information=information)
        )
        if candidate_memory is None:
            return self.invalid_candidate_decision(
                candidate_raw_output,
                candidate_error,
            )
        return await self.decide_candidate(
            state,
            feedback,
            candidate_memory=candidate_memory,
            candidate_raw_output=candidate_raw_output,
            candidate_memories=candidate_memories or [],
        )

    async def generate_candidate(
        self,
        state: EpisodeState,
        feedback: dict[str, Any] | None = None,
        *,
        information: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, str] | None, str, str | None]:
        """Stage 1: distill the evaluated case, optionally assessing Information."""

        feedback = feedback or {}

        records = (
            information
            if information is not None
            else self._information_from_state(state)
        )

        candidate_prompt = self._build_candidate_prompt(
            state,
            feedback,
            information=records,
        )
        candidate_raw_output = ""
        candidate_memory: dict[str, str] | None = None
        candidate_error: str | None = None
        for _ in range(max(self.max_retries, 1)):
            candidate_raw_output = await self._generate_json(
                candidate_prompt, schema_kind=(
                    "candidate_with_information" if records else "candidate"))
            candidate_memory, candidate_error = parse_candidate_memory_json(
                candidate_raw_output
            )
            if candidate_memory is not None:

                usage = parse_information_usage(
                    candidate_raw_output,
                    records,
                )
                if records:
                    if usage[1] is not None:
                        candidate_memory = None
                        candidate_error = usage[1]
                    else:
                        state.metadata["memory_information_usage"] = usage[0]
                break

        return candidate_memory, candidate_raw_output, candidate_error

    def invalid_candidate_decision(
        self,
        raw_output: str,
        error: str | None,
    ) -> MemoryWriteDecision:
        """Convert an invalid Stage-1 result into the configured failure policy."""

        if self.strict_json:
            raise ValueError(f"Invalid candidate memory output: {raw_output}")
        return MemoryWriteDecision(
            action="ignore",
            reason="Invalid candidate memory output.",
            candidate_raw_output=raw_output,
            error=error or "invalid candidate memory output",
        )

    async def decide_candidate(
        self,
        state: EpisodeState,
        feedback: dict[str, Any] | None,
        *,
        candidate_memory: dict[str, str],
        candidate_raw_output: str,
        candidate_memories: list[dict[str, Any]],
    ) -> MemoryWriteDecision:
        """Stage 2: add directly or compare the candidate with filtered references."""

        feedback = feedback or {}
        state.metadata["candidate_applicable_conditions"] = str(
            candidate_memory.get("applicable_conditions") or ""
        ).strip()
        if not candidate_memories:

            item = self._item_from_summary_payload(candidate_memory, state, feedback)
            if item is None:
                return self.invalid_candidate_decision(
                    candidate_raw_output,
                    "candidate summary could not be converted to a repository item",
                )
            operation = MemoryWriteAction(
                action="add",
                items=[item],
                reason="No relevant existing summary passed retrieval filtering.",
            )
            return MemoryWriteDecision(
                action="add",
                items=[item],
                actions=[operation],
                candidate_memory=candidate_memory,
                candidate_raw_output=candidate_raw_output,
            )

        prompt, target_id_map = self._build_relation_prompt(
            state=state,
            candidate_memory=candidate_memory,
            candidate_memories=candidate_memories or [],
        )
        raw_output = ""
        last_error: str | None = None
        messages = [{"role": "user", "content": prompt}]
        attempts: list[dict[str, Any]] = []
        for _ in range(max(self.max_retries, 1)):
            request_messages = [dict(message) for message in messages]
            raw_output = await self._generate_json(prompt, messages=request_messages)
            parsed = parse_memory_writer_json(raw_output)
            decision = self._decision_from_payload(
                parsed,
                state,
                feedback,
                raw_output,
                target_id_map=target_id_map,
            )
            attempts.append({
                "messages": request_messages,
                "raw_output": raw_output,
                "validation_errors": [decision.error] if decision.action == "invalid" else [],
            })
            if decision.action != "invalid":
                decision.candidate_memory = candidate_memory
                decision.candidate_raw_output = candidate_raw_output
                decision.action_prompt = prompt
                decision.action_attempts = attempts
                return decision
            last_error = decision.error

            history = set(state.metadata.get("memory_history_ids") or [])
            historical_labels = [label for label, real_id in target_id_map.items() if real_id in history]
            reminders = []
            if historical_labels:
                reminders.append(
                    "Existing summaries " + ", ".join(historical_labels)
                    + " come from the current question. The current temporary summary cannot independently CONFIRM them.")
            if state.metadata.get("memory_relation_route") == "history_only_single":
                reminders.append("This route permits exactly one action: keep or update.")
            repair = replace_prompt_placeholders(
                self.prompt_bundle.read("action_repair.txt"),
                VALIDATION_ERRORS=last_error or "Invalid JSON action structure.",
                ROUTE_REMINDER="\n".join(reminders),
            )
            attempts[-1]["repair_prompt"] = repair
            messages.extend([{"role": "assistant", "content": raw_output},
                             {"role": "user", "content": repair}])

        if self.strict_json:
            raise ValueError(f"Invalid memory writer output: {raw_output}")
        return MemoryWriteDecision(
            action="ignore",
            reason="Invalid memory writer output.",
            candidate_memory=candidate_memory,
            candidate_raw_output=candidate_raw_output,
            raw_output=raw_output,
            error=last_error or "invalid writer output",
            action_prompt=prompt,
            action_attempts=attempts,
        )

    async def decide_window(
        self,
        candidates: list[dict[str, Any]],
        candidate_memories: list[dict[str, Any]],
    ) -> MemoryWriteDecision:
        """Jointly resolve one evidence window in a single Stage-2 call.

        Stage 1 remains independent for every case.  This method only receives
        the resulting temporary summaries, their questions/usage assessments,
        and the deduplicated union of old summaries retrieved for the window.
        """

        if not candidates:
            return self._invalid_decision("", "evidence window has no candidates")

        for item in candidate_memories:
            sources = item.get("content_source_ids") or dict(item.get("metadata") or {}).get("content_source_ids")
            if not isinstance(sources, list) or not sources or not all(isinstance(s, str) and s.strip() for s in sources):
                return self._invalid_decision("", "existing summary has missing or invalid content sources")
        for candidate in candidates:
            state = candidate["state"]
            if not (state.metadata.get("source_id") or source_id_for_case(state.case)):
                return self._invalid_decision("", "candidate has missing source question")

        prompt, target_id_map, history_by_candidate = self._build_window_prompt(
            candidates,
            candidate_memories,
        )
        allowed_candidate_ids = {
            str(candidate.get("id") or "").strip()
            for candidate in candidates
            if str(candidate.get("id") or "").strip()
        }
        representative_state = candidates[0]["state"]
        representative_feedback = dict(candidates[0].get("feedback") or {})
        raw_output = ""
        last_error: str | None = None
        messages = [{"role": "user", "content": prompt}]
        attempts: list[dict[str, Any]] = []
        for _ in range(max(self.max_retries, 1)):
            request_messages = [dict(m) for m in messages]
            raw_output = await self._generate_json(
                prompt, schema_kind="window_actions", messages=request_messages)
            payload = parse_memory_writer_json(raw_output)
            errors, hints = validate_window(payload, allowed_candidate_ids, target_id_map, history_by_candidate)
            attempts.append({"messages": request_messages, "raw_output": raw_output,
                             "validation_errors": errors, "repair_hints": hints})
            if errors:
                last_error = "\n".join(errors)
                messages.extend([
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": replace_prompt_placeholders(
                        self.prompt_bundle.window_repair,
                        VALIDATION_ERRORS="\n".join(f"- {hint}" for hint in hints))},
                ])
                continue
            decision = self._decision_from_payload(
                payload,
                representative_state,
                representative_feedback,
                raw_output,
                target_id_map=target_id_map,
                allow_legacy=False,
                allowed_candidate_ids=allowed_candidate_ids,
                history_ids_by_candidate=history_by_candidate,
            )
            if decision.action != "invalid":
                decision.window_prompt = prompt
                decision.window_attempts = attempts
                decision.candidate_memory = {
                    "window_candidates": [
                        {"id": candidate.get("id"), **dict(candidate.get("memory") or {})}
                        for candidate in candidates
                    ]
                }
                return decision
            last_error = decision.error
            attempts[-1]["validation_errors"] = [last_error]
            messages.extend([
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": replace_prompt_placeholders(
                    self.prompt_bundle.window_repair,
                    VALIDATION_ERRORS="- The response does not follow the required JSON structure.")},
            ])

        return MemoryWriteDecision(
            action="ignore",
            reason="Invalid evidence-window writer output.",
            candidate_memory={
                "window_candidates": [
                    {"id": candidate.get("id"), **dict(candidate.get("memory") or {})}
                    for candidate in candidates
                ]
            },
            raw_output=raw_output,
            error=last_error or "invalid evidence-window writer output",
            window_prompt=prompt,
            window_attempts=attempts,
        )

    async def _generate_json(self, prompt: str, *, schema_kind: str = "actions", **kwargs: Any) -> str:
        if self.use_json_response_format:
            kwargs["response_format"] = response_format(schema_kind)
        return await self.llm.generate(prompt, **kwargs)

    def _build_candidate_prompt(
        self,
        state: EpisodeState,
        feedback: dict[str, Any],
        *,
        information: list[dict[str, Any]] | None = None,
    ) -> str:
        question = self._question_text(state)
        rubric_groups = dict(feedback.get("rubrics") or {})
        evaluation_feedback = feedback.get("evaluation_feedback")
        if not isinstance(evaluation_feedback, dict):
            evaluation_feedback = None
        records = (
            information
            if information is not None
            else self._information_from_state(state)
        )
        state.metadata["memory_information_id_map"] = {
            f"Information {index}": str(item.get("id"))
            for index, item in enumerate(records, start=1)
            if item.get("id")
        }
        current_case = self._render_stage1_case(
            question=question,
            response=state.final_answer or "",
            rubric_groups=rubric_groups,
            evaluation_feedback=evaluation_feedback,
            information=records,
        )
        template = (
            self.prompt_bundle.stage1_with_information
            if records
            else self.prompt_bundle.stage1_no_information
        )

        return replace_prompt_placeholders(template, CURRENT_CASE=current_case)

    def _render_stage1_case(
        self,
        *,
        question: str,
        response: str,
        rubric_groups: dict[str, Any],
        evaluation_feedback: dict[str, Any] | None,
        information: list[dict[str, Any]],
    ) -> str:
        sections = ["### Input", "#### Question", question]
        if information:
            sections.append("#### Information")
            for index, item in enumerate(information, start=1):
                sections.extend(
                    [
                        f"##### Information {index}",
                        str(item.get("summary") or item.get("content") or "").strip(),
                        "",
                        "Applicable when:",
                        str(
                            item.get("applicable_conditions")
                            or item.get("condition")
                            or ""
                        ).strip(),
                    ]
                )
        sections.extend(["#### Response", response])
        if evaluation_feedback is not None:
            sections.extend(
                [
                    "#### Evaluation Feedback",
                    json.dumps(evaluation_feedback, ensure_ascii=False, indent=2),
                ]
            )
        else:
            sections.extend(
                [
                    "#### Evaluation Criteria",
                    self._format_rubric_groups(rubric_groups),
                ]
            )
        return "\n\n".join(sections)

    def _information_from_state(self, state: EpisodeState) -> list[dict[str, Any]]:
        """Project answer-time memory results to anonymous Information blocks."""

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for context in state.retrieved_contexts:
            if context.get("source") != "memory":
                continue
            for raw in context.get("results") or []:
                if not isinstance(raw, dict):
                    continue
                real_id = str(raw.get("id") or "").strip()
                if not real_id or real_id in seen:
                    continue
                seen.add(real_id)
                metadata = dict(raw.get("metadata") or {})
                records.append(
                    {
                        "id": real_id,
                        "summary": str(raw.get("content") or "").strip(),
                        "condition": str(raw.get("condition") or "").strip(),
                        "applicable_conditions": str(raw.get("condition") or "").strip(),
                        "lifecycle_status": metadata.get(
                            "lifecycle_status", "active"
                        ),
                    }
                )
        state.metadata["memory_information_id_map"] = {
            f"Information {index}": str(item["id"])
            for index, item in enumerate(records, start=1)
        }
        return records

    def _build_relation_prompt(
        self,
        *,
        state: EpisodeState,
        candidate_memory: dict[str, str],
        candidate_memories: list[dict[str, Any]],
    ) -> tuple[str, dict[str, str]]:
        """Render one of the README route-specific prompts with fake IDs."""
        references: list[dict[str, str]] = []
        target_id_map: dict[str, str] = {}
        current_source = str(
            state.metadata.get("source_id")
            or source_id_for_case(state.case)
            or ""
        )
        history: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []
        for item in candidate_memories:
            real_id = str(item.get("id") or "").strip()
            if not real_id:
                continue
            source_values = (
                item.get("content_source_ids")
                or dict(item.get("metadata") or {}).get("content_source_ids")
                or []
            )
            source_values = {str(value) for value in source_values}
            is_history = bool(item.get("is_history")) or (
                bool(current_source) and current_source in source_values
            )
            (history if is_history else other).append(item)

        ordered = [*history, *other]
        for index, item in enumerate(ordered, start=1):
            real_id = str(item.get("id") or "").strip()
            fake_id = str(index)
            target_id_map[fake_id] = real_id
            references.append(
                {
                    "id": fake_id,
                    "summary": str(item.get("summary") or item.get("content") or "").strip(),
                    "applicable_conditions": str(
                        item.get("applicable_conditions") or item.get("condition") or ""
                    ).strip(),
                    "usage_assessment": str(
                        item.get("usage_assessment")
                        or dict(item.get("metadata") or {}).get("usage_assessment")
                        or ""
                    ).strip(),
                }
            )
        history_refs = references[: len(history)]
        other_refs = references[len(history) :]
        state.metadata["memory_history_ids"] = [
            str(item.get("id")) for item in history if item.get("id")
        ]
        if history and not other and len(history) == 1:
            state.metadata["memory_relation_route"] = "history_only_single"
        elif history and not other:
            state.metadata["memory_relation_route"] = "history_only_multiple"
        elif history and other:
            state.metadata["memory_relation_route"] = "history_and_other"
        else:
            state.metadata["memory_relation_route"] = "other_only"
        temporary = self._render_temporary_summary(candidate_memory)
        if history and not other and len(history) == 1:
            prompt = replace_prompt_placeholders(
                self.prompt_bundle.history_compare,
                TEMPORARY_SUMMARY=temporary,
                PREVIOUS_VERSION=self._render_references(
                    history_refs,
                    historical=True,
                ),
            )

            prompt = prompt.replace('"previous_summary_id"', '"1"')
        else:
            route_shot = (
                self.prompt_bundle.action_example_2
                if history and other
                else self.prompt_bundle.action_example_1
                if other
                else ""
            )
            blocks = [temporary]
            if history:
                blocks.append(
                    self._render_references(history_refs, historical=True)
                )
            if other:
                blocks.append(self._render_references(other_refs, historical=False))
            prompt = replace_prompt_placeholders(
                self.prompt_bundle.action_common,
                ROUTE_SHOT=route_shot,
                CURRENT_CASE="\n\n".join(blocks),
            )
        state.metadata["writer_summary_prompt_id_map"] = dict(target_id_map)
        return prompt, target_id_map

    def _build_window_prompt(
        self,
        candidates: list[dict[str, Any]],
        candidate_memories: list[dict[str, Any]],
    ) -> tuple[str, dict[str, str], dict[str, set[str]]]:
        """Render independent candidate/old-summary blocks with anonymous IDs."""

        candidate_blocks: list[str] = ["## Temporary Summaries"]
        candidate_sources: dict[str, str] = {}
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "").strip()
            state = candidate["state"]
            memory = dict(candidate.get("memory") or {})
            source_id = str(
                state.metadata.get("source_id")
                or source_id_for_case(state.case)
                or ""
            )
            candidate_sources[candidate_id] = source_id
            candidate_blocks.extend(
                [
                    "",
                    f"### Temporary Summary {candidate_id}",
                    "",
                    f"Temporary ID: `{candidate_id}`",
                    "",
                    "Question:",
                    "",
                    str(candidate.get("question") or self._question_text(state)).strip(),
                    "",
                    "Summary:",
                    "",
                    str(memory.get("summary") or "").strip(),
                    "",
                    "Applicable Conditions:",
                    "",
                    str(memory.get("applicable_conditions") or "").strip(),
                ]
            )
            usage = candidate.get("information_usage")
            if isinstance(usage, dict) and usage:
                candidate_blocks.extend(
                    [
                        "",
                        "Overall Usage Assessment:",
                        "",
                        str(usage.get("overall") or "").strip(),
                    ]
                )

        target_id_map: dict[str, str] = {}
        history_by_candidate: dict[str, set[str]] = {
            candidate_id: set() for candidate_id in candidate_sources
        }
        existing_blocks: list[str] = []
        if candidate_memories:
            existing_blocks = [
                "## Existing Summaries",
                "",
                "The following summaries were retrieved as related to at least one temporary summary.",
                'Under each existing summary, the "Same-question temporary IDs" field lists candidates that share a source question with that summary. They may be used for version comparison, but do not count as independent support for confirming that summary.',
            ]
        for index, item in enumerate(candidate_memories, start=1):
            real_id = str(item.get("id") or "").strip()
            if not real_id:
                continue
            fake_id = str(index)
            target_id_map[fake_id] = real_id
            source_values = {
                str(value)
                for value in (
                    item.get("content_source_ids")
                    or dict(item.get("metadata") or {}).get("content_source_ids")
                    or []
                )
            }
            historical_for = [
                candidate_id
                for candidate_id, source_id in candidate_sources.items()
                if source_id and source_id in source_values
            ]
            for candidate_id in historical_for:
                history_by_candidate[candidate_id].add(real_id)
            retrieved_for = [
                str(value)
                for value in item.get("retrieved_for_candidate_ids") or []
                if str(value)
            ]
            assessments = item.get("usage_assessments")
            existing_blocks.extend(
                [
                    "",
                    f"### Existing Summary {fake_id}",
                    "",
                    f"ID: `{fake_id}`",
                    "",
                    "Retrieved for temporary IDs:",
                    "",
                    ", ".join(retrieved_for) if retrieved_for else "Not recorded",
                    "",
                    "Same-question temporary IDs (not independent support):",
                    "",
                    ", ".join(historical_for) if historical_for else "None",
                    "",
                    "Summary:",
                    "",
                    str(item.get("summary") or item.get("content") or "").strip(),
                    "",
                    "Applicable Conditions:",
                    "",
                    str(item.get("applicable_conditions") or item.get("condition") or "").strip(),
                ]
            )
            if isinstance(assessments, dict) and assessments:
                existing_blocks.extend(
                    [
                        "",
                        "Usage Assessments by Temporary ID:",
                        "",
                        json.dumps(assessments, ensure_ascii=False),
                    ]
                )

        prompt = replace_prompt_placeholders(
            self.prompt_bundle.window_action,
            TEMPORARY_SUMMARIES="\n".join(candidate_blocks),
            EXISTING_SUMMARIES="\n".join(existing_blocks),
        )

        reminders = [
            f"Temporary Summary {candidate_id} and Existing Summary {fake_id} "
            f"come from the same question. Do not use {candidate_id} as independent "
            f"support to CONFIRM Summary {fake_id}."
            for fake_id, real_id in target_id_map.items()
            for candidate_id, historical_ids in history_by_candidate.items()
            if real_id in historical_ids
        ]
        if reminders:
            prompt += "\n\nReminder:\n" + "\n".join(reminders)
        return prompt, target_id_map, history_by_candidate

    def _render_temporary_summary(self, summary: dict[str, str]) -> str:
        return (
            "## Temporary Summary\n\n"
            "Summary:\n\n"
            f"{str(summary.get('summary') or '').strip()}\n\n"
            "Applicable Conditions:\n\n"
            f"{str(summary.get('applicable_conditions') or '').strip()}"
        )

    def _render_references(
        self,
        references: list[dict[str, str]],
        *,
        historical: bool,
    ) -> str:
        if not references:
            return ""
        if historical:
            lines = [
                "## Previous Versions Produced from the Current Question",
                "",
                "The following summaries were produced from the current question in previous rounds.",
                "They are historical versions and may be used for version comparison.",
                "They cannot independently support a `confirm` action.",
            ]
        else:
            lines = [
                "## Summaries from Other Questions",
                "",
                "The following summaries were produced from other questions.",
                "They may provide independent support and may be considered for `confirm`.",
            ]
        for ref in references:
            lines.extend(
                [
                    "",
                    f"### Summary {ref['id']}",
                    "",
                    f"ID: `{ref['id']}`",
                    "",
                    "Summary:",
                    "",
                    ref["summary"],
                    "",
                    "Applicable Conditions:",
                    "",
                    ref["applicable_conditions"],
                ]
            )
            if ref.get("usage_assessment"):
                lines.extend(
                    [
                        "",
                        "Usage Assessment:",
                        "",
                        ref["usage_assessment"],
                    ]
                )
        return "\n".join(lines)

    def _question_text(self, state: EpisodeState) -> str:
        messages = state.case.get("messages")
        if isinstance(messages, list) and messages:
            return "\n\n".join(
                f"{str(message.get('role') or 'user').capitalize()}: "
                f"{str(message.get('content') or '')}"
                for message in messages
                if isinstance(message, dict)
            )
        return str(state.case.get("question") or "")

    def _format_rubric_groups(self, groups: dict[str, Any]) -> str:
        sections: list[str] = []
        for key, title in (
            ("positive", "#### Positive Rubrics"),
            ("negative", "#### Negative Rubrics"),
        ):
            lines = [title]
            rubrics = groups.get(key)
            if not isinstance(rubrics, list) or not rubrics:
                lines.append("\n* None")
            else:
                for rubric in rubrics:
                    if not isinstance(rubric, dict):
                        continue
                    criterion = str(rubric.get("criterion") or "").strip()
                    if not criterion:
                        continue
                    observed = "true" if rubric.get("observed") is True else "false"
                    lines.append(f"\n* {criterion} — **Observed: {observed}**")
            sections.append("".join(lines))
        return "\n\n".join(sections)

    def _decision_from_payload(
        self,
        payload: dict[str, Any],
        state: EpisodeState,
        feedback: dict[str, Any] | None,
        raw_output: str,
        target_id_map: dict[str, str] | None = None,
        *,
        allow_legacy: bool = True,
        allowed_candidate_ids: set[str] | None = None,
        history_ids_by_candidate: dict[str, set[str]] | None = None,
    ) -> MemoryWriteDecision:
        target_id_map = dict(target_id_map or {})
        raw_actions = payload.get("actions")
        if raw_actions is None and allow_legacy:

            raw_actions = [
                {
                    "action": payload.get("action"),
                    "target_ids": payload.get("target_ids"),
                    "target_id": payload.get("target_id"),
                    "reason": payload.get("reason"),
                    "summaries": payload.get(
                        "summaries", payload.get("memories", [])
                    ),
                    "memory": payload.get("memory"),
                }
            ]
        if not isinstance(raw_actions, list) or not raw_actions:
            return self._invalid_decision(raw_output, "writer output must contain actions")

        parsed_actions: list[MemoryWriteAction] = []
        covered_candidate_ids: set[str] = set()
        target_actions: dict[str, list[str]] = {}
        history_ids = {
            str(value) for value in state.metadata.get("memory_history_ids") or []
        }
        relation_route = str(state.metadata.get("memory_relation_route") or "")
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                return self._invalid_decision(raw_output, "each action must be an object")
            action = str(raw_action.get("action") or "").strip().lower()
            aliases = {
                "ignore": "reject",
                "direct_conflict": "dispute",
                "too_specific": "reject",
            }
            action = aliases.get(action, action)
            if action not in {
                "add",
                "keep",
                "confirm",
                "update",
                "merge",
                "split",
                "dispute",
                "reject",
            }:
                return self._invalid_decision(
                    raw_output, f"unsupported action: {action or '<empty>'}"
                )

            raw_candidate_ids = raw_action.get("candidate_ids", [])
            if isinstance(raw_candidate_ids, str):
                raw_candidate_ids = [raw_candidate_ids]
            if not isinstance(raw_candidate_ids, list):
                return self._invalid_decision(
                    raw_output, f"{action} candidate_ids must be an array"
                )
            candidate_ids = [
                str(value).strip()
                for value in raw_candidate_ids
                if str(value).strip()
            ]
            if allowed_candidate_ids is not None:
                candidate_ids = list(dict.fromkeys(candidate_ids))
            if len(candidate_ids) != len(set(candidate_ids)):
                return self._invalid_decision(
                    raw_output, f"{action} contains duplicate candidate_ids"
                )
            if allowed_candidate_ids is not None:
                if not candidate_ids:
                    return self._invalid_decision(
                        raw_output, f"{action} requires at least one candidate_id"
                    )
                unknown_candidates = [
                    value for value in candidate_ids if value not in allowed_candidate_ids
                ]
                if unknown_candidates:
                    return self._invalid_decision(
                        raw_output, f"unknown candidate_ids: {unknown_candidates}"
                    )
                covered_candidate_ids.update(candidate_ids)

            raw_targets = raw_action.get("target_ids")
            if raw_targets is None and raw_action.get("target_id") is not None:
                raw_targets = [raw_action.get("target_id")]
            if isinstance(raw_targets, str):
                raw_targets = [raw_targets]
            if not isinstance(raw_targets, list):
                return self._invalid_decision(
                    raw_output, f"{action} target_ids must be an array"
                )
            model_target_ids = [
                str(value).strip() for value in raw_targets if str(value).strip()
            ]
            if allowed_candidate_ids is not None:
                model_target_ids = list(dict.fromkeys(model_target_ids))
            if len(set(model_target_ids)) != len(model_target_ids):
                return self._invalid_decision(
                    raw_output, f"{action} contains duplicate target_ids"
                )
            unknown_ids = [
                value for value in model_target_ids if target_id_map and value not in target_id_map
            ]
            if unknown_ids:
                return self._invalid_decision(
                    raw_output, f"unknown model target_ids: {unknown_ids}"
                )
            target_ids = [
                target_id_map.get(value, value) for value in model_target_ids
            ]

            raw_summaries = raw_action.get("summaries")
            if raw_summaries is None:
                raw_summaries = raw_action.get("memories", [])
            if raw_summaries is None:
                raw_summaries = []
            if not isinstance(raw_summaries, list):
                return self._invalid_decision(
                    raw_output, f"{action} summaries must be an array"
                )
            if not raw_summaries and raw_action.get("memory") is not None:
                raw_summaries = [raw_action["memory"]]
            items: list[KnowledgeItem] = []
            for raw_summary in raw_summaries:
                if not isinstance(raw_summary, dict):
                    return self._invalid_decision(
                        raw_output, f"{action} summary must be an object"
                    )
                item = self._item_from_summary_payload(raw_summary, state, feedback)
                if item is None:

                    item = self._item_from_payload(raw_summary, state, feedback)
                if item is None:
                    return self._invalid_decision(
                        raw_output, f"{action} contains an invalid summary"
                    )
                items.append(item)

            constraint_error = self._validate_action_constraints(
                action,
                target_ids,
                items,
                history_ids=history_ids,
            )
            if (
                constraint_error is None
                and action == "confirm"
                and history_ids_by_candidate is not None
            ):
                for target_id in target_ids:
                    if all(
                        target_id in history_ids_by_candidate.get(candidate_id, set())
                        for candidate_id in candidate_ids
                    ):
                        constraint_error = (
                            "confirm requires at least one candidate from an independent question"
                        )
                        break
            if (
                constraint_error is None
                and relation_route == "history_only_single"
                and action not in {"keep", "update"}
            ):
                constraint_error = (
                    "the single-history route permits only keep or update"
                )
            if constraint_error:
                return self._invalid_decision(raw_output, constraint_error)
            for target in target_ids:
                previous = target_actions.get(target)
                if previous is not None and allowed_candidate_ids is None:
                    return self._invalid_decision(
                        raw_output,
                        f"target {next((label for label, real in target_id_map.items() if real == target), target)} is used more than once ({previous}, {action})",
                    )
                names = [*(previous or []), action]
                error = combination_error(names)
                if error:
                    return self._invalid_decision(raw_output, error)
                target_actions[target] = names
            parsed_actions.append(
                MemoryWriteAction(
                    action=action,
                    items=items,
                    target_ids=target_ids,
                    model_target_ids=model_target_ids,
                    candidate_ids=candidate_ids,
                    reason=str(raw_action.get("reason") or "").strip(),
                )
            )

        if (
            allowed_candidate_ids is not None
            and covered_candidate_ids != allowed_candidate_ids
        ):
            missing = sorted(allowed_candidate_ids - covered_candidate_ids)
            return self._invalid_decision(
                raw_output, f"every temporary summary must be handled; missing {missing}"
            )

        if relation_route == "history_only_single" and len(parsed_actions) != 1:
            return self._invalid_decision(
                raw_output,
                "the single-history route requires exactly one keep or update action",
            )

        first = parsed_actions[0]
        return MemoryWriteDecision(
            action=first.action,
            items=list(first.items),
            target_id=first.target_ids[0] if first.target_ids else None,
            target_ids=list(first.target_ids),
            model_target_ids=list(first.model_target_ids),
            target_id_map=target_id_map,

            relation=None,
            reason=first.reason,
            raw_output=raw_output,
            actions=parsed_actions,
        )

    def _validate_action_constraints(
        self,
        action: str,
        target_ids: list[str],
        items: list[KnowledgeItem],
        *,
        history_ids: set[str],
    ) -> str | None:
        if action == "add":
            return (
                "add requires no targets and exactly one summary"
                if target_ids or len(items) != 1
                else None
            )
        if action == "keep":
            return (
                "keep requires at least one target and no summaries"
                if not target_ids or items
                else None
            )
        if action == "confirm":
            if not target_ids or items:
                return "confirm requires targets and no summaries"
            if history_ids.intersection(target_ids):
                return "historical summaries cannot be confirmed"
            return None
        if action == "update":
            return (
                "update requires exactly one target and one summary"
                if len(target_ids) != 1 or len(items) != 1
                else None
            )
        if action == "merge":
            return (
                "merge requires at least two targets and one summary"
                if len(target_ids) < 2 or len(items) != 1
                else None
            )
        if action == "split":
            if not target_ids or len(items) < 2:
                return "split requires targets and at least two summaries"
            if any(not item.condition or not item.condition.strip() for item in items):
                return "split summaries must have explicit applicable conditions"
            return None
        if action == "dispute":
            return (
                "dispute requires targets and no summaries"
                if not target_ids or items
                else None
            )
        if action == "reject":
            return (
                "reject must not target or create summaries"
                if target_ids or items
                else None
            )
        return f"unsupported action: {action}"

    def _invalid_decision(self, raw_output: str, error: str) -> MemoryWriteDecision:
        return MemoryWriteDecision(action="invalid", raw_output=raw_output, error=error)

    def _item_from_payload(
        self,
        payload: dict[str, Any],
        state: EpisodeState,
        feedback: dict[str, Any] | None,
    ) -> KnowledgeItem | None:
        key = payload.get("key")
        content = payload.get("content")
        if not isinstance(key, str) or not key.strip():
            return None
        if not isinstance(content, str) or not content.strip():
            return None

        metadata = self._base_metadata(state, feedback)
        metadata.update(dict(payload.get("metadata") or {}))
        return KnowledgeItem.create(
            method=self.method,
            name=str(payload.get("name") or key).strip(),
            key=key.strip(),
            content=content.strip(),
            condition=(
                str(payload["condition"]).strip()
                if payload.get("condition") is not None
                else (
                    str(state.metadata.get("candidate_applicable_conditions") or "").strip()
                    or None
                )
            ),
            category=(
                str(payload["category"]).strip()
                if payload.get("category") is not None
                else None
            ),
            tools=[str(tool) for tool in payload.get("tools") or []],
            metadata=metadata,
        )

    def _item_from_summary_payload(
        self,
        payload: dict[str, Any],
        state: EpisodeState,
        feedback: dict[str, Any] | None,
    ) -> KnowledgeItem | None:
        """Convert the model-facing summary schema into the repository schema."""
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        content = summary.strip()
        condition_value = payload.get("applicable_conditions")
        condition = str(condition_value).strip() if condition_value is not None else None
        if not condition:
            return None
        key = " ".join(content.split())[:240]
        name = content.split(".", 1)[0].strip()[:120] or key[:120]
        return KnowledgeItem.create(
            method=self.method,
            name=name,
            key=key,
            content=content,
            condition=condition,
            category=None,
            tools=[],
            metadata=self._base_metadata(state, feedback),
        )

    def _base_metadata(
        self,
        state: EpisodeState,
        feedback: dict[str, Any] | None,
    ) -> dict[str, Any]:
        feedback = feedback or {}
        score = feedback.get("score")
        metrics = feedback.get("metrics")
        if score is None and isinstance(metrics, dict):
            score = metrics.get("overall_score")
        success = feedback.get("success")
        source_id = str(
            state.metadata.get("source_id")
            or source_id_for_case(state.case)
            or ""
        ).strip() or None
        return {
            "created_at": utc_now_iso(),
            "updated_at": None,
            "updated_from": None,
            "updated_to": None,

            "lifecycle_status": "provisional",
            "trajectory_id": state.episode_id,
            "question_id": state.case.get("case_id"),
            "source_id": source_id,
            "dataset": state.case.get("dataset"),
            "score": score if isinstance(score, int | float) else None,
            "success": success if isinstance(success, bool) else None,
            "feedback_summary": feedback.get("summary"),
        }

def parse_memory_writer_json(raw_output: str) -> dict[str, Any]:
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}

def parse_candidate_memory_json(
    raw_output: str,
) -> tuple[dict[str, str] | None, str | None]:
    """Parse the small Stage-1 schema before repository relation judging."""

    payload = parse_memory_writer_json(raw_output)
    summary = payload.get("summary")
    conditions = payload.get("applicable_conditions")
    if not isinstance(summary, str) or not summary.strip():
        return None, "candidate memory has no non-empty summary"
    if not isinstance(conditions, str) or not conditions.strip():
        return None, "candidate memory has no non-empty applicable_conditions"
    return {
        "summary": summary.strip(),
        "applicable_conditions": conditions.strip(),
    }, None

def parse_information_usage(
    raw_output: str,
    information: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    """Validate the per-Information usage assessment from Stage 1.

    The model-facing IDs are ephemeral ``Information 1`` labels.  The returned
    mapping is keyed by the stable repository ID, so later relation prompts and
    audit records never confuse a display number with a database key.
    """

    if not information:
        return {}, None
    payload = parse_memory_writer_json(raw_output)
    usage = payload.get("information_usage")
    if not isinstance(usage, dict):
        return {}, "information_usage is required when Information is provided"
    overall = usage.get("overall")
    if not isinstance(overall, str) or not overall.strip():
        return {}, "information_usage.overall must be a non-empty string"
    effects = usage.get("information_effects")
    if not isinstance(effects, list):
        return {}, "information_usage.information_effects must be an array"
    expected = {f"Information {index}" for index in range(1, len(information) + 1)}
    seen: set[str] = set()
    mapping = {
        str(index): str(item.get("id"))
        for index, item in enumerate(information, start=1)
        if item.get("id")
    }
    result: dict[str, Any] = {"overall": overall.strip(), "effects": {}}
    for effect in effects:
        if not isinstance(effect, dict):
            return {}, "each information effect must be an object"
        info_id = str(effect.get("information_id") or "").strip()
        analysis = effect.get("analysis")
        if info_id not in expected:
            return {}, f"unknown information_id: {info_id}"
        if info_id in seen:
            return {}, f"duplicate information_id: {info_id}"
        if not isinstance(analysis, str) or not analysis.strip():
            return {}, f"{info_id} analysis must be a non-empty string"
        seen.add(info_id)
        fake_number = info_id.rsplit(" ", 1)[-1]
        real_id = mapping.get(fake_number)
        if not real_id:
            return {}, f"{info_id} cannot be mapped to a repository ID"
        result["effects"][real_id] = analysis.strip()
    if seen != expected:
        return {}, "information_effects must contain exactly one entry per Information item"
    return result, None
