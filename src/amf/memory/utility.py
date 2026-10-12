from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from amf.llm import BaseLLMClient
from amf.state import EpisodeState

@dataclass(slots=True)
class MemoryUtilityDecision:
    version_id: str
    label: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {
            "version_id": self.version_id,
            "label": self.label,
            "evidence": self.evidence,
        }

class EpochMemoryUtilityJudge:
    """Label whether each retrieved Memory helped the completed answer."""

    LABELS = {"helpful", "harmful", "neutral", "uncertain"}

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        max_retries: int = 2,
        use_json_response_format: bool = False,
    ) -> None:
        self.llm = llm
        self.max_retries = max(1, max_retries)
        self.use_json_response_format = use_json_response_format

    def fork(self) -> "EpochMemoryUtilityJudge":
        """Return a utility Judge with episode-local LLM request state."""

        return EpochMemoryUtilityJudge(
            self.llm.fork(),
            max_retries=self.max_retries,
            use_json_response_format=self.use_json_response_format,
        )

    async def judge(
        self,
        state: EpisodeState,
        *,
        evaluation: dict[str, Any],
        reference: dict[str, Any],
    ) -> list[MemoryUtilityDecision]:
        memories = self._retrieved_memories(state)
        if not memories:
            return []
        display_map = {
            str(index): str(memory["id"])
            for index, memory in enumerate(memories, start=1)
            if memory.get("id")
        }

        reference_rubrics = (
            list(reference.get("rubrics") or [])
            if isinstance(reference, dict)
            else []
        )
        payload = {
            "question": state.case.get("question"),
            "tool_calls": [
                {
                    "name": dict(entry.get("action") or {}).get("name"),
                    "arguments": dict(dict(entry.get("action") or {}).get("arguments") or {}),
                }
                for entry in state.actions
            ],
            "tool_observations": [
                {
                    "ok": dict(observation.get("observation") or {}).get("ok"),
                    "data": dict(observation.get("observation") or {}).get("data"),
                    "error": dict(observation.get("observation") or {}).get("error"),
                }
                for observation in state.observations
                if isinstance(observation, dict)
            ],
            "global_skill": state.metadata.get("global_skill"),
            "retrieved_information": [
                {
                    "id": f"Information {index}",
                    "content": memory.get("content"),
                    "condition": memory.get("condition"),
                    "status": dict(memory.get("metadata") or {}).get("lifecycle_status"),
                }
                for index, memory in enumerate(memories, start=1)
            ],
            "answer": state.final_answer,
            "previous_epoch_score": state.metadata.get("previous_epoch_score"),
            "current_score_change": state.metadata.get("score_change"),
            "rubric_evaluation": evaluation,
            "evaluation_criteria": reference_rubrics,
        }
        prompt = (
            "Judge whether each retrieved Information item affected the quality of the final "
            "answer. Use one label per Information item: helpful, harmful, neutral, or "
            "uncertain. helpful means it contributed relevant correct content or "
            "reasoning; harmful means it caused an error or omission; neutral means "
            "it was irrelevant or unused; uncertain means the evidence is "
            "insufficient. Do not assign scores. Return JSON only in this form: "
            '{"information_results":[{"id":"Information 1","label":"helpful",'
            '"evidence":"brief reason"}]}.\n\n'
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        for _ in range(self.max_retries):
            kwargs = (
                {"response_format": {"type": "json_object"}}
                if self.use_json_response_format
                else {}
            )
            raw = await self.llm.generate(prompt, **kwargs)
            parsed = self._parse(raw)
            decisions = self._decisions(parsed, memories, display_map)
            if decisions is not None:
                return decisions
        return [
            MemoryUtilityDecision(
                version_id=str(memory["id"]),
                label="uncertain",
                evidence="Memory utility judge returned invalid output.",
            )
            for memory in memories
            if memory.get("id")
        ]

    def _retrieved_memories(self, state: EpisodeState) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for context in state.retrieved_contexts:
            if context.get("source") != "memory":
                continue
            for memory in context.get("results") or []:
                if isinstance(memory, dict) and memory.get("id"):
                    by_id[str(memory["id"])] = memory
        return list(by_id.values())

    def _decisions(
        self,
        payload: dict[str, Any],
        memories: list[dict[str, Any]],
        display_map: dict[str, str] | None = None,
    ) -> list[MemoryUtilityDecision] | None:

        values = payload.get("information_results", payload.get("memory_results"))
        if not isinstance(values, list):
            return None
        display_map = dict(display_map or {})
        expected = set(display_map)
        results: dict[str, MemoryUtilityDecision] = {}
        for value in values:
            if not isinstance(value, dict):
                return None
            display_id = str(value.get("id") or "")
            label = str(value.get("label") or "").lower()
            if display_id not in expected or label not in self.LABELS:
                return None
            version_id = display_map[display_id]
            results[version_id] = MemoryUtilityDecision(
                version_id=version_id,
                label=label,
                evidence=str(value.get("evidence") or ""),
            )
        if set(results) != set(display_map.values()):
            return None
        return [results[str(memory["id"])] for memory in memories]

    @staticmethod
    def _parse(raw: Any) -> dict[str, Any]:
        text = str(raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
