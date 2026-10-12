from __future__ import annotations

from amf.memory.output_schema import response_format

"""Epoch-end adjudication for sample-level Summary disputes."""

import json
import re
from dataclasses import dataclass
from typing import Any

from amf.llm import BaseLLMClient
from amf.prompt.epoch_memory import EpochPromptBundle, replace_prompt_placeholders

@dataclass(slots=True)
class ConflictDecision:
    decision: str
    reason: str
    summaries: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "target_id": "1",
            "reason": self.reason,
            "summaries": list(self.summaries),
        }

class EpochConflictResolver:
    """Resolve only queued disputes; never rewrite the whole repository."""

    DECISIONS = {"restore", "update", "split", "defer", "abandon"}

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        max_retries: int = 3,
        use_json_response_format: bool = True,
        min_abandon_sources: int = 2,
        max_input_memories: int = 5,
        prompt_bundle: EpochPromptBundle | None = None,
        prompt_root: str | None = None,
        prompt_dataset: str = "healthbench",
    ) -> None:
        self.llm = llm
        self.max_retries = max(1, int(max_retries))
        self.use_json_response_format = use_json_response_format
        self.min_abandon_sources = max(1, int(min_abandon_sources))

        self.max_input_memories = max(2, int(max_input_memories))
        self.prompt_bundle = prompt_bundle or EpochPromptBundle.for_dataset(
            prompt_dataset,
            prompt_root=prompt_root,
        )

    async def resolve_epoch(
        self,
        store: Any,
        *,
        epoch_id: int,
        run_id: str,
        trace: bool = False,
    ) -> list[dict[str, Any]]:
        groups = await store.pending_conflict_groups(through_epoch_id=epoch_id)
        if trace:
            print(
                f"[amf] epoch={epoch_id} action=conflict.scan groups={len(groups)}",
                flush=True,
            )
        results: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups, start=1):
            all_conflicts = list(group.get("conflicts") or [])
            max_visible_conflicts = self.max_input_memories - 1
            visible_conflicts = all_conflicts[:max_visible_conflicts]
            omitted_conflict_count = len(all_conflicts) - len(visible_conflicts)
            visible_group = {
                **group,
                "conflicts": visible_conflicts,
                "omitted_conflict_count": omitted_conflict_count,
            }
            abandon_allowed = int(group.get("conflict_source_count") or 0) >= (
                self.min_abandon_sources
            )
            prompt = self._build_prompt(
                visible_group,
                abandon_allowed=abandon_allowed,
            )
            raw_output = ""
            parse_error: str | None = None
            decision: ConflictDecision | None = None
            retry_prompt = prompt
            for _ in range(self.max_retries):
                kwargs = (
                    {"response_format": response_format("conflict")}
                    if self.use_json_response_format
                    else {}
                )
                raw_output = await self.llm.generate(retry_prompt, **kwargs)
                decision, parse_error = self._parse_decision(
                    raw_output,
                    abandon_allowed=abandon_allowed,
                )
                if decision is not None:
                    break
                retry_prompt = (
                    prompt
                    + "\n\nYour previous JSON response was invalid: "
                    + str(parse_error or "unknown schema error")
                    + "\nReturn a corrected JSON object using the required schema."
                )

            target = dict(group.get("target") or {})
            target_id = str(target.get("id") or "")
            audit_record: dict[str, Any] = {
                "group_index": group_index,
                "target_version_id": target_id,
                "target_summary": {
                    "summary": target.get("content"),
                    "applicable_conditions": target.get("condition"),
                },
                "conflicts": list(group.get("conflicts") or []),
                "support_source_count": int(group.get("support_source_count") or 0),
                "conflict_source_count": int(group.get("conflict_source_count") or 0),
                "visible_memory_count": 1 + len(visible_conflicts),
                "omitted_conflict_count": omitted_conflict_count,
                "abandon_allowed": abandon_allowed,
                "raw_output": raw_output,
            }
            if decision is None:
                audit_record.update(
                    {
                        "status": "error",
                        "error": parse_error or "invalid conflict resolver output",
                    }
                )
                if trace:
                    print(
                        f"[amf] epoch={epoch_id} action=conflict.invalid "
                        f"group={group_index}",
                        flush=True,
                    )
                results.append(audit_record)
                continue

            resolution_key = (
                f"{run_id}:epoch:{epoch_id}:conflict-target:{target_id}"
            )
            try:
                committed = await store.apply_conflict_resolution(
                    target_version_id=target_id,
                    decision=decision.decision,
                    summaries=decision.summaries,
                    reason=decision.reason,
                    epoch_id=epoch_id,
                    resolution_key=resolution_key,
                    min_abandon_sources=self.min_abandon_sources,
                )
            except (TypeError, ValueError, PermissionError) as exc:
                audit_record.update(
                    {
                        "status": "error",
                        "decision": decision.to_dict(),
                        "error": str(exc),
                    }
                )
                if trace:
                    print(
                        f"[amf] epoch={epoch_id} action=conflict.commit_error "
                        f"group={group_index}",
                        flush=True,
                    )
                results.append(audit_record)
                continue
            audit_record.update(
                {
                    "status": "committed",
                    "decision": decision.to_dict(),
                    "commit": committed,
                }
            )
            if trace:
                print(
                    f"[amf] epoch={epoch_id} action=conflict.resolve "
                    f"group={group_index} decision={decision.decision}",
                    flush=True,
                )
            results.append(audit_record)
        return results

    def _build_prompt(
        self,
        group: dict[str, Any],
        *,
        abandon_allowed: bool,
    ) -> str:
        target = dict(group.get("target") or {})
        sections = [
            "## Target Summary",
            "",
            "ID: `1`",
            "",
            "Summary:",
            "",
            str(target.get("content") or "").strip(),
            "",
            "Applicable Conditions:",
            "",
            str(target.get("condition") or "").strip(),
            "",
            "## Evidence Counts",
            "",
            f"Independent support sources: {int(group.get('support_source_count') or 0)}",
            f"Independent conflict sources: {int(group.get('conflict_source_count') or 0)}",
            f"Abandonment allowed: {'yes' if abandon_allowed else 'no'}",
        ]
        omitted_conflict_count = int(group.get("omitted_conflict_count") or 0)
        if omitted_conflict_count:
            sections.extend(
                [
                    f"Additional conflict items not shown: {omitted_conflict_count}",
                ]
            )
        for index, conflict in enumerate(group.get("conflicts") or [], start=1):
            snapshot = dict(conflict.get("candidate_summary") or {})
            sections.extend(
                [
                    "",
                    f"## Conflict Evidence {index}",
                    "",
                    "Temporary Summary:",
                    "",
                    str(snapshot.get("summary") or "Not recorded.").strip(),
                    "",
                    "Applicable Conditions:",
                    "",
                    str(
                        snapshot.get("applicable_conditions") or "Not recorded."
                    ).strip(),
                    "",
                    "Why it was disputed:",
                    "",
                    str(conflict.get("reason") or "Not recorded.").strip(),
                ]
            )
        return replace_prompt_placeholders(
            self.prompt_bundle.conflict_resolution,
            CURRENT_CONFLICT="\n".join(sections),
        )

    def _parse_decision(
        self,
        raw_output: str,
        *,
        abandon_allowed: bool,
    ) -> tuple[ConflictDecision | None, str | None]:
        text = str(raw_output or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "output must be a JSON object"
        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in self.DECISIONS:
            return None, f"unsupported decision: {decision or '<empty>'}"
        if str(payload.get("target_id") or "").strip() != "1":
            return None, "target_id must be 1"
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return None, "reason must be a non-empty string"
        summaries = payload.get("summaries")
        if not isinstance(summaries, list):
            return None, "summaries must be an array"
        normalized: list[dict[str, str]] = []
        for summary in summaries:
            if not isinstance(summary, dict):
                return None, "every summary must be an object"
            content = summary.get("summary")
            conditions = summary.get("applicable_conditions")
            if not isinstance(content, str) or not content.strip():
                return None, "every summary needs non-empty summary text"
            if not isinstance(conditions, str) or not conditions.strip():
                return None, "every summary needs non-empty applicable conditions"
            normalized.append(
                {
                    "summary": content.strip(),
                    "applicable_conditions": conditions.strip(),
                }
            )
        if decision in {"restore", "defer", "abandon"} and normalized:
            return None, f"{decision} must not return summaries"
        if decision == "update" and len(normalized) != 1:
            return None, "update requires exactly one summary"
        if decision == "split" and len(normalized) < 2:
            return None, "split requires at least two summaries"
        if decision == "abandon" and not abandon_allowed:
            return None, "abandon is not allowed with the current source count"
        return ConflictDecision(decision, reason.strip(), normalized), None
