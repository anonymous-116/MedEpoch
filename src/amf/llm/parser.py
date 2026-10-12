from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DecisionType = Literal["final", "tool_call", "invalid"]

@dataclass(slots=True)
class AgentDecision:
    type: DecisionType
    raw_output: str
    answer: str | None = None
    tool_call: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class OutputParser:
    """Parse JSON decisions and compact ReAct-style action outputs."""

    def parse(self, raw_output: str) -> AgentDecision:
        text = raw_output.strip()
        if not text:
            return AgentDecision(type="invalid", raw_output=raw_output, error="empty output")

        text, normalization_notes = self._normalize_protocol_text(text)

        if text.lower() == "i don't know.":
            return AgentDecision(type="final", raw_output=raw_output, answer=text)

        parsed = self._parse_json(text)
        if parsed is not None:
            if normalization_notes:
                self._add_normalization_notes(parsed, normalization_notes)
            return parsed

        parsed_react, segment_note = self._parse_react_segments(text, raw_output)
        if parsed_react is not None:
            if segment_note:
                normalization_notes.append(segment_note)
            if normalization_notes:
                self._add_normalization_notes(parsed_react, normalization_notes)
            return parsed_react

        return AgentDecision(type="invalid", raw_output=raw_output, error="unrecognized output")

    def _normalize_protocol_text(self, text: str) -> tuple[str, list[str]]:
        """Normalize known provider artifacts without hiding the raw output.

        The model protocol is line-oriented, but reasoning models sometimes
        emit ``</think>Thought:`` or concatenate multiple attempts.  Segment
        selection happens only after these control tokens become line breaks.
        """

        notes: list[str] = []
        normalized = text

        without_think_tags = re.sub(
            r"</?think>\s*",
            "\n",
            normalized,
            flags=re.IGNORECASE,
        )
        if without_think_tags != normalized:
            notes.append("think_tag_normalized")
            normalized = without_think_tags

        separated = re.sub(
            r"([}\]])\s*(?=Thought\s*:)",
            r"\1\n",
            normalized,
            flags=re.IGNORECASE,
        )
        if separated != normalized:
            notes.append("protocol_boundary_normalized")
            normalized = separated

        return normalized, notes

    def _parse_react_segments(
        self, text: str, raw_output: str
    ) -> tuple[AgentDecision | None, str | None]:
        """Use the first *complete* line-oriented Thought protocol block.

        A model may emit an internal Thought with no action before its actual
        ReAct response.  Splitting at each line-start Thought (the first may
        start at position zero) lets us skip incomplete blocks while retaining
        the first valid Action/Input or Final Answer block.  Legacy responses
        without a Thought still use the existing parser unchanged.
        """

        thoughts = list(re.finditer(r"(?im)^[ \t]*Thought\s*:", text))
        if not thoughts:
            return self._parse_react_text(text, raw_output), None

        for index, match in enumerate(thoughts):
            end = thoughts[index + 1].start() if index + 1 < len(thoughts) else len(text)
            segment = text[match.start() : end].strip()
            decision = self._parse_react_text(segment, raw_output)
            if decision is not None and decision.type in {"final", "tool_call"}:
                note = (
                    f"complete_thought_segment_{index + 1}_of_{len(thoughts)}"
                    if len(thoughts) > 1
                    else None
                )
                return decision, note

        return None, None

    @staticmethod
    def _add_normalization_notes(
        decision: AgentDecision, notes: list[str]
    ) -> None:
        existing = str(decision.metadata.get("parser_normalization") or "")
        values = [value for value in existing.split(";") if value]
        for note in notes:
            if note not in values:
                values.append(note)
        decision.metadata["parser_normalization"] = ";".join(values)

    def _parse_react_text(self, text: str, raw_output: str) -> AgentDecision | None:

        lines = [
            self._normalize_react_label(line.strip())
            for line in text.splitlines()
            if line.strip()
        ]
        if not lines:
            return None

        thought = self._extract_prefixed_text(lines, "thought:")
        final_index = self._find_prefixed_line(lines, "final answer:")
        action_index = self._find_prefixed_line(lines, "action:")
        if final_index is None and action_index is None:
            return None

        if final_index is not None and (
            action_index is None or final_index < action_index
        ):
            answer = lines[final_index].split(":", 1)[1].strip()
            trailing = lines[final_index + 1 :]
            if trailing:
                answer = "\n".join([answer, *trailing]).strip()
            return AgentDecision(
                type="final",
                raw_output=raw_output,
                answer=answer,
                metadata={"thought": thought} if thought else {},
            )

        assert action_index is not None
        action_text = lines[action_index].split(":", 1)[1].strip()
        if action_text.startswith("{"):
            decision = self._parse_legacy_action_json(action_text, raw_output)
            if thought:
                decision.metadata["thought"] = thought
            return decision

        tool_name = action_text
        if not tool_name:
            return AgentDecision(type="invalid", raw_output=raw_output, error="missing action name")

        input_alias_used = False
        for index in range(action_index + 1, len(lines)):
            if re.match(r"^input\s*:", lines[index], flags=re.IGNORECASE):
                lines[index] = re.sub(
                    r"^input\s*:", "Action Input:", lines[index], count=1, flags=re.IGNORECASE
                )
                input_alias_used = True

        arguments = None
        parse_error = "missing action input"
        for index, line in enumerate(lines[action_index + 1 :], start=action_index + 1):
            if line.lower().startswith("action input:"):
                arguments, parse_error = self._parse_action_input(lines, index)
                break
        if arguments is None:
            return AgentDecision(type="invalid", raw_output=raw_output, error=parse_error)
        if not isinstance(arguments, dict):
            return AgentDecision(type="invalid", raw_output=raw_output, error="action input must be a JSON object")
        metadata = {"thought": thought} if thought else {}
        if input_alias_used:
            metadata["parser_normalization"] = "input_alias"
        return AgentDecision(
            type="tool_call",
            raw_output=raw_output,
            tool_call={"name": tool_name, "arguments": arguments},
            metadata=metadata,
        )

    def _normalize_react_label(self, line: str) -> str:
        normalized = re.sub(r"^#{1,6}\s*", "", line).strip()
        label_pattern = r"(thought|action input|action|observation|final answer)"
        bold = re.match(
            rf"^\*\*{label_pattern}\s*:?\*\*\s*:?\s*(.*)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if bold:
            return f"{bold.group(1)}: {bold.group(2)}".rstrip()
        bare = re.match(
            rf"^{label_pattern}\s*$",
            normalized,
            flags=re.IGNORECASE,
        )
        if bare:
            return f"{bare.group(1)}:"
        return normalized

    def _find_prefixed_line(self, lines: list[str], prefix: str) -> int | None:
        lowered_prefix = prefix.lower()
        for index, line in enumerate(lines):
            if line.lower().startswith(lowered_prefix):
                return index
        return None

    def _extract_prefixed_text(self, lines: list[str], prefix: str) -> str:
        index = self._find_prefixed_line(lines, prefix)
        if index is None:
            return ""
        return lines[index].split(":", 1)[1].strip()

    def _parse_action_input(self, lines: list[str], input_index: int) -> tuple[Any | None, str]:
        first = lines[input_index].split(":", 1)[1].strip()
        following: list[str] = []
        for line in lines[input_index + 1 :]:
            if self._is_react_label(line):
                break
            following.append(line)

        candidates: list[str] = []
        if first and following:
            candidates.append("\n".join([first, *following]).strip())
        if first:
            candidates.append(first)
        if not first and following:
            candidates.append("\n".join(following).strip())

        if not candidates:
            return None, "missing action input"
        last_error = "invalid action input"
        for candidate in candidates:
            try:
                return json.loads(candidate), ""
            except json.JSONDecodeError as exc:
                last_error = str(exc)
        return None, last_error

    def _is_react_label(self, line: str) -> bool:
        lowered = line.lower()
        return lowered.startswith(
            (
                "thought:",
                "action:",
                "action input:",
                "observation:",
                "final answer:",
            )
        )

    def _parse_legacy_action_json(self, action_text: str, raw_output: str) -> AgentDecision:
        try:
            action = json.loads(action_text)
        except json.JSONDecodeError as exc:
            return AgentDecision(type="invalid", raw_output=raw_output, error=str(exc))
        if not isinstance(action, dict):
            return AgentDecision(
                type="invalid",
                raw_output=raw_output,
                error="action JSON must be an object",
            )
        if "name" in action:
            return AgentDecision(
                type="tool_call",
                raw_output=raw_output,
                tool_call={"name": action["name"], "arguments": action.get("arguments", {})},
            )
        return AgentDecision(type="invalid", raw_output=raw_output, error="action JSON missing name")

    def _parse_json(self, text: str) -> AgentDecision | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):

            return AgentDecision(
                type="invalid",
                raw_output=text,
                error="decision JSON must be an object",
            )

        decision_type = payload.get("type")
        if decision_type == "final":
            answer = payload.get("answer") or payload.get("content")
            if answer is None:
                return AgentDecision(
                    type="invalid",
                    raw_output=text,
                    error="final decision missing answer",
                )
            return AgentDecision(type="final", raw_output=text, answer=str(answer), metadata=payload)

        if decision_type == "tool_call":
            tool_call = payload.get("tool_call") or {}
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return AgentDecision(
                    type="invalid",
                    raw_output=text,
                    error="tool_call requires name and object arguments",
                )
            return AgentDecision(
                type="tool_call",
                raw_output=text,
                tool_call={"name": name, "arguments": arguments},
                metadata=payload,
            )

        return AgentDecision(type="invalid", raw_output=text, error=f"unknown type: {decision_type}")
