from __future__ import annotations

import json
from typing import Any

from amf.prompt.agent import (
    FINAL_OUTPUT_FORMAT,
    FORCED_FINAL_OUTPUT_FORMAT,
    FINAL_TURN_INSTRUCTION,
    MODE_INSTRUCTIONS,
    STRICT_OUTPUT_FORMAT_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    TOOL_OUTPUT_FORMAT,
)
from amf.prompt.memory_context import (
    MemoryContextRendererRegistry,
    default_memory_context_renderers,
)
from amf.prompt.tool_observation import (
    ToolObservationRendererRegistry,
    default_tool_observation_renderers,
)
from amf.state import EpisodeState

class ContextAssembler:
    """Build the only prompt string the Agent sees for a turn."""

    def __init__(
        self,
        max_context_items: int = 8,
        memory_renderers: MemoryContextRendererRegistry | None = None,
        observation_renderers: ToolObservationRendererRegistry | None = None,
    ) -> None:
        self.max_context_items = max_context_items

        self.memory_renderers = memory_renderers or default_memory_context_renderers()
        self.observation_renderers = (
            observation_renderers or default_tool_observation_renderers()
        )

    def build_prompt(
        self,
        state: EpisodeState,
        tool_specs: list[dict[str, Any]] | None = None,
        mode: str = "react",
        force_final: bool = False,
    ) -> str:

        active_tool_specs = [] if force_final else tool_specs
        if force_final:
            system_instruction = FINAL_TURN_INSTRUCTION
        else:
            system_instruction = SYSTEM_INSTRUCTION
        sections = [self._section("System Instruction", system_instruction)]
        sections.extend(
            self._runtime_context_sections(
                state,
                active_tool_specs,
                mode,
                force_final=force_final,
            )
        )
        sections.append(self._section("Task", self._task_with_scratchpad_text(state)))
        return "\n\n".join(sections)

    def build_messages(
        self,
        state: EpisodeState,
        tool_specs: list[dict[str, Any]] | None = None,
        mode: str = "react",
    ) -> list[dict[str, str]]:
        """Build a single-message prompt."""

        prompt = self.build_prompt(state, tool_specs, mode)
        return self.build_messages_from_prompt(prompt)

    def build_messages_from_prompt(self, prompt: str) -> list[dict[str, str]]:
        """Wrap an already assembled prompt without rebuilding it."""

        return [{"role": "system", "content": prompt}]

    def _contexts_by_source(self, state: EpisodeState, source: str) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        for context in state.retrieved_contexts:
            if context["source"] == source:
                contexts.extend(context.get("results") or [])
        return contexts[-self.max_context_items :]

    def _runtime_context_sections(
        self,
        state: EpisodeState,
        tool_specs: list[dict[str, Any]] | None,
        mode: str,
        *,
        force_final: bool = False,
    ) -> list[str]:
        sections: list[str] = []
        skills = self._contexts_by_source(state, "skill")
        memories = self._contexts_by_source(state, "memory")
        knowledge = self._contexts_by_source(state, "knowledge")
        global_skill = str(state.metadata.get("global_skill") or "").strip()
        if global_skill:
            skill_parts = [
                "Global Skill provides general guidance for solving related tasks.",
                global_skill,
            ]
            mode_guidance = MODE_INSTRUCTIONS.get(mode, "")
            if mode_guidance and mode != "epoch":
                skill_parts.append(mode_guidance)
            sections.append(self._section("Global Skill", "\n\n".join(skill_parts)))
        elif MODE_INSTRUCTIONS.get(mode) and mode != "epoch":
            sections.append(
                self._section(
                    "Global Skill",
                    "General method guidance:\n\n" + MODE_INSTRUCTIONS[mode],
                )
            )
        if skills:
            sections.append(self._section("Retrieved Skills", self._json(skills)))
        for memory_section in self.memory_renderers.render(mode, memories):
            sections.append(
                self._section(memory_section.title, memory_section.content)
            )
        if knowledge:
            sections.append(
                self._section(
                    "Retrieved External Evidence",
                    "The following evidence was retrieved once using the current question. "
                    "Use it as supporting information and do not invent details beyond it.\n\n"
                    + self._json(knowledge),
                )
            )
        if state.dialogue_history:
            sections.append(self._section("Runtime History", self._json(state.dialogue_history)))
        if tool_specs:
            sections.append(self._section("Tool Use Protocol", self._tool_use_protocol()))
            sections.append(self._section("Tools", self._format_tool_specs(tool_specs)))

        if force_final:
            output_format = FORCED_FINAL_OUTPUT_FORMAT
        else:
            output_format = FINAL_OUTPUT_FORMAT
            if tool_specs:
                output_format += "\n" + TOOL_OUTPUT_FORMAT
            output_format += "\n" + STRICT_OUTPUT_FORMAT_INSTRUCTION
        if output_format:
            sections.append(self._section("Output Format", output_format))
        return sections

    def _section(self, title: str, content: str) -> str:
        return f"## {title}\n{content}"

    def _json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _tool_use_protocol(self) -> str:
        return "\n".join(
            [
                "When calling a tool, provide one Action and a valid JSON Action Input ",
                "matching that tool's listed Input format. Do not provide a Final Answer ",
                "in the same turn. The runtime will execute the Action and provide the ",
                "Observation in the next turn; you do not need to generate it yourself.",
            ]
        )

    def _format_tool_specs(self, tool_specs: list[dict[str, Any]]) -> str:
        """Render tool specs as compact model-facing instructions."""

        blocks: list[str] = []
        for spec in tool_specs:
            name = str(spec.get("name") or "").strip()
            if not name:
                continue
            description = str(spec.get("description") or "").strip()
            input_format = self._format_tool_input(spec.get("input_schema") or {})
            output_schema = dict(spec.get("output_schema") or {})
            output_description = str(output_schema.get("description") or "").strip()
            blocks.append(
                "\n".join(
                    [
                        name,
                        f"Description: {description}" if description else "Description: No description provided.",
                        f"Input: {input_format}",
                        (
                            f"Returns: {output_description}."
                            if output_description
                            else "Returns: an observation from the runtime."
                        ),
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _format_tool_input(self, input_schema: dict[str, Any]) -> str:
        properties = dict(input_schema.get("properties") or {})
        required = set(input_schema.get("required") or [])
        visible_fields = [
            (name, schema)
            for name, schema in properties.items()
            if name != "top_k"
        ]
        if not visible_fields:
            return "{}"

        parts: list[str] = []
        for name, schema in visible_fields:
            value_type = str(dict(schema or {}).get("type") or "any")
            suffix = "" if name in required else " optional"
            parts.append(f'"{name}": {value_type}{suffix}')
        return "{" + ", ".join(parts) + "}"

    def _case_to_task_text(self, case: dict[str, Any]) -> str:
        """Render only model-visible task content, not experiment metadata."""

        messages = case.get("messages")
        if isinstance(messages, list) and messages:
            rendered: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = self._message_content_to_text(message.get("content"))
                if content:
                    role = str(message.get("role") or "user").strip().lower()
                    role_label = {
                        "user": "User",
                        "assistant": "Assistant",
                        "system": "System",
                        "tool": "Tool",
                    }.get(role, role.title() or "User")
                    rendered.append(f"{role_label}: {content}")
            if rendered:
                return "\n\n".join(rendered)

        for key in ("question", "input", "query", "task"):
            if case.get(key):
                return str(case[key])

        return self._json(case)

    def _task_with_scratchpad_text(self, state: EpisodeState) -> str:
        text = f"Question: {self._case_to_task_text(state.case)}"
        if state.trajectory:
            text += "\n\n" + self._trajectory_text(state)
        return text

    def _trajectory_text(self, state: EpisodeState) -> str:
        lines: list[str] = []
        for entry in state.trajectory[-self.max_context_items * 4 :]:
            kind = str(entry.get("kind") or "")
            content = entry.get("content")
            if kind == "thought":
                lines.append(f"Thought: {content}")
            elif kind == "action":
                lines.append(f"Action: {content}")
            elif kind == "action_input":
                lines.append(f"Action Input: {self._compact_json(content)}")
            elif kind == "observation":
                lines.append(f"Observation: {self._format_observation(content)}")
            elif kind == "final_answer":
                lines.append(f"Final Answer: {content}")
        return "\n\n".join(lines)

    def _compact_json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _format_observation(self, observation: Any) -> str:
        return self.observation_renderers.render(observation)

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part.strip() for part in parts if part.strip())
        if content is None:
            return ""
        return str(content).strip()
