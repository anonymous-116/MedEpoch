from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class PromptSection:
    title: str
    content: str

class MemoryContextRenderer(Protocol):
    def render(self, memories: list[dict[str, object]]) -> list[PromptSection]: ...

class DefaultMemoryContextRenderer:
    def render(self, memories: list[dict[str, object]]) -> list[PromptSection]:
        if not memories:
            return []
        return [
            PromptSection(
                title="Retrieved Memories",
                content=(
                    "Retrieved Memories are optional reusable information. "
                    "Use them only when their conditions match the current case.\n\n"
                    + json.dumps(memories, ensure_ascii=False, indent=2)
                ),
            )
        ]

class EpochMemoryContextRenderer:
    def render(self, memories: list[dict[str, object]]) -> list[PromptSection]:
        blocks: list[str] = []
        for index, memory in enumerate(memories, 1):
            metadata = memory.get("metadata")
            status = (
                str(metadata.get("lifecycle_status") or "provisional")
                if isinstance(metadata, dict)
                else "provisional"
            )
            lines = [
                f"Information {index}",
                f"Content: {str(memory.get('content') or '').strip()}",
            ]
            condition = str(memory.get("condition") or "").strip()
            if condition:
                lines.append(f"Use when: {condition}")
            if status == "provisional":
                lines.append(
                    "Note: this information has not yet been independently confirmed."
                )
            elif status == "disputed":
                lines.append(
                    "Warning: this information has unresolved conflicting evidence."
                )
            blocks.append("\n".join(lines))
        if not blocks:
            return []
        return [
            PromptSection(
                title="Relevant Information",
                content=(
                    "The following information may or may not be relevant to the "
                    "current case. Assess each item before using it, and use only "
                    "the information that applies.\n\n"
                    + "\n\n".join(blocks)
                ),
            )
        ]

class MemoryContextRendererRegistry:
    def __init__(self, default: MemoryContextRenderer | None = None) -> None:
        self.default = default or DefaultMemoryContextRenderer()
        self._renderers: dict[str, MemoryContextRenderer] = {}

    def register(self, mode: str, renderer: MemoryContextRenderer) -> None:
        self._renderers[mode] = renderer

    def render(
        self,
        mode: str,
        memories: list[dict[str, object]],
    ) -> list[PromptSection]:
        return self._renderers.get(mode, self.default).render(memories)

def default_memory_context_renderers() -> MemoryContextRendererRegistry:
    registry = MemoryContextRendererRegistry()
    registry.register("epoch", EpochMemoryContextRenderer())
    return registry
