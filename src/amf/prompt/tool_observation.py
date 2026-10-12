from __future__ import annotations

import json
from typing import Any, Callable

ObservationRenderer = Callable[[Any], str]

class ToolObservationRendererRegistry:
    """Turn runtime tool envelopes into concise model-facing observations.

    The complete structured response remains in ``EpisodeState`` and trajectory
    artifacts for reproducibility.  Only this rendered view is sent back to the
    solving model.
    """

    def __init__(self) -> None:
        self._renderers: dict[str, ObservationRenderer] = {}

    def register(self, tool_name: str, renderer: ObservationRenderer) -> None:
        self._renderers[tool_name] = renderer

    def render(self, observation: Any) -> str:
        if not isinstance(observation, dict):
            return str(observation)
        if observation.get("error"):
            return f"Tool error: {observation['error']}"

        data = observation.get("data")
        if not isinstance(data, dict):
            return _compact_value(observation)
        if data.get("error"):
            return f"Tool error: {data['error']}"

        tool_name = str(
            data.get("tool_name")
            or dict(observation.get("metadata") or {}).get("tool_name")
            or ""
        )
        output = data.get("output")
        renderer = self._renderers.get(tool_name)
        if renderer is not None:
            return renderer(output)

        return _compact_value(output if output is not None else data)

def _render_http_search(output: Any) -> str:
    if not isinstance(output, dict):
        return _compact_value(output)
    texts = [
        str(item.get("text") or "").strip()
        for item in output.get("results") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]

    blocks = [
        f"检索结果 {index}：\n{text}"
        for index, text in enumerate(texts, start=1)
    ]
    return "\n\n".join(blocks) if blocks else "No results returned."

def _render_general_search(output: Any) -> str:
    """Show only Tavily's title, URL, and concise content to the model."""

    if not isinstance(output, dict):
        return _compact_value(output)
    blocks: list[str] = []
    for item in output.get("results") or []:
        if not isinstance(item, dict):
            continue
        metadata = dict(item.get("metadata") or {})
        title = str(metadata.get("title") or "").strip()
        url = str(metadata.get("url") or "").strip()
        content = str(item.get("text") or "").strip()
        if not any((title, url, content)):
            continue
        lines = [f"Result {int(item.get('rank') or len(blocks) + 1)}"]
        if title:
            lines.append(f"Title: {title}")
        if url:
            lines.append(f"URL: {url}")
        if content:
            lines.append(f"Content: {content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No results returned."

def _render_medical_term_search(output: Any) -> str:
    """Expose the MedlinePlus topic summary without XML/storage fields."""

    if not isinstance(output, dict):
        return _compact_value(output)
    blocks: list[str] = []
    for item in output.get("results") or []:
        if not isinstance(item, dict):
            continue
        metadata = dict(item.get("metadata") or {})
        title = str(metadata.get("title") or "").strip()
        url = str(metadata.get("url") or "").strip()
        summary = str(item.get("text") or "").strip()
        if not any((title, url, summary)):
            continue
        lines: list[str] = []
        if title:
            lines.append(f"Medical term: {title}")
        aliases = [
            str(value).strip()
            for value in metadata.get("also_called") or []
            if str(value).strip()
        ]
        if aliases:
            lines.append(f"Also called: {', '.join(aliases)}")
        warning = str(metadata.get("match_warning") or "").strip()
        if warning:
            lines.append(f"Warning: {warning}")
        if summary:
            lines.append(f"Summary: {summary}")
        if url:
            lines.append(f"Source: {url}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No matching MedlinePlus topic found."

def _render_calculator_search(output: Any) -> str:
    if not isinstance(output, dict):
        return _compact_value(output)
    blocks: list[str] = []
    for match in output.get("matches") or []:
        if not isinstance(match, dict):
            continue
        calculator_id = str(match.get("calculator_id") or "").strip()
        description = str(match.get("description") or "").strip()
        schema = dict(match.get("required_schema") or {})
        required = set(schema.get("required") or [])
        parameters: list[str] = []
        for name, raw_spec in dict(schema.get("properties") or {}).items():
            spec = dict(raw_spec or {})
            value_type = str(spec.get("type") or "any")
            requirement = "required" if name in required else "optional"
            detail = str(spec.get("description") or "").strip()
            rendered = f"- {name} ({value_type}, {requirement})"
            if detail:
                rendered += f": {detail}"
            parameters.append(rendered)
        lines = [f"Calculator: {calculator_id}"]
        if description:
            lines.append(f"Description: {description}")
        if parameters:
            lines.extend(["Parameters:", *parameters])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No matching calculator found."

def _render_calculator_result(output: Any) -> str:
    if not isinstance(output, dict):
        return _compact_value(output)
    lines = [f"Calculated value: {_compact_value(output.get('value'))}"]
    interpretation = str(output.get("interpretation") or "").strip()
    if interpretation:
        lines.append(f"Interpretation: {interpretation}")

    return "\n".join(lines)

def _render_drug_lookup(output: Any) -> str:
    if not isinstance(output, dict):
        return _compact_value(output)
    blocks: list[str] = []
    for result in output.get("results") or []:
        if not isinstance(result, dict):
            continue
        query = str(result.get("query") or "").strip()
        matches = [item for item in result.get("matches") or [] if isinstance(item, dict)]
        if not matches:
            blocks.append(f"Drug query: {query}\nNo matching drug found.")
            continue
        for match in matches:
            name = str(match.get("name") or "Unknown drug")
            drugbank_id = str(match.get("drugbank_id") or "").strip()
            heading = f"Drug: {name}"
            if drugbank_id:
                heading += f" ({drugbank_id})"
            lines = [f"Query: {query}", heading]
            matched_name = str(match.get("matched_name") or "").strip()
            match_type = str(match.get("match_type") or "").strip()
            if matched_name:
                lines.append(f"Matched as: {matched_name} ({match_type or 'name'})")
            groups = [str(value) for value in match.get("groups") or [] if value]
            if groups:
                lines.append(f"Groups: {', '.join(groups)}")
            for label, key in (
                ("Description", "description"),
                ("Indication", "indication"),
            ):
                limits = {
                    "description": 500,
                    "indication": 800,
                }
                value = _clean_text(match.get(key), limit=limits[key])
                if value:
                    lines.append(f"{label}: {value}")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No drug results returned."

def _render_drug_interaction(output: Any) -> str:
    if not isinstance(output, dict):
        return _compact_value(output)

    def resolved_text(key: str) -> str:
        values = []
        for item in output.get(key) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "Unknown")
            drugbank_id = str(item.get("drugbank_id") or "").strip()
            values.append(f"{name} ({drugbank_id})" if drugbank_id else name)
        return ", ".join(values) if values else "unresolved"

    lines = [
        f"Resolved drug A: {resolved_text('resolved_drug_a')}",
        f"Resolved drug B: {resolved_text('resolved_drug_b')}",
    ]

    interactions: list[str] = []
    seen_interactions: set[str] = set()
    for item in output.get("interactions") or []:
        if not isinstance(item, dict):
            continue
        description = _clean_text(item.get("description"))
        if description and description not in seen_interactions:
            seen_interactions.add(description)
            interactions.append(description)
    if interactions:
        lines.append("Recorded interactions:")
        lines.extend(f"- {description}" for description in interactions)
    else:
        lines.append("Recorded interactions: none returned.")
    warning = str(output.get("warning") or "").strip()
    if warning:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)

def _compact_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return "No result returned."
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

def _clean_text(value: Any, *, limit: int | None = None) -> str:
    """Collapse database formatting and optionally cap model-visible text."""

    text = " ".join(str(value or "").split())
    if limit is not None and len(text) > limit:
        return text[:limit].rstrip() + "…"
    return text

def default_tool_observation_renderers() -> ToolObservationRendererRegistry:
    registry = ToolObservationRendererRegistry()
    for tool_name in (
        "http_search",
        "medical_text_search",
    ):
        registry.register(tool_name, _render_http_search)
    registry.register("general_search", _render_general_search)
    registry.register("medical_term_search", _render_medical_term_search)
    registry.register("search_clinical_calculators", _render_calculator_search)
    registry.register("execute_clinical_calculator", _render_calculator_result)
    registry.register("lookup_drugs", _render_drug_lookup)
    registry.register("check_drug_interaction", _render_drug_interaction)
    return registry
