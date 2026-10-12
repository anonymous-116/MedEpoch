from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from amf.tools.base import BaseTool, ToolResult

class OpenMedicineRuntime:
    """Load and execute an external clinical calculator registry."""

    def __init__(self, source_root: str | Path, max_search_results: int = 3) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.max_search_results = max(1, max_search_results)
        self._registry: dict[str, Any] | None = None
        self._search = None

    def _load(self) -> tuple[dict[str, Any], Any]:
        if self._registry is not None and self._search is not None:
            return self._registry, self._search
        if not self.source_root.is_dir():
            raise RuntimeError(f"Open Medicine source directory not found: {self.source_root}")

        source = str(self.source_root)
        if source not in sys.path:
            sys.path.insert(0, source)
        from open_medicine.mcp.registry import CALCULATOR_REGISTRY
        from open_medicine.mcp.search_utils import tokenized_search

        self._registry = CALCULATOR_REGISTRY
        self._search = tokenized_search
        return self._registry, self._search

    def search(self, query: str) -> list[dict[str, Any]]:
        registry, tokenized_search = self._load()
        items = [
            {
                "calculator_id": calculator_id,
                "description": definition.description,
                "required_schema": definition.schema,
                "searchable_text": f"{calculator_id} {definition.description}",
            }
            for calculator_id, definition in registry.items()
        ]

        matches = tokenized_search(query, items, max_results=self.max_search_results)

        for match in matches:
            match.pop("_score", None)
        return matches

    def execute(self, calculator_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        registry, _ = self._load()
        if calculator_id not in registry:
            raise ValueError(
                f"Unknown calculator_id '{calculator_id}'. "
                "Call search_clinical_calculators first."
            )
        definition = registry[calculator_id]
        validated = definition.pydantic_model(**parameters)
        result = definition.execute_function(validated)
        return result.model_dump(mode="json")

class SearchClinicalCalculatorsTool(BaseTool):
    name = "search_clinical_calculators"
    description = (
        "Search the local Open Medicine registry for a clinical calculator. "
        "Always call this before execution and use the returned calculator_id "
        "and required_schema instead of guessing parameter names. Use English keywords."
    )
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    output_schema = {
        "type": "object",
        "properties": {"matches": {"type": "array"}},
        "required": ["matches"],
    }

    def __init__(self, runtime: OpenMedicineRuntime) -> None:
        self.runtime = runtime

    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        self.validate_input(arguments)
        query = str(arguments["query"]).strip()
        if not query:
            return ToolResult(ok=False, tool_name=self.name, error="query must not be empty")
        try:
            matches = self.runtime.search(query)
        except Exception as exc:
            return ToolResult(ok=False, tool_name=self.name, error=str(exc))
        return ToolResult(ok=True, tool_name=self.name, output={"matches": matches})

class ExecuteClinicalCalculatorTool(BaseTool):
    name = "execute_clinical_calculator"
    description = (
        "Execute a deterministic Open Medicine clinical calculator. Only use a "
        "calculator_id and parameters copied from search_clinical_calculators; "
        "do not infer missing patient facts."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "calculator_id": {"type": "string"},
            "parameters": {"type": "object"},
        },
        "required": ["calculator_id", "parameters"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "value": {},
            "interpretation": {"type": "string"},
            "evidence": {"type": "object"},
            "fhir_code": {"type": "string"},
            "fhir_system": {"type": "string"},
            "fhir_display": {"type": "string"},
        },
        "required": ["value", "interpretation", "evidence"],
    }

    def __init__(self, runtime: OpenMedicineRuntime) -> None:
        self.runtime = runtime

    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        self.validate_input(arguments)
        try:
            output = self.runtime.execute(
                str(arguments["calculator_id"]),
                dict(arguments["parameters"]),
            )
        except Exception as exc:
            return ToolResult(ok=False, tool_name=self.name, error=str(exc))
        return ToolResult(ok=True, tool_name=self.name, output=output)
