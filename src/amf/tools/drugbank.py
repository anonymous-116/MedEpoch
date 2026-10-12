from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from amf.tools.base import BaseTool, ToolResult

class DrugBankRuntime:
    """Lazy adapter for the vendored, read-only DrugBank SQLite runtime."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        database_path: str | Path,
        max_matches_per_name: int = 2,
        max_names_per_call: int = 10,
        max_text_chars: int = 4000,
        max_interaction_results: int = 10,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.database_path = Path(database_path).expanduser().resolve()
        self.settings = {
            "max_matches_per_name": max_matches_per_name,
            "max_names_per_call": max_names_per_call,
            "max_text_chars": max_text_chars,
            "max_interaction_results": max_interaction_results,
        }
        self._database = None

    def database(self):
        if self._database is not None:
            return self._database
        if not self.source_root.is_dir():
            raise RuntimeError(f"DrugBank source directory not found: {self.source_root}")
        source = str(self.source_root)
        if source not in sys.path:
            sys.path.insert(0, source)
        from drugbank_mcp.database import DrugBankDatabase

        self._database = DrugBankDatabase(self.database_path, **self.settings)

        self._database.metadata()
        return self._database

class LookupDrugsTool(BaseTool):
    name = "lookup_drugs"
    description = (
        "Look up one or more drugs in the local DrugBank database by exact standard "
        "name, synonym, product name, or DrugBank ID. Use English names. This dataset "
        "is historical; the result is drug information, not a prescribing recommendation."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["names"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "drugbank_version": {"type": "string"},
            "drugbank_exported_on": {"type": "string"},
            "results": {"type": "array"},
        },
        "required": ["drugbank_version", "drugbank_exported_on", "results"],
    }

    def __init__(self, runtime: DrugBankRuntime) -> None:
        self.runtime = runtime

    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        self.validate_input(arguments)
        try:
            output = self.runtime.database().lookup_drugs(arguments["names"])
        except Exception as exc:
            return ToolResult(ok=False, tool_name=self.name, error=str(exc))
        return ToolResult(ok=True, tool_name=self.name, output=output)

class CheckDrugInteractionTool(BaseTool):
    name = "check_drug_interaction"
    description = (
        "Check the local DrugBank dataset for recorded interactions between two "
        "drugs, resolving exact standard names, synonyms, product names, or DrugBank "
        "IDs. No record does not prove that a combination is safe."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "drug_a": {"type": "string"},
            "drug_b": {"type": "string"},
        },
        "required": ["drug_a", "drug_b"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "drugbank_version": {"type": "string"},
            "drugbank_exported_on": {"type": "string"},
            "query": {"type": "object"},
            "resolved_drug_a": {"type": "array"},
            "resolved_drug_b": {"type": "array"},
            "interactions": {"type": "array"},
            "warning": {},
        },
        "required": [
            "drugbank_version",
            "drugbank_exported_on",
            "query",
            "resolved_drug_a",
            "resolved_drug_b",
            "interactions",
            "warning",
        ],
    }

    def __init__(self, runtime: DrugBankRuntime) -> None:
        self.runtime = runtime

    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        self.validate_input(arguments)
        try:
            output = self.runtime.database().check_drug_interaction(
                arguments["drug_a"], arguments["drug_b"]
            )
        except Exception as exc:
            return ToolResult(ok=False, tool_name=self.name, error=str(exc))
        return ToolResult(ok=True, tool_name=self.name, output=output)
