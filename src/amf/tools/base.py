from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass(slots=True)
class ToolSpec:
    """Simplified MCP-style public description of a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class ToolResult:
    """Structured result returned by every tool implementation."""

    ok: bool
    tool_name: str
    output: dict[str, Any] | list[Any] | str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class BaseTool(ABC):
    """Base class for local tools and future protocol adapters.

    A real MCP tool can be wrapped behind this interface by mapping MCP's
    discovered schema to ToolSpec and MCP's call result to ToolResult.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

    def validate_input(self, arguments: dict[str, Any]) -> None:

        required = self.input_schema.get("required", [])
        for key in required:
            if key not in arguments:
                raise ValueError(f"Missing required tool argument: {key}")

        properties = self.input_schema.get("properties", {})
        for key, schema in properties.items():
            if key not in arguments:
                continue
            expected_type = schema.get("type")
            if expected_type == "string" and not isinstance(arguments[key], str):
                raise ValueError(f"Argument {key} must be a string.")
            if expected_type == "number" and not isinstance(arguments[key], int | float):
                raise ValueError(f"Argument {key} must be a number.")
            if expected_type == "integer" and not isinstance(arguments[key], int):
                raise ValueError(f"Argument {key} must be an integer.")
            if expected_type == "object" and not isinstance(arguments[key], dict):
                raise ValueError(f"Argument {key} must be an object.")
            if expected_type == "array" and not isinstance(arguments[key], list):
                raise ValueError(f"Argument {key} must be an array.")

    @abstractmethod
    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        """Call the tool with structured JSON arguments."""
