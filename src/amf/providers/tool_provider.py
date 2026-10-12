from __future__ import annotations

from amf.providers.base import BaseProvider, ProviderRequest, ProviderResponse
from amf.tools.base import BaseTool

class ToolProvider(BaseProvider):
    name = "tool"

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:

        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_specs(self) -> list[dict]:
        return [tool.spec().to_dict() for tool in self._tools.values()]

    async def call(self, request: ProviderRequest) -> ProviderResponse:
        if request.operation == "list":
            return ProviderResponse(ok=True, data=self.list_specs())

        if request.operation != "call":
            return ProviderResponse(ok=False, error=f"Unsupported tool operation: {request.operation}")

        tool_call = request.payload.get("tool_call") or request.payload
        name = tool_call.get("name")
        arguments = tool_call.get("arguments") or {}

        if name not in self._tools:
            return ProviderResponse(
                ok=False,
                error=f"Tool not found: {name}",
                metadata={"available_tools": sorted(self._tools)},
            )
        tool = self._tools[name]
        try:
            tool.validate_input(arguments)
            result = await tool.call(arguments)
        except Exception as exc:
            return ProviderResponse(ok=False, error=str(exc), metadata={"tool_name": name})
        return ProviderResponse(
            ok=result.ok,
            data=result.to_dict(),
            error=result.error,
            metadata={"tool_name": name},
        )
