from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from amf.tools.base import BaseTool, ToolResult

DEFAULT_SEARCH_URL = "http://127.0.0.1:9001/search"

SEARCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer"},
    },
    "required": ["query"],
}

SEARCH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "score": {"type": "number"},
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            },
        },
    },
    "required": ["query", "top_k", "results"],
}

class HTTPSearchTool(BaseTool):
    """POST a JSON search request to a retrieval service."""

    name = "http_search"
    description = "Search a remote retrieval service."
    input_schema = SEARCH_INPUT_SCHEMA
    output_schema = SEARCH_OUTPUT_SCHEMA

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_SEARCH_URL,
        timeout: int = 30,
        default_top_k: int = 2,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.default_top_k = default_top_k

    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        self.validate_input(arguments)
        query = str(arguments["query"])

        top_k = self.default_top_k
        return await asyncio.to_thread(self._call_sync, query, top_k)

    def _call_sync(self, query: str, top_k: int) -> ToolResult:
        body = {"query": query, "top_k": top_k}
        request = urllib.request.Request(
            url=self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ToolResult(ok=False, tool_name=self.name, error=f"HTTP {exc.code}: {detail}")
        except Exception as exc:
            return ToolResult(ok=False, tool_name=self.name, error=str(exc))

        return ToolResult(
            ok=True,
            tool_name=self.name,
            output=normalize_search_payload(payload, fallback_query=query, fallback_top_k=top_k),
            metadata={"endpoint": self.endpoint},
        )

class MedicalTextSearchTool(HTTPSearchTool):
    name = "medical_text_search"
    description = (
        "Search medical literature or clinical text snippets. Use this for "
        "medical evidence, treatment, diagnosis, medication, guideline, or "
        "patient-safety questions."
    )

class GeneralSearchTool(HTTPSearchTool):
    name = "general_search"
    description = (
        "Search general background information. Use this when the question needs "
        "broad non-specialized retrieval or context beyond medical text."
    )

class MedicalTermSearchTool(HTTPSearchTool):
    name = "medical_term_search"
    description = (
        "Search explanations of medical terms, abbreviations, diseases, drugs, "
        "tests, symptoms, or clinical concepts."
    )

def normalize_search_payload(
    payload: dict[str, Any],
    *,
    fallback_query: str,
    fallback_top_k: int,
) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        results = []

    normalized_results: list[dict[str, Any]] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        normalized_results.append(
            {
                "rank": int(item.get("rank") or index),
                "score": item.get("score"),
                "text": str(item.get("text") or ""),
                "metadata": dict(item.get("metadata") or {}),
            }
        )

    top_k = payload.get("top_k", fallback_top_k)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = fallback_top_k

    return {
        "query": str(payload.get("query") or fallback_query),
        "top_k": top_k,
        "results": normalized_results,
    }
