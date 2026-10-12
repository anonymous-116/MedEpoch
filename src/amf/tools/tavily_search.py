from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any

from amf.tools.base import BaseTool, ToolResult
from amf.tools.http_search import SEARCH_INPUT_SCHEMA, SEARCH_OUTPUT_SCHEMA

TavilyClientFactory = Callable[[str], Any]

class TavilyAPIKeyPool:
    """Shuffle once at startup, then hand out API keys round-robin.

    ``acquire`` is protected by a lock because Episode batches may call tools
    from several worker threads at the same time.  Keys are never included in
    ToolResult metadata or error messages.
    """

    def __init__(
        self,
        api_keys: Sequence[str],
        *,
        shuffle: bool = True,
        randomizer: random.Random | random.SystemRandom | None = None,
    ) -> None:

        keys = list(
            dict.fromkeys(
                str(value).strip()
                for value in api_keys
                if str(value).strip()
            )
        )
        if not keys:
            raise ValueError("Tavily API key pool cannot be empty.")
        if shuffle:
            (randomizer or random.SystemRandom()).shuffle(keys)
        self._keys = tuple(keys)
        self._next_index = 0
        self._lock = Lock()

    @property
    def size(self) -> int:
        return len(self._keys)

    def acquire(self) -> tuple[str, int]:
        """Return ``(key, shuffled_slot)`` and advance the shared cursor."""

        with self._lock:
            slot = self._next_index
            key = self._keys[slot]
            self._next_index = (slot + 1) % len(self._keys)
        return key, slot

    def redact(self, text: str) -> str:
        redacted = str(text)
        for key in self._keys:
            redacted = redacted.replace(key, "[redacted]")
        return redacted

class GeneralSearchTool(BaseTool):
    """General web search backed by a shuffled Tavily account pool."""

    name = "general_search"
    description = (
        "Search the public web for current or general background information. "
        "Use this when up-to-date or broad information beyond the local medical "
        "retrieval corpus is needed."
    )
    input_schema = SEARCH_INPUT_SCHEMA
    output_schema = SEARCH_OUTPUT_SCHEMA

    def __init__(
        self,
        *,
        api_keys: Sequence[str] | None = None,
        search_depth: str = "basic",
        max_results: int = 3,
        topic: str = "general",
        include_raw_content: bool = False,
        shuffle_keys: bool = True,
        client_factory: TavilyClientFactory | None = None,
    ) -> None:
        if search_depth != "basic":
            raise ValueError("GeneralSearchTool currently requires search_depth='basic'.")
        if max_results != 3:
            raise ValueError("GeneralSearchTool currently requires max_results=3.")
        if include_raw_content:
            raise ValueError(
                "GeneralSearchTool currently requires include_raw_content=False."
            )
        configured_keys = list(api_keys or [])
        if not configured_keys:
            environment_key = os.getenv("TAVILY_API_KEY", "").strip()
            if environment_key:
                configured_keys = [environment_key]
        self.key_pool = (
            TavilyAPIKeyPool(configured_keys, shuffle=shuffle_keys)
            if configured_keys
            else None
        )
        self.search_depth = search_depth
        self.max_results = max_results
        self.topic = topic
        self.include_raw_content = include_raw_content
        self.client_factory = client_factory or _default_tavily_client

    async def call(self, arguments: dict[str, Any]) -> ToolResult:
        self.validate_input(arguments)
        query = str(arguments["query"]).strip()
        if not query:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error="Tavily search query cannot be empty.",
            )
        if self.key_pool is None:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=(
                    "No Tavily API keys are configured. Set tools.general_search."
                    "api_keys_file or TAVILY_API_KEY."
                ),
            )
        return await asyncio.to_thread(self._call_sync, query)

    def _call_sync(self, query: str) -> ToolResult:
        assert self.key_pool is not None
        api_key, account_slot = self.key_pool.acquire()
        try:
            client = self.client_factory(api_key)
            payload = client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=self.max_results,
                topic=self.topic,
                include_raw_content=self.include_raw_content,
            )
            if not isinstance(payload, dict):
                raise TypeError("Tavily SDK returned a non-object response.")
            output = normalize_tavily_payload(
                payload,
                query=query,
                max_results=self.max_results,
            )
            return ToolResult(
                ok=True,
                tool_name=self.name,
                output=output,
                metadata={
                    "provider": "tavily",
                    "account_slot": account_slot,
                    "pool_size": self.key_pool.size,
                    "search_depth": self.search_depth,
                    "max_results": self.max_results,
                    "include_raw_content": self.include_raw_content,
                },
            )
        except Exception as exc:

            message = self.key_pool.redact(str(exc))
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"Tavily search failed ({type(exc).__name__}): {message}",
                metadata={
                    "provider": "tavily",
                    "account_slot": account_slot,
                    "pool_size": self.key_pool.size,
                },
            )

def normalize_tavily_payload(
    payload: dict[str, Any],
    *,
    query: str,
    max_results: int,
) -> dict[str, Any]:
    """Map Tavily fields onto the framework's compact search contract."""

    normalized_results: list[dict[str, Any]] = []
    for index, raw_result in enumerate(payload.get("results") or [], start=1):
        if not isinstance(raw_result, dict):
            continue
        content = " ".join(str(raw_result.get("content") or "").split())
        normalized_results.append(
            {
                "rank": index,
                "score": raw_result.get("score"),
                "text": content,
                "metadata": {
                    "title": str(raw_result.get("title") or "").strip(),
                    "url": str(raw_result.get("url") or "").strip(),
                },
            }
        )
        if len(normalized_results) >= max_results:
            break
    return {
        "query": str(payload.get("query") or query),
        "top_k": max_results,
        "results": normalized_results,
    }

def _default_tavily_client(api_key: str) -> Any:
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError(
            "tavily-python is not installed; run `pip install tavily-python`."
        ) from exc
    return TavilyClient(api_key=api_key)
