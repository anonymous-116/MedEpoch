from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextvars import ContextVar
from threading import Lock
from typing import Any

def _as_dict(value: Any) -> dict[str, Any]:
    """Convert SDK response models and ordinary mappings to a plain dict."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump() or {})
    if hasattr(value, "dict"):
        return dict(value.dict() or {})
    return {
        name: getattr(value, name)
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name, None))
    }

def _integer(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float):
            return int(value)
    return None

def normalize_usage(usage: Any) -> dict[str, Any]:
    """Normalize provider-specific usage fields."""

    raw = _as_dict(usage)
    prompt_details = _as_dict(
        raw.get("prompt_tokens_details") or raw.get("input_tokens_details")
    )
    completion_details = _as_dict(
        raw.get("completion_tokens_details") or raw.get("output_tokens_details")
    )
    cached_tokens = _integer(
        raw,
        "prompt_cache_hit_tokens",
        "cached_prompt_tokens",
        "cache_read_input_tokens",
    )
    if cached_tokens is None:
        cached_tokens = _integer(prompt_details, "cached_tokens", "cache_read_tokens")

    prompt_tokens = _integer(raw, "prompt_tokens", "input_tokens")
    completion_tokens = _integer(raw, "completion_tokens", "output_tokens")
    total_tokens = _integer(raw, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "usage_available": bool(raw),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_tokens,
        "reasoning_tokens": _integer(completion_details, "reasoning_tokens"),
        "raw_usage": raw,
    }

@dataclass
class LLMUsageTracker:
    """Collect all model-call usage records belonging to one experiment run."""

    records: list[dict[str, Any]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _context: ContextVar[dict[str, Any]] = field(
        default_factory=lambda: ContextVar("llm_usage_context", default={}),
        init=False,
        repr=False,
    )

    def set_context(self, **context: Any) -> None:
        """Attach run/row/epoch provenance to subsequent model calls."""

        self._context.set(
            {key: value for key, value in context.items() if value is not None}
        )

    def record(
        self,
        *,
        role: str,
        provider: str,
        model: str,
        usage: Any,
        response_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_usage(usage)
        with self._lock:
            record = {
                "call_id": len(self.records) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": role,
                "provider": provider,
                "model": model,
                "response_id": response_id,
                "context": dict(self._context.get()),
                **normalized,
                **dict(metadata or {}),
            }
            self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        numeric_fields = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_prompt_tokens",
            "reasoning_tokens",
        )
        totals = {
            field: sum(
                int(record[field])
                for record in self.records
                if isinstance(record.get(field), int | float)
            )
            for field in numeric_fields
        }
        cached = totals["cached_prompt_tokens"]
        prompt = totals["prompt_tokens"]
        roles = sorted({str(record.get("role")) for record in self.records})
        cache_stats_available = any(
            isinstance(record.get("cached_prompt_tokens"), int | float)
            for record in self.records
        )
        return {
            "num_calls": len(self.records),
            "num_calls_with_usage": sum(
                1 for record in self.records if record.get("usage_available")
            ),
            "num_fallbacks": sum(
                1 for record in self.records if record.get("fallback") is True
            ),
            **totals,
            "cache_stats_available": cache_stats_available,
            "prompt_cache_hit_rate": (
                cached / prompt if cache_stats_available and prompt else None
            ),
            "calls_by_role": {
                role: sum(1 for record in self.records if record.get("role") == role)
                for role in roles
            },
            "usage_by_role": {
                role: {
                    "num_calls": sum(
                        1 for record in self.records if record.get("role") == role
                    ),
                    **{
                        field: sum(
                            int(record[field])
                            for record in self.records
                            if record.get("role") == role
                            and isinstance(record.get(field), int | float)
                        )
                        for field in numeric_fields
                    },
                }
                for role in roles
            },
        }
