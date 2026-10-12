from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from amf.events import utc_now_iso

@dataclass(slots=True)
class KnowledgeItem:
    """One reusable memory record.

    Epoch Memory uses one sparse schema so optional fields stay independent of
    the dataset.
    """

    id: str
    method: str
    name: str
    key: str
    content: str
    condition: str | None = None
    category: str | None = None
    tools: list[str] = field(default_factory=list)
    abandoned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        key: str,
        content: str,
        method: str | None = None,
        name: str | None = None,
        condition: str | None = None,
        category: str | None = None,
        tools: list[str] | None = None,
        abandoned: bool = False,
        metadata: dict[str, Any] | None = None,
        **legacy: Any,
    ) -> "KnowledgeItem":

        legacy_kind = legacy.get("kind")
        legacy_type = legacy.get("type")
        resolved_method = method or legacy.get("method") or legacy_type or legacy_kind or "memory"
        resolved_name = name or key
        resolved_metadata = dict(metadata or {})
        resolved_metadata.setdefault("created_at", utc_now_iso())
        resolved_metadata.setdefault("updated_at", None)
        resolved_metadata.setdefault("updated_from", None)
        resolved_metadata.setdefault("updated_to", None)
        if legacy_kind is not None:
            resolved_metadata.setdefault("legacy_kind", legacy_kind)
        if legacy_type is not None:
            resolved_metadata.setdefault("legacy_type", legacy_type)
        return cls(
            id=str(uuid4()),
            method=str(resolved_method),
            name=str(resolved_name),
            key=key,
            content=content,
            condition=condition,
            category=category,
            tools=list(tools or []),
            abandoned=abandoned,
            metadata=resolved_metadata,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeItem":
        metadata = dict(payload.get("metadata") or {})
        if "created_at" in payload:
            metadata.setdefault("created_at", payload.get("created_at"))
        if "updated_at" in payload:
            metadata.setdefault("updated_at", payload.get("updated_at"))
        metadata.setdefault("created_at", utc_now_iso())
        metadata.setdefault("updated_at", None)
        metadata.setdefault("updated_from", None)
        metadata.setdefault("updated_to", None)

        legacy_kind = payload.get("kind")
        legacy_type = payload.get("type")
        if legacy_kind is not None:
            metadata.setdefault("legacy_kind", legacy_kind)
        if legacy_type is not None:
            metadata.setdefault("legacy_type", legacy_type)

        method = payload.get("method") or metadata.get("method") or legacy_type or legacy_kind or "memory"
        return cls(
            id=str(payload["id"]),
            method=str(method),
            name=str(payload.get("name") or payload.get("key") or method),
            key=str(payload["key"]),
            content=str(payload["content"]),
            condition=(
                str(payload["condition"])
                if payload.get("condition") is not None
                else None
            ),
            category=(
                str(payload["category"])
                if payload.get("category") is not None
                else None
            ),
            tools=[str(tool) for tool in payload.get("tools") or []],
            abandoned=bool(payload.get("abandoned", False)),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_score(self, score: float) -> "KnowledgeItem":
        payload = self.to_dict()
        metadata = dict(payload.get("metadata") or {})
        metadata["retrieval_score"] = score
        payload["metadata"] = metadata
        return KnowledgeItem.from_dict(payload)
