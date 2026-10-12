from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass(slots=True)
class Event:
    event_id: str
    episode_id: str
    turn_id: int
    type: str
    payload: dict[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)

    @classmethod
    def create(
        cls,
        episode_id: str,
        turn_id: int,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> "Event":
        return cls(
            event_id=str(uuid4()),
            episode_id=episode_id,
            turn_id=turn_id,
            type=event_type,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
