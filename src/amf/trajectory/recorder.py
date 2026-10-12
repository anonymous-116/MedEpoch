from __future__ import annotations

from pathlib import Path
from typing import Any

from amf.events import Event
from amf.state import EpisodeState
from amf.trajectory.serializer import TrajectorySerializer

class TrajectoryRecorder:
    def __init__(self, serializer: TrajectorySerializer) -> None:
        self.serializer = serializer
        self.events: list[Event] = []

    def record(
        self,
        episode_id: str,
        turn_id: int,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        event = Event.create(
            episode_id=episode_id,
            turn_id=turn_id,
            event_type=event_type,
            payload=payload,
        )
        self.events.append(event)
        return event

    def save(self, state: EpisodeState) -> Path:
        payload = {
            "episode_id": state.episode_id,
            "case": state.case,
            "state": state.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }
        return self.serializer.save(state.episode_id, payload)
