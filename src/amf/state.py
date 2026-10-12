from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

@dataclass(slots=True)
class EpisodeState:
    episode_id: str
    case: dict[str, Any]
    turn_id: int = 0
    dialogue_history: list[dict[str, Any]] = field(default_factory=list)
    retrieved_contexts: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    status: str = "INIT"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, case: dict[str, Any], episode_id: str | None = None) -> "EpisodeState":
        return cls(episode_id=episode_id or str(uuid4()), case=case)

    def set_status(self, status: str) -> None:
        self.status = status

    def append_dialogue(self, role: str, content: str, **metadata: Any) -> None:
        self.dialogue_history.append(
            {"turn_id": self.turn_id, "role": role, "content": content, "metadata": metadata}
        )

    def append_retrieved_context(
        self,
        source: str,
        query: str,
        results: list[dict[str, Any]],
        **metadata: Any,
    ) -> None:
        self.retrieved_contexts.append(
            {
                "turn_id": self.turn_id,
                "source": source,
                "query": query,
                "results": results,
                "metadata": metadata,
            }
        )

    def append_trajectory(self, kind: str, content: Any, **metadata: Any) -> None:
        self.trajectory.append(
            {
                "turn_id": self.turn_id,
                "kind": kind,
                "content": content,
                "metadata": metadata,
            }
        )

    def append_action(self, action: dict[str, Any]) -> None:
        self.actions.append({"turn_id": self.turn_id, "action": action})

    def append_observation(self, observation: dict[str, Any]) -> None:
        self.observations.append({"turn_id": self.turn_id, "observation": observation})

    def set_final_answer(self, answer: str) -> None:
        self.final_answer = answer
        self.status = "EPISODE_END"

    def next_turn(self) -> None:
        self.turn_id += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
