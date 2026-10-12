from __future__ import annotations

import json
from pathlib import Path
from typing import Any

class TrajectorySerializer:
    def __init__(self, trajectory_dir: str | Path) -> None:
        self.trajectory_dir = Path(trajectory_dir)

    def save(self, episode_id: str, payload: dict[str, Any]) -> Path:
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        path = self.trajectory_dir / f"{episode_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, episode_id: str) -> dict[str, Any]:
        path = self.trajectory_dir / f"{episode_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_episode_ids(self) -> list[str]:
        if not self.trajectory_dir.exists():
            return []
        return sorted(path.stem for path in self.trajectory_dir.glob("*.json"))
