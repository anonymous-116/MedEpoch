from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

class RunCheckpoint:
    """Small atomic run journal shared by dataset and epoch entry points.

    ``checkpoint.json`` is the commit log for a run.  Large artifacts such as
    predictions and trajectories stay in their own files; this object records
    only run identity, completed work units, and lightweight progress.  The
    entry point uses the same ``--run-id/--resume`` rules for every Epoch run.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        identity: dict[str, Any],
        resume: bool,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "checkpoint.json"
        self.run_id = run_id
        self.identity = identity

        if resume:

            if not self.path.exists():
                raise FileNotFoundError(
                    f"Cannot resume run without checkpoint: {self.path}"
                )
            payload = self.read_json(self.path)
            previous = dict(payload.get("identity") or {})
            if previous != identity:
                if not self._is_safe_end_index_extension(previous, identity):
                    raise ValueError(
                        "Resume configuration does not match the original run. "
                        f"Expected {previous}, got {identity}."
                    )

                payload.setdefault("identity_history", []).append(previous)
                payload["identity"] = identity
            self.payload = payload
            self.payload["status"] = "running"
            self._save()
            return

        if self.path.exists() or (
            self.run_dir.exists() and any(self.run_dir.iterdir())
        ):
            raise FileExistsError(
                f"Run already exists; use --resume with the same --run-id: {run_id}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.payload = {
            "version": 1,
            "run_id": run_id,
            "status": "running",
            "identity": identity,
            "completed_units": [],
        }
        self._save()

    @staticmethod
    def _is_safe_end_index_extension(
        previous: dict[str, Any], current: dict[str, Any]
    ) -> bool:
        """Allow a resume to append a contiguous dataset suffix, and nothing else."""

        if set(previous) != set(current) or "end_index" not in previous:
            return False
        try:
            old_end = int(previous["end_index"])
            new_end = int(current["end_index"])
        except (TypeError, ValueError):
            return False
        return new_end > old_end and all(
            key == "end_index" or previous[key] == current[key]
            for key in previous
        )

    @property
    def completed_units(self) -> set[str]:
        return {str(value) for value in self.payload.get("completed_units") or []}

    def mark_completed(self, unit: str, **progress: Any) -> None:
        """Atomically commit one replay-safe unit (a row or an epoch batch)."""

        completed = self.completed_units
        completed.add(str(unit))
        self.payload["completed_units"] = sorted(completed)
        self.payload["progress"] = {**dict(self.payload.get("progress") or {}), **progress}
        self._save()

    def mark_run_complete(self, **summary: Any) -> None:
        self.payload["status"] = "complete"
        self.payload["summary"] = summary
        self._save()

    def write_records(self, path: str | Path, records: Iterable[dict[str, Any]]) -> None:
        """Replace a JSONL artifact atomically with the supplied records."""

        text = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
        self._atomic_write(Path(path), text)

    def write_json(self, path: str | Path, payload: dict[str, Any]) -> None:
        self._atomic_write(
            Path(path),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def read_records(path: str | Path) -> list[dict[str, Any]]:
        target = Path(path)
        if not target.exists():
            return []
        records: list[dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return records

    @staticmethod
    def read_json(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.write_json(self.path, self.payload)

    def _atomic_write(self, path: Path, text: str) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
