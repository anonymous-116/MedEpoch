from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from uuid import uuid4

_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        path.with_name(path.name + suffix).unlink(missing_ok=True)

def snapshot_sqlite(src: str | Path, dst: str | Path) -> Path:
    source_path = Path(src)
    target = Path(dst)
    if not source_path.exists():
        raise FileNotFoundError(f"SQLite source database does not exist: {source_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
    source = sqlite3.connect(
        f"file:{source_path.resolve()}?mode=ro", uri=True, timeout=30
    )
    try:
        destination = sqlite3.connect(temporary, timeout=30)
        try:
            source.backup(destination)
            destination.commit()
            check = destination.execute("PRAGMA integrity_check").fetchone()
            if not check or check[0] != "ok":
                raise sqlite3.DatabaseError(
                    f"SQLite snapshot integrity check failed: {check}"
                )
        finally:
            destination.close()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        source.close()
    os.replace(temporary, target)
    _remove_sqlite_sidecars(target)
    return target
