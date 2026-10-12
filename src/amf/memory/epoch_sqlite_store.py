from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from amf.events import utc_now_iso
from amf.memory.item import KnowledgeItem
from amf.memory.provenance import source_ids_from_metadata
from amf.memory.sqlite_store import snapshot_sqlite
from amf.memory.store import BaseKnowledgeStore

class EpochSQLiteMemoryStore(BaseKnowledgeStore):
    """Versioned SQLite memory store for the Epoch method.

    The FTS5 table is deliberately only a search projection. Version state,
    provenance, and audit events live in ordinary SQLite tables so they can be
    filtered and updated without turning the search index into the repository.
    """

    VALID_MODES = {"append", "reset", "create", "resume", "read_only"}
    supports_versioning = True
    ACTIVE_LIFECYCLE_STATUSES = {"provisional", "active", "disputed"}
    LIFECYCLE_STATUSES = ACTIVE_LIFECYCLE_STATUSES | {"abandoned"}
    SOURCE_ROLES = {"content", "support", "conflict"}
    CONFLICT_STATUSES = {"pending", "deferred", "resolved"}
    CONFLICT_RESOLUTION_ACTIONS = {
        "restore",
        "update",
        "split",
        "defer",
        "abandon",
    }

    def __init__(self, path: str | Path, *, mode: str = "append") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Epoch SQLite mode must be one of {sorted(self.VALID_MODES)}.")
        self.path = Path(path)
        self.read_only = mode == "read_only"
        existed = self.path.exists()
        if mode == "create" and existed:
            raise FileExistsError(f"SQLite memory database already exists: {self.path}")
        if mode in {"resume", "read_only"} and not existed:
            raise FileNotFoundError(f"SQLite memory database does not exist: {self.path}")
        if mode in {"resume", "read_only"} and not self._is_initialized_database():
            raise ValueError(
                f"Path exists but is not an initialized Epoch memory database: {self.path}"
            )
        if self.read_only:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if mode == "reset":
            self.clear()

    async def add(self, item: KnowledgeItem) -> KnowledgeItem:
        self._require_writable()
        with self._connect() as connection:
            return self._insert_item_tx(connection, item)

    def _insert_item_tx(
        self,
        connection: sqlite3.Connection,
        item: KnowledgeItem,
        *,
        event_type: str = "add",
        event_payload: dict[str, Any] | None = None,
        skip_event: bool = False,
    ) -> KnowledgeItem:
        """Insert one version using an existing transaction.

        All compound Epoch actions use this helper instead of calling the
        public async ``add`` method.  That keeps FTS, source links, the version
        row, and its audit event in the same SQLite transaction.
        """

        metadata = dict(item.metadata)
        skip_add_event = bool(metadata.pop("_skip_add_event", False))
        item.metadata.pop("_skip_add_event", None)
        version_id = str(item.id)
        memory_id = str(metadata.get("memory_id") or version_id)
        version_state = str(metadata.get("version_state") or "current")

        lifecycle_status = str(metadata.get("lifecycle_status") or "provisional")
        if version_state not in {"current", "superseded"}:
            raise ValueError("version_state must be current or superseded")
        if lifecycle_status not in self.LIFECYCLE_STATUSES:
            raise ValueError(
                f"lifecycle_status must be one of {sorted(self.LIFECYCLE_STATUSES)}"
            )
        created_at = str(metadata.get("created_at") or utc_now_iso())
        epoch_id = self._optional_int(metadata.get("epoch_id"))
        batch_id = self._optional_int(metadata.get("batch_id"))
        metadata.update(
            {
                "memory_id": memory_id,
                "version_id": version_id,
                "version_state": version_state,
                "lifecycle_status": lifecycle_status,
            }
        )
        item.metadata.update(metadata)
        connection.execute(
            """
            INSERT INTO memory_versions (
                version_id, memory_id, parent_version_id, method, name, key,
                content, condition_text, category, tools_json,
                lifecycle_status, version_state, epoch_id, batch_id,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                memory_id,
                metadata.get("parent_version_id"),
                item.method,
                item.name,
                item.key,
                item.content,
                item.condition or "",
                item.category or "",
                json.dumps(item.tools, ensure_ascii=False),
                lifecycle_status,
                version_state,
                epoch_id,
                batch_id,
                json.dumps(metadata, ensure_ascii=False),
                created_at,
                metadata.get("updated_at"),
            ),
        )
        connection.execute(
            "INSERT INTO memory_fts(version_id, name, content, condition_text) VALUES (?, ?, ?, ?)",
            (version_id, item.name, item.content, item.condition or ""),
        )
        self._insert_sources(connection, version_id, metadata)
        if not skip_event and not skip_add_event:
            self._insert_event(
                connection,
                event_type=event_type,
                version_id=version_id,
                memory_id=memory_id,

                source_id=(
                    str(metadata.get("source_id"))
                    if metadata.get("source_id") is not None
                    else None
                ),
                epoch_id=epoch_id,
                batch_id=batch_id,
                payload=event_payload or {"item": item.to_dict()},
            )
        return item

    async def get(self, item_id: str) -> KnowledgeItem | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_versions WHERE version_id = ? LIMIT 1",
                (str(item_id),),
            ).fetchone()
        return self._row_to_item(row) if row is not None else None

    async def list(self, filters: dict[str, Any] | None = None) -> list[KnowledgeItem]:
        filters = dict(filters or {})
        clauses, parameters = self._filter_sql(filters, table_alias="memory_versions")
        sql = "SELECT memory_versions.* FROM memory_versions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY memory_versions.created_at, memory_versions.version_id"
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._row_to_item(row) for row in rows]

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[KnowledgeItem]:
        if top_k <= 0:
            return []
        filters = dict(filters or {})
        match_query = self._match_query(query)
        clauses, parameters = self._filter_sql(filters, table_alias="v")
        where = ["v.version_id = f.version_id", *clauses]
        sql = (
            "SELECT v.*, bm25(memory_fts) AS bm25_score "
            "FROM memory_fts AS f JOIN memory_versions AS v "
            "ON v.version_id = f.version_id"
        )
        if match_query:
            where.insert(0, "memory_fts MATCH ?")
            parameters.insert(0, match_query)
        sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY bm25_score ASC LIMIT ?"
        parameters.append(top_k)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        items: list[KnowledgeItem] = []
        for row in rows:
            item = self._row_to_item(row)
            item.metadata["retrieval_score"] = -float(row["bm25_score"] or 0.0)
            items.append(item)
        return items

    async def update(self, item_id: str, patch: dict[str, Any]) -> KnowledgeItem | None:
        """Create a new current version and supersede the old one atomically."""

        self._require_writable()
        old = await self.get(item_id)
        if old is None:
            return None
        if str(old.metadata.get("version_state") or "current") != "current":
            return None
        if str(old.metadata.get("lifecycle_status") or "") == "abandoned":
            return None
        payload = old.to_dict()
        patch = dict(patch)
        patch_metadata = dict(patch.pop("metadata", {}) or {})
        patch.pop("id", None)
        payload.update(patch)
        metadata = dict(payload.get("metadata") or {})
        metadata.update(patch_metadata)
        old_metadata = dict(old.metadata)
        memory_id = str(old_metadata.get("memory_id") or old.id)

        new_version_id = str(patch_metadata.get("version_id") or uuid4())
        metadata.update(
            {
                "memory_id": memory_id,
                "version_id": new_version_id,
                "parent_version_id": old.id,
                "version_state": "current",
                "updated_from": old.id,
                "updated_at": utc_now_iso(),
                "lifecycle_status": str(
                    metadata.get("lifecycle_status")
                    or old_metadata.get("lifecycle_status")
                    or "provisional"
                ),
            }
        )
        if metadata["lifecycle_status"] not in self.LIFECYCLE_STATUSES:
            raise ValueError(
                f"lifecycle_status must be one of {sorted(self.LIFECYCLE_STATUSES)}"
            )
        payload["id"] = new_version_id
        payload["metadata"] = metadata
        new_item = KnowledgeItem.from_dict(payload)

        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_versions SET version_state = 'superseded', updated_at = ? WHERE version_id = ?",
                (metadata["updated_at"], old.id),
            )
            connection.execute(
                """
                INSERT INTO memory_versions (
                    version_id, memory_id, parent_version_id, method, name, key,
                    content, condition_text, category, tools_json,
                    lifecycle_status, version_state, epoch_id, batch_id,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_version_id,
                    memory_id,
                    old.id,
                    new_item.method,
                    new_item.name,
                    new_item.key,
                    new_item.content,
                    new_item.condition or "",
                    new_item.category or "",
                    json.dumps(new_item.tools, ensure_ascii=False),
                    metadata["lifecycle_status"],
                    "current",
                    self._optional_int(metadata.get("epoch_id")),
                    self._optional_int(metadata.get("batch_id")),
                    json.dumps(metadata, ensure_ascii=False),
                    str(metadata.get("created_at") or utc_now_iso()),
                    metadata["updated_at"],
                ),
            )
            connection.execute(
                "INSERT INTO memory_fts(version_id, name, content, condition_text) VALUES (?, ?, ?, ?)",
                (new_version_id, new_item.name, new_item.content, new_item.condition or ""),
            )

            self._copy_sources(connection, old.id, new_version_id)
            self._insert_sources(connection, new_version_id, metadata)
            self._insert_event(
                connection,
                event_type="update",
                version_id=new_version_id,
                memory_id=memory_id,
                source_id=(
                    str(metadata.get("source_id"))
                    if metadata.get("source_id") is not None
                    else None
                ),
                epoch_id=self._optional_int(metadata.get("epoch_id")),
                batch_id=self._optional_int(metadata.get("batch_id")),
                payload={"from_version_id": old.id, "item": new_item.to_dict()},
            )

        return await self.get(new_version_id)

    async def record_use(
        self,
        version_id: str,
        source_id: str,
        label: str,
        *,
        evidence: str | None = None,
        epoch_id: int | None = None,
        batch_id: int | None = None,
        score_before: float | None = None,
        score_after: float | None = None,
    ) -> None:
        """Record retrieval/use/helpfulness without changing memory content."""

        self._require_writable()
        if label not in {"helpful", "harmful", "neutral", "uncertain"}:
            raise ValueError("memory use label must be helpful, harmful, neutral, or uncertain")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT memory_id FROM memory_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory version: {version_id}")
            duplicate = connection.execute(
                """
                SELECT 1 FROM memory_events
                WHERE event_type = 'use' AND version_id = ? AND source_id = ?
                  AND label = ? AND (epoch_id = ? OR (epoch_id IS NULL AND ? IS NULL))
                LIMIT 1
                """,
                (version_id, source_id, label, epoch_id, epoch_id),
            ).fetchone()
            if duplicate is not None:
                return
            self._insert_event(
                connection,
                event_type="use",
                version_id=version_id,
                memory_id=str(row["memory_id"]),
                source_id=source_id,
                epoch_id=epoch_id,
                batch_id=batch_id,
                label=label,
                payload={
                    "evidence": evidence,
                    "score_before": score_before,
                    "score_after": score_after,
                },
            )
            if label == "helpful":
                connection.execute(
                    "UPDATE memory_versions SET lifecycle_status = 'active' WHERE version_id = ? AND version_state = 'current'",
                    (version_id,),
                )
            elif label == "harmful":

                pass

    async def confirm(
        self,
        version_id: str,
        source_id: str,
        *,
        evidence: str | None = None,
        epoch_id: int | None = None,
        batch_id: int | None = None,
    ) -> KnowledgeItem | None:
        """Add independent support and activate a provisional Memory."""

        self._require_writable()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT memory_id, lifecycle_status FROM memory_versions "
                "WHERE version_id = ? AND version_state = 'current'",
                (version_id,),
            ).fetchone()
            if row is None:
                return None
            if str(row["lifecycle_status"]) == "abandoned":
                return None
            same_source = connection.execute(
                "SELECT 1 FROM memory_sources WHERE version_id = ? AND source_role = 'content' AND source_id = ?",
                (version_id, source_id),
            ).fetchone()
            if same_source is not None:
                return await self.get(version_id)
            cursor = connection.execute(
                "INSERT OR IGNORE INTO memory_sources(version_id, source_id, source_role, evidence, created_at) VALUES (?, ?, 'support', ?, ?)",
                (version_id, source_id, evidence, utc_now_iso()),
            )
            if cursor.rowcount == 0:
                return await self.get(version_id)
            connection.execute(
                "UPDATE memory_versions SET lifecycle_status = 'active' WHERE version_id = ?",
                (version_id,),
            )
            self._insert_event(
                connection,
                event_type="confirm",
                version_id=version_id,
                memory_id=str(row["memory_id"]),
                source_id=source_id,
                epoch_id=epoch_id,
                batch_id=batch_id,
                payload={"evidence": evidence},
            )
        return await self.get(version_id)

    async def dispute(
        self,
        version_ids: list[str],
        source_id: str,
        *,
        evidence: str | None = None,
        epoch_id: int | None = None,
        batch_id: int | None = None,
    ) -> list[KnowledgeItem]:
        """Mark direct conflicts without choosing a winner from one case."""

        self._require_writable()
        updated: list[str] = []
        with self._connect() as connection:
            for version_id in dict.fromkeys(version_ids):
                row = connection.execute(
                    "SELECT memory_id, lifecycle_status FROM memory_versions "
                    "WHERE version_id = ? AND version_state = 'current'",
                    (version_id,),
                ).fetchone()
                if row is None:
                    continue
                if str(row["lifecycle_status"]) == "abandoned":
                    continue
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO memory_sources(version_id, source_id, source_role, evidence, created_at) VALUES (?, ?, 'conflict', ?, ?)",
                    (version_id, source_id, evidence, utc_now_iso()),
                )
                if cursor.rowcount == 0:
                    updated.append(version_id)
                    continue
                connection.execute(
                    "UPDATE memory_versions SET lifecycle_status = 'disputed' WHERE version_id = ?",
                    (version_id,),
                )
                self._insert_event(
                    connection,
                    event_type="dispute",
                    version_id=version_id,
                    memory_id=str(row["memory_id"]),
                    source_id=source_id,
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                    payload={"evidence": evidence},
                )
                self._insert_conflict_tx(
                    connection,
                    target_version_id=version_id,
                    candidate_summary=None,
                    source_id=source_id,
                    reason=evidence,
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                )
                updated.append(version_id)
        return [item for item in [await self.get(value) for value in updated] if item]

    async def merge(
        self,
        target_ids: list[str],
        item: KnowledgeItem,
    ) -> KnowledgeItem:
        """Create one Memory from several current versions and inherit sources."""

        self._require_writable()
        targets = [value for value in dict.fromkeys(target_ids) if value]
        if len(targets) < 2:
            raise ValueError("merge requires at least two target versions")
        with self._connect() as connection:
            self._validate_current_targets_tx(connection, targets, operation="merge")
            saved = self._merge_tx(
                connection,
                targets,
                item,
                epoch_id=self._optional_int(item.metadata.get("epoch_id")),
                batch_id=self._optional_int(item.metadata.get("batch_id")),
                event_payload={"target_ids": targets},
            )
        return await self.get(saved.id) or saved

    async def split(
        self,
        target_ids: list[str] | str,
        items: list[KnowledgeItem],
    ) -> list[KnowledgeItem]:
        """Replace one or more summaries with conditional child summaries."""

        self._require_writable()
        if len(items) < 2:
            raise ValueError("split requires at least two replacement memories")
        if isinstance(target_ids, str):
            targets = [target_ids]
        else:
            targets = [str(value) for value in target_ids if value]
        targets = list(dict.fromkeys(targets))
        if not targets:
            raise ValueError("split requires at least one target version")
        with self._connect() as connection:
            self._validate_current_targets_tx(connection, targets, operation="split")
            saved_items = self._split_tx(
                connection,
                targets,
                items,
                epoch_id=self._optional_int(items[0].metadata.get("epoch_id")),
                batch_id=self._optional_int(items[0].metadata.get("batch_id")),
                event_payload={"target_ids": targets},
            )
        return [await self.get(item.id) or item for item in saved_items]

    async def apply_actions(
        self,
        actions: list[dict[str, Any]],
        *,
        method: str = "epoch",
        source_id: str | None = None,
        epoch_id: int | None = None,
        batch_id: int | None = None,
        candidate_summary: dict[str, Any] | None = None,
        information_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and commit a complete action list atomically.

        The writer and provider layers intentionally exchange plain action
        dictionaries instead of a ``MemoryWriteDecision`` object.  This keeps
        the database boundary independent of any particular LLM writer.  All
        target checks happen before the first write; one malformed action
        therefore rolls back the entire list.
        """

        self._require_writable()
        if not isinstance(actions, list) or not actions:
            raise ValueError("apply_actions requires a non-empty actions list")
        normalized = self._normalize_actions(actions)
        with self._connect() as connection:

            connection.execute("BEGIN IMMEDIATE")
            prepared = self._prepare_action_items(
                connection,
                normalized,
                method=method,
                source_id=source_id,
                epoch_id=epoch_id,
                batch_id=batch_id,
            )
            self._mark_replayed_actions(connection, prepared)
            self._validate_action_set(
                connection,
                prepared,
                method=method,
                source_id=source_id,
            )
            committed: list[dict[str, Any]] = []

            ordered = sorted(prepared, key=lambda pair: pair[0]["action"] in {"update", "merge", "split"})
            for action, items in ordered:
                committed.append(
                    self._apply_action_tx(
                        connection,
                        action,
                        items,
                        method=method,
                        source_id=source_id,
                        epoch_id=epoch_id,
                        batch_id=batch_id,
                        candidate_summary=candidate_summary,
                        information_usage=information_usage,
                    )
                )
                if action.get("candidate_ids") and action["action"] in {"update", "merge", "split"} and not action.get("_reused_result"):

                    for version_id in committed[-1].get("created_version_ids", []):
                        connection.execute("DELETE FROM memory_sources WHERE version_id = ? AND source_role = 'support'", (version_id,))
                        row = connection.execute("SELECT metadata_json FROM memory_versions WHERE version_id = ?", (version_id,)).fetchone()
                        metadata = json.loads(row["metadata_json"] or "{}")
                        metadata.update(support_source_ids=[], lifecycle_status="provisional")
                        connection.execute("UPDATE memory_versions SET lifecycle_status = 'provisional', metadata_json = ? WHERE version_id = ?", (json.dumps(metadata, ensure_ascii=False), version_id))
            return {
                "committed": True,
                "actions": committed,
                "epoch_id": epoch_id,
                "batch_id": batch_id,
                "source_id": source_id,
            }

    def _mark_replayed_actions(
        self,
        connection: sqlite3.Connection,
        prepared: list[tuple[dict[str, Any], list[KnowledgeItem]]],
    ) -> None:
        """Mark fully committed checkpointed actions as safe no-ops.

        A process can die after SQLite commits but before the outer run
        checkpoint is written.  On resume, generated replacement items carry
        the same checkpoint keys; recognizing a complete set here avoids
        duplicating an ADD or attempting UPDATE against an already superseded
        target.  Non-checkpointed actions retain normal validation semantics.
        """

        for action, items in prepared:

            action_key = str(action.get("checkpoint_key") or "").strip()
            if action_key:
                event_rows = connection.execute(
                    """
                    SELECT event_type, version_id
                    FROM memory_events
                    WHERE json_extract(payload_json, '$.checkpoint_key') = ?
                    ORDER BY created_at
                    """,
                    (action_key,),
                ).fetchall()
                if event_rows:
                    version_ids = [
                        str(row["version_id"])
                        for row in event_rows
                        if row["version_id"] is not None
                    ]
                    action["_reused_result"] = {
                        "action": action["action"],
                        "target_ids": list(action.get("target_ids") or []),
                        "version_ids": list(dict.fromkeys(version_ids)),
                        "created_version_ids": (
                            list(dict.fromkeys(version_ids))
                            if action["action"]
                            in {"add", "update", "merge", "split"}
                            else []
                        ),
                        "reused": True,
                    }
                    continue
            if action.get("action") not in {"add", "update", "merge", "split"}:
                continue
            keys = [
                str(item.metadata.get("checkpoint_key") or "").strip()
                for item in items
            ]
            if not keys or any(not key for key in keys):
                continue
            existing_ids: list[str] = []
            for key in keys:
                row = connection.execute(
                    """
                    SELECT version_id
                    FROM memory_versions
                    WHERE json_extract(metadata_json, '$.checkpoint_key') = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (key,),
                ).fetchone()
                if row is None:
                    existing_ids = []
                    break
                existing_ids.append(str(row["version_id"]))
            if len(existing_ids) == len(keys):
                action["_reused_result"] = {
                    "action": action["action"],
                    "target_ids": list(action.get("target_ids") or []),
                    "version_ids": existing_ids,
                    "created_version_ids": existing_ids,
                    "reused": True,
                }

    @staticmethod
    def _normalize_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for raw in actions:
            if not isinstance(raw, dict):
                raise ValueError("Each memory action must be a JSON object")
            action = str(raw.get("action") or "").strip().lower()
            if action not in {
                "add",
                "keep",
                "confirm",
                "update",
                "merge",
                "split",
                "dispute",
                "reject",
            }:
                raise ValueError(f"Unsupported memory action: {action or '<empty>'}")
            value = dict(raw)
            value["action"] = action
            target_ids = value.get("target_ids")
            if target_ids is None and value.get("target_id") is not None:
                target_ids = [value.get("target_id")]
            if isinstance(target_ids, str):
                target_ids = [target_ids]
            if not isinstance(target_ids, list):
                target_ids = []
            value["target_ids"] = list(
                dict.fromkeys(str(target).strip() for target in target_ids if str(target).strip())
            )
            summaries = value.get("summaries")
            if summaries is None:
                summaries = value.get("items")
            if summaries is None:
                summaries = []
            if isinstance(summaries, dict):
                summaries = [summaries]
            if not isinstance(summaries, list):
                raise ValueError(f"{action} summaries must be a list")
            value["summaries"] = summaries
            normalized.append(value)
        return normalized

    def _prepare_action_items(
        self,
        connection: sqlite3.Connection,
        actions: list[dict[str, Any]],
        *,
        method: str,
        source_id: str | None,
        epoch_id: int | None,
        batch_id: int | None,
    ) -> list[tuple[dict[str, Any], list[KnowledgeItem]]]:
        prepared: list[tuple[dict[str, Any], list[KnowledgeItem]]] = []
        for action in actions:
            action_source_ids = [
                str(value).strip()
                for value in action.get("source_ids") or []
                if str(value).strip()
            ]
            item_source_id = action_source_ids[0] if action_source_ids else source_id
            items: list[KnowledgeItem] = []
            for payload in action["summaries"]:
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"{action['action']} summaries must contain JSON objects"
                    )
                item = self._item_from_action_payload(
                    payload,
                    method=method,
                    source_id=item_source_id,
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                )
                if item is None:
                    raise ValueError(
                        f"{action['action']} contains an incomplete summary"
                    )
                items.append(item)
            prepared.append((action, items))
        return prepared

    def _validate_action_set(
        self,
        connection: sqlite3.Connection,
        prepared: list[tuple[dict[str, Any], list[KnowledgeItem]]],
        *,
        method: str,
        source_id: str | None,
    ) -> None:
        current_rows = connection.execute(
            "SELECT version_id, method, lifecycle_status FROM memory_versions "
            "WHERE version_state = 'current'"
        ).fetchall()
        current_status = {
            str(row["version_id"]): str(row["lifecycle_status"])
            for row in current_rows
        }
        current_methods = {
            str(row["version_id"]): str(row["method"])
            for row in current_rows
        }
        current_ids = set(current_status)
        touched: dict[str, list[str]] = {}
        for action, items in prepared:
            if action.get("_reused_result") is not None:
                continue
            name = action["action"]
            target_ids = action["target_ids"]
            if len(target_ids) != len(set(target_ids)):
                raise ValueError(f"{name} contains duplicate target_ids")
            for target_id in target_ids:
                if target_id not in current_ids:
                    raise ValueError(
                        f"{name} targets unknown or non-current version: {target_id}"
                    )

                action_method = str(method or "").strip()
                if action_method and current_methods[target_id] != action_method:
                    raise ValueError(
                        f"{name} target {target_id} belongs to method "
                        f"{current_methods[target_id]!r}, not {action_method!r}"
                    )
                if current_status[target_id] == "abandoned":
                    raise ValueError(
                        f"{name} cannot target an abandoned version: {target_id}"
                    )
                previous = touched.get(target_id)
                if previous is not None and not action.get("candidate_ids"):
                    raise ValueError(
                        f"target {target_id} is used more than once "
                        f"({previous}, {name})"
                    )
                from amf.memory.window_validation import combination_error
                names = [*(previous or []), name]
                error = combination_error(names)
                if error:
                    raise ValueError(error)
                touched[target_id] = names

            summary_count = len(items)
            if name == "add":
                if target_ids or summary_count != 1:
                    raise ValueError("add requires no targets and exactly one summary")
            elif name == "keep":
                if not target_ids or summary_count:
                    raise ValueError("keep requires targets and no replacement summaries")
            elif name == "confirm":
                if not target_ids or summary_count:
                    raise ValueError(
                        "confirm requires targets and no replacement summaries"
                    )
                action_source_ids = [
                    str(value).strip()
                    for value in action.get("source_ids") or []
                    if str(value).strip()
                ] or ([source_id] if source_id else [])
                if action_source_ids:
                    placeholders = ",".join("?" for _ in target_ids)
                    source_placeholders = ",".join("?" for _ in action_source_ids)
                    rows = connection.execute(
                        f"""
                        SELECT version_id, source_id FROM memory_sources
                        WHERE version_id IN ({placeholders})
                          AND source_role = 'content'
                          AND source_id IN ({source_placeholders})
                        """,
                        [*target_ids, *action_source_ids],
                    ).fetchall()
                    same_source_by_target: dict[str, set[str]] = {}
                    for row in rows:
                        same_source_by_target.setdefault(
                            str(row["version_id"]), set()
                        ).add(str(row["source_id"]))
                    if any(
                        set(action_source_ids).issubset(
                            same_source_by_target.get(target_id, set())
                        )
                        for target_id in target_ids
                    ):
                        raise ValueError(
                            "confirm requires an independent source for every target"
                        )
            elif name == "update":
                if len(target_ids) != 1 or summary_count != 1:
                    raise ValueError("update requires one target and one summary")
            elif name == "merge":
                if len(target_ids) < 2 or summary_count != 1:
                    raise ValueError("merge requires at least two targets and one summary")
            elif name == "split":
                if not target_ids or summary_count < 2:
                    raise ValueError("split requires targets and at least two summaries")
                if any(not (item.condition or "").strip() for item in items):
                    raise ValueError("split summaries must have applicable conditions")
            elif name == "dispute":
                if not target_ids or summary_count:
                    raise ValueError(
                        "dispute requires targets and no replacement summaries"
                    )
            elif name == "reject":
                if target_ids or summary_count:
                    raise ValueError("reject must not target or create summaries")

    def _apply_action_tx(
        self,
        connection: sqlite3.Connection,
        action: dict[str, Any],
        items: list[KnowledgeItem],
        *,
        method: str,
        source_id: str | None,
        epoch_id: int | None,
        batch_id: int | None,
        candidate_summary: dict[str, Any] | None,
        information_usage: dict[str, Any] | None,
    ) -> dict[str, Any]:
        name = action["action"]
        targets = list(action["target_ids"])
        action_source_ids = [
            str(value).strip()
            for value in action.get("source_ids") or []
            if str(value).strip()
        ] or ([source_id] if source_id else [])
        action_source_id = action_source_ids[0] if action_source_ids else None
        reused_result = action.get("_reused_result")
        if isinstance(reused_result, dict):
            return dict(reused_result)
        reason = str(action.get("reason") or "")
        audit_payload = {
            "action": name,
            "target_ids": targets,
            "reason": reason,
            "checkpoint_key": action.get("checkpoint_key"),
            "candidate_ids": list(action.get("candidate_ids") or []),
            "source_ids": action_source_ids,
            "candidate_summary": candidate_summary,
            "information_usage": information_usage,
            "summaries": [
                {
                    "summary": item.content,
                    "applicable_conditions": item.condition,
                }
                for item in items
            ],
        }

        if name == "add":
            saved = self._insert_item_tx(
                connection,
                items[0],
                event_payload=audit_payload,
            )
            return {
                "action": name,
                "target_ids": targets,
                "version_ids": [saved.id],
                "created_version_ids": [saved.id],
            }

        if name == "keep":
            for target_id in targets:
                row = connection.execute(
                    "SELECT memory_id FROM memory_versions WHERE version_id = ?",
                    (target_id,),
                ).fetchone()
                self._insert_event(
                    connection,
                    event_type="keep",
                    version_id=target_id,
                    memory_id=str(row["memory_id"]) if row else None,
                    source_id=action_source_id,
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                    payload=audit_payload,
                )
            return {
                "action": name,
                "target_ids": targets,
                "version_ids": [],
                "created_version_ids": [],
            }

        if name == "confirm":
            confirmed: list[str] = []
            for target_id in targets:
                row = connection.execute(
                    "SELECT memory_id FROM memory_versions WHERE version_id = ?",
                    (target_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Unknown confirm target: {target_id}")
                content_rows = connection.execute(
                    "SELECT source_id FROM memory_sources "
                    "WHERE version_id = ? AND source_role = 'content'",
                    (target_id,),
                ).fetchall()
                content_sources = {str(value["source_id"]) for value in content_rows}
                independent_sources = [
                    value for value in action_source_ids if value not in content_sources
                ]
                for independent_source in independent_sources:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO memory_sources(
                            version_id, source_id, source_role, evidence, created_at
                        ) VALUES (?, ?, 'support', ?, ?)
                        """,
                        (target_id, independent_source, reason, utc_now_iso()),
                    )
                connection.execute(
                    "UPDATE memory_versions SET lifecycle_status = 'active' "
                    "WHERE version_id = ? AND version_state = 'current'",
                    (target_id,),
                )
                self._insert_event(
                    connection,
                    event_type="confirm",
                    version_id=target_id,
                    memory_id=str(row["memory_id"]),
                    source_id=(independent_sources[0] if independent_sources else action_source_id),
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                    payload=audit_payload,
                )
                confirmed.append(target_id)
            return {
                "action": name,
                "target_ids": targets,
                "version_ids": confirmed,
                "created_version_ids": [],
            }

        if name == "update":
            saved = self._update_tx(
                connection,
                targets[0],
                items[0],
                epoch_id=epoch_id,
                batch_id=batch_id,
                event_payload=audit_payload,
            )
            return {
                "action": name,
                "target_ids": targets,
                "version_ids": [saved.id],
                "created_version_ids": [saved.id],
            }

        if name == "merge":
            saved = self._merge_tx(
                connection,
                targets,
                items[0],
                epoch_id=epoch_id,
                batch_id=batch_id,
                event_payload=audit_payload,
            )
            return {
                "action": name,
                "target_ids": targets,
                "version_ids": [saved.id],
                "created_version_ids": [saved.id],
            }

        if name == "split":
            saved_items = self._split_tx(
                connection,
                targets,
                items,
                epoch_id=epoch_id,
                batch_id=batch_id,
                event_payload=audit_payload,
            )
            return {
                "action": name,
                "target_ids": targets,
                "version_ids": [item.id for item in saved_items],
                "created_version_ids": [item.id for item in saved_items],
            }

        if name == "dispute":
            disputed: list[str] = []
            for conflict_source in action_source_ids or [None]:
                disputed.extend(
                    self._dispute_tx(
                        connection,
                        targets,
                        source_id=conflict_source,
                        epoch_id=epoch_id,
                        batch_id=batch_id,
                        evidence=reason,
                        payload=audit_payload,
                    )
                )
            disputed = list(dict.fromkeys(disputed))
            return {
                "action": name,
                "target_ids": targets,
                "version_ids": disputed,
                "created_version_ids": [],
            }

        if name == "reject":
            self._insert_event(
                connection,
                event_type="reject",
                source_id=action_source_id,
                epoch_id=epoch_id,
                batch_id=batch_id,
                payload=audit_payload,
            )
            return {
                "action": name,
                "target_ids": [],
                "version_ids": [],
                "created_version_ids": [],
            }

        raise ValueError(f"Unsupported action: {name}")

    @staticmethod
    def _item_from_action_payload(
        payload: dict[str, Any],
        *,
        method: str,
        source_id: str | None,
        epoch_id: int | None,
        batch_id: int | None,
    ) -> KnowledgeItem | None:
        """Build a repository item from either model or internal action shape."""

        explicit_id = payload.get("id")
        existing_item: KnowledgeItem | None = None
        condition_value: Any = None
        if (
            explicit_id is not None
            and isinstance(payload.get("content"), str)
            and isinstance(payload.get("key"), str)
        ):
            existing_item = KnowledgeItem.from_dict(payload)
            content = existing_item.content.strip()
            condition = str(existing_item.condition or "").strip()
            if not content or not condition:
                return None
            metadata = dict(existing_item.metadata)
            name = existing_item.name
            key = existing_item.key
            category = existing_item.category
            tools = existing_item.tools
            condition_value = condition
        elif isinstance(payload.get("content"), str) and isinstance(
            payload.get("key"), str
        ):
            content = str(payload["content"]).strip()
            key = str(payload["key"]).strip()
            condition_value = payload.get("condition")
            name = str(payload.get("name") or key).strip()
            metadata = dict(payload.get("metadata") or {})
            category = payload.get("category")
            tools = [str(value) for value in payload.get("tools") or []]
        else:
            content = str(payload.get("summary") or "").strip()
            key = " ".join(content.split())[:240]
            name = content.split(".", 1)[0].strip()[:120] or key[:120]
            condition_value = payload.get("applicable_conditions")
            metadata = dict(payload.get("metadata") or {})
            category = payload.get("category")
            tools = [str(value) for value in payload.get("tools") or []]
        condition = (
            str(condition_value).strip()
            if condition_value is not None
            else ""
        )
        if not content or not key or not condition:
            return None
        metadata.setdefault("created_at", utc_now_iso())
        metadata.update(
            {
                "method": method,
                "epoch_id": epoch_id,
                "batch_id": batch_id,
            }
        )
        if source_id:
            metadata["source_id"] = source_id
            for role in ("content", "support", "conflict"):
                values = source_ids_from_metadata(metadata, role)
                if role == "content" and source_id not in values:
                    values.append(source_id)
                metadata[f"{role}_source_ids"] = values
        if existing_item is not None:
            existing_item.metadata.update(metadata)
            existing_item.method = method
            existing_item.name = name
            existing_item.key = key
            existing_item.content = content
            existing_item.condition = condition
            existing_item.category = category
            existing_item.tools = tools
            return existing_item
        return KnowledgeItem.create(
            method=method,
            name=name,
            key=key,
            content=content,
            condition=condition,
            category=str(category).strip() if category is not None else None,
            tools=tools,
            metadata=metadata,
        )

    def _source_lists_tx(
        self,
        connection: sqlite3.Connection,
        version_ids: list[str],
    ) -> dict[str, list[str]]:
        result = {role: [] for role in self.SOURCE_ROLES}
        if not version_ids:
            return result
        placeholders = ",".join("?" for _ in version_ids)
        rows = connection.execute(
            f"""
            SELECT source_id, source_role
            FROM memory_sources
            WHERE version_id IN ({placeholders})
            ORDER BY source_id, source_role
            """,
            version_ids,
        ).fetchall()
        for row in rows:
            role = str(row["source_role"])
            value = str(row["source_id"])
            if value not in result[role]:
                result[role].append(value)
        return result

    def _validate_current_targets_tx(
        self,
        connection: sqlite3.Connection,
        target_ids: list[str],
        *,
        operation: str,
    ) -> None:
        if not target_ids:
            raise ValueError(f"{operation} requires at least one target version")
        placeholders = ",".join("?" for _ in target_ids)
        rows = connection.execute(
            f"""
            SELECT version_id
            FROM memory_versions
            WHERE version_state = 'current'
              AND lifecycle_status <> 'abandoned'
              AND version_id IN ({placeholders})
            """,
            target_ids,
        ).fetchall()
        found = {str(row["version_id"]) for row in rows}
        missing = [value for value in target_ids if value not in found]
        if missing:
            raise ValueError(
                f"{operation} targets unknown or non-current versions: {missing}"
            )

    def _update_tx(
        self,
        connection: sqlite3.Connection,
        target_id: str,
        item: KnowledgeItem,
        *,
        epoch_id: int | None,
        batch_id: int | None,
        event_payload: dict[str, Any],
    ) -> KnowledgeItem:
        row = connection.execute(
            "SELECT * FROM memory_versions WHERE version_id = ? AND version_state = 'current'",
            (target_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"update targets unknown or non-current version: {target_id}")
        old_metadata = json.loads(row["metadata_json"] or "{}")
        source_map = self._source_lists_tx(connection, [target_id])
        new_metadata = dict(old_metadata)
        new_metadata.update(item.metadata)
        new_version_id = str(item.id)
        new_metadata.update(
            {
                "memory_id": str(row["memory_id"]),
                "version_id": new_version_id,
                "parent_version_id": target_id,
                "parent_version_ids": [target_id],
                "version_state": "current",
                "updated_from": target_id,
                "updated_at": utc_now_iso(),
                "epoch_id": epoch_id,
                "batch_id": batch_id,
                "content_source_ids": list(
                    dict.fromkeys(
                        [
                            *source_map["content"],
                            *source_ids_from_metadata(item.metadata, "content"),
                        ]
                    )
                ),
                "support_source_ids": source_map["support"],
                "conflict_source_ids": source_map["conflict"],
            }
        )
        new_item = KnowledgeItem.from_dict(
            {
                **item.to_dict(),
                "id": new_version_id,
                "method": item.method or str(row["method"]),
                "metadata": new_metadata,
            }
        )
        old_metadata.update(
            {
                "updated_at": new_metadata["updated_at"],
                "updated_to": new_version_id,
            }
        )
        connection.execute(
            """
            UPDATE memory_versions
            SET version_state = 'superseded', updated_at = ?, metadata_json = ?
            WHERE version_id = ?
            """,
            (
                new_metadata["updated_at"],
                json.dumps(old_metadata, ensure_ascii=False),
                target_id,
            ),
        )
        saved = self._insert_item_tx(
            connection,
            new_item,
            event_type="update",
            event_payload={
                **event_payload,
                "from_version_id": target_id,
                "item": new_item.to_dict(),
            },
        )

        self._copy_sources(connection, target_id, new_version_id)
        return saved

    def _merge_tx(
        self,
        connection: sqlite3.Connection,
        target_ids: list[str],
        item: KnowledgeItem,
        *,
        epoch_id: int | None,
        batch_id: int | None,
        event_payload: dict[str, Any],
    ) -> KnowledgeItem:
        source_map = self._source_lists_tx(connection, target_ids)
        metadata = dict(item.metadata)
        metadata.update(
            {
                "_skip_add_event": True,
                "parent_version_ids": list(target_ids),
                "parent_version_id": target_ids[0],
                "content_source_ids": list(
                    dict.fromkeys(
                        [
                            *source_map["content"],
                            *source_ids_from_metadata(item.metadata, "content"),
                        ]
                    )
                ),
                "support_source_ids": source_map["support"],
                "conflict_source_ids": source_map["conflict"],
                "epoch_id": epoch_id,
                "batch_id": batch_id,
            }
        )
        item.metadata.update(metadata)
        saved = self._insert_item_tx(
            connection,
            item,
            event_type="merge",
            event_payload={
                **event_payload,
                "target_ids": list(target_ids),
                "item": item.to_dict(),
            },
        )
        now = utc_now_iso()
        placeholders = ",".join("?" for _ in target_ids)
        connection.execute(
            f"""
            UPDATE memory_versions
            SET version_state = 'superseded', updated_at = ?
            WHERE version_id IN ({placeholders})
            """,
            [now, *target_ids],
        )

        self._insert_event(
            connection,
            event_type="merge",
            version_id=saved.id,
            memory_id=str(saved.metadata.get("memory_id") or saved.id),
            source_id=(
                str(saved.metadata.get("source_id"))
                if saved.metadata.get("source_id") is not None
                else None
            ),
            epoch_id=epoch_id,
            batch_id=batch_id,
            payload={
                **event_payload,
                "target_ids": list(target_ids),
                "new_version_id": saved.id,
            },
        )
        return saved

    def _split_tx(
        self,
        connection: sqlite3.Connection,
        target_ids: list[str],
        items: list[KnowledgeItem],
        *,
        epoch_id: int | None,
        batch_id: int | None,
        event_payload: dict[str, Any],
    ) -> list[KnowledgeItem]:
        source_map = self._source_lists_tx(connection, target_ids)
        saved_items: list[KnowledgeItem] = []
        for item in items:
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "_skip_add_event": True,
                    "parent_version_ids": list(target_ids),
                    "parent_version_id": target_ids[0],
                    "content_source_ids": list(
                        dict.fromkeys(
                            [
                                *source_map["content"],
                                *source_ids_from_metadata(item.metadata, "content"),
                            ]
                        )
                    ),
                    "support_source_ids": source_map["support"],
                    "conflict_source_ids": source_map["conflict"],
                    "epoch_id": epoch_id,
                    "batch_id": batch_id,
                }
            )
            item.metadata.update(metadata)
            saved_items.append(
                self._insert_item_tx(
                    connection,
                    item,
                    skip_event=True,
                )
            )
        now = utc_now_iso()
        placeholders = ",".join("?" for _ in target_ids)
        connection.execute(
            f"""
            UPDATE memory_versions
            SET version_state = 'superseded', updated_at = ?
            WHERE version_id IN ({placeholders})
            """,
            [now, *target_ids],
        )
        self._insert_event(
            connection,
            event_type="split",
            version_id=saved_items[0].id,
            memory_id=str(saved_items[0].metadata.get("memory_id") or saved_items[0].id),
            source_id=(
                str(saved_items[0].metadata.get("source_id"))
                if saved_items[0].metadata.get("source_id") is not None
                else None
            ),
            epoch_id=epoch_id,
            batch_id=batch_id,
            payload={
                **event_payload,
                "target_ids": list(target_ids),
                "new_version_ids": [item.id for item in saved_items],
            },
        )
        return saved_items

    def _dispute_tx(
        self,
        connection: sqlite3.Connection,
        target_ids: list[str],
        *,
        source_id: str | None,
        epoch_id: int | None,
        batch_id: int | None,
        evidence: str | None,
        payload: dict[str, Any],
    ) -> list[str]:
        updated: list[str] = []
        for target_id in target_ids:
            row = connection.execute(
                "SELECT memory_id FROM memory_versions "
                "WHERE version_id = ? AND version_state = 'current'",
                (target_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"dispute targets unknown or non-current version: {target_id}")
            if source_id:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO memory_sources(
                        version_id, source_id, source_role, evidence, created_at
                    ) VALUES (?, ?, 'conflict', ?, ?)
                    """,
                    (target_id, source_id, evidence, utc_now_iso()),
                )
            connection.execute(
                "UPDATE memory_versions SET lifecycle_status = 'disputed' "
                "WHERE version_id = ? AND version_state = 'current'",
                (target_id,),
            )
            self._insert_event(
                connection,
                event_type="dispute",
                version_id=target_id,
                memory_id=str(row["memory_id"]),
                source_id=source_id,
                epoch_id=epoch_id,
                batch_id=batch_id,
                payload=payload,
            )

            self._insert_conflict_tx(
                connection,
                target_version_id=target_id,
                candidate_summary=(
                    dict(payload["candidate_summary"])
                    if isinstance(payload.get("candidate_summary"), dict)
                    else None
                ),
                source_id=source_id,
                reason=evidence,
                epoch_id=epoch_id,
                batch_id=batch_id,
                checkpoint_key=(
                    str(payload["checkpoint_key"])
                    if payload.get("checkpoint_key") is not None
                    else None
                ),
            )
            updated.append(target_id)
        return updated

    def _insert_conflict_tx(
        self,
        connection: sqlite3.Connection,
        *,
        target_version_id: str,
        candidate_summary: dict[str, Any] | None,
        source_id: str | None,
        reason: str | None,
        epoch_id: int | None,
        batch_id: int | None,
        checkpoint_key: str | None = None,
    ) -> str:
        """Append one pending conflict while staying inside the action transaction."""

        if checkpoint_key:
            existing = connection.execute(
                """
                SELECT conflict_id FROM memory_conflicts
                WHERE target_version_id = ?
                  AND json_extract(candidate_summary_json, '$._checkpoint_key') = ?
                LIMIT 1
                """,
                (target_version_id, checkpoint_key),
            ).fetchone()
            if existing is not None:
                return str(existing["conflict_id"])
        snapshot = dict(candidate_summary or {})
        if checkpoint_key:
            snapshot["_checkpoint_key"] = checkpoint_key
        conflict_id = str(uuid4())
        now = utc_now_iso()
        connection.execute(
            """
            INSERT INTO memory_conflicts(
                conflict_id, target_version_id, candidate_summary_json,
                source_id, reason, epoch_id, batch_id, status,
                resolution_version_ids_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', '[]', ?, ?)
            """,
            (
                conflict_id,
                target_version_id,
                json.dumps(snapshot, ensure_ascii=False),
                source_id,
                reason,
                epoch_id,
                batch_id,
                now,
                now,
            ),
        )
        return conflict_id

    async def pending_conflict_groups(
        self,
        *,
        through_epoch_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return disputed current versions that have at least one new conflict.

        Deferred evidence is included when a later independent conflict causes
        the same target to be reconsidered. A deferred group with no new
        evidence is intentionally not sent to the LLM every Epoch.
        """

        self._require_writable()
        with self._connect() as connection:
            now = utc_now_iso()

            connection.execute(
                """
                UPDATE memory_conflicts
                SET status = 'resolved',
                    resolution_action = 'superseded_by_sample_action',
                    resolution_reason = 'The target was superseded before Epoch-end resolution.',
                    updated_at = ?
                WHERE status IN ('pending', 'deferred')
                  AND EXISTS (
                    SELECT 1 FROM memory_versions v
                    WHERE v.version_id = memory_conflicts.target_version_id
                      AND v.version_state = 'superseded'
                  )
                """,
                (now,),
            )
            pending_where = "status = 'pending'"
            pending_args: list[Any] = []
            if through_epoch_id is not None:
                pending_where += " AND (epoch_id IS NULL OR epoch_id <= ?)"
                pending_args.append(int(through_epoch_id))
            target_rows = connection.execute(
                f"""
                SELECT DISTINCT target_version_id
                FROM memory_conflicts
                WHERE {pending_where}
                ORDER BY target_version_id
                """,
                pending_args,
            ).fetchall()
            target_ids = [str(row["target_version_id"]) for row in target_rows]
            groups: list[dict[str, Any]] = []
            for target_id in target_ids:
                version_row = connection.execute(
                    """
                    SELECT * FROM memory_versions
                    WHERE version_id = ? AND version_state = 'current'
                      AND lifecycle_status = 'disputed'
                    """,
                    (target_id,),
                ).fetchone()
                if version_row is None:
                    continue
                conflict_rows = connection.execute(
                    """
                    SELECT * FROM memory_conflicts
                    WHERE target_version_id = ?
                      AND status IN ('pending', 'deferred')
                    ORDER BY created_at, conflict_id
                    """,
                    (target_id,),
                ).fetchall()
                source_rows = connection.execute(
                    """
                    SELECT source_role, COUNT(DISTINCT source_id) AS n
                    FROM memory_sources
                    WHERE version_id = ?
                    GROUP BY source_role
                    """,
                    (target_id,),
                ).fetchall()
                source_counts = {
                    str(row["source_role"]): int(row["n"])
                    for row in source_rows
                }
                conflicts: list[dict[str, Any]] = []
                for row in conflict_rows:
                    snapshot = json.loads(row["candidate_summary_json"] or "{}")
                    if isinstance(snapshot, dict):
                        snapshot.pop("_checkpoint_key", None)
                    else:
                        snapshot = {}
                    conflicts.append(
                        {
                            "conflict_id": str(row["conflict_id"]),
                            "candidate_summary": snapshot,
                            "source_id": row["source_id"],
                            "reason": row["reason"],
                            "epoch_id": row["epoch_id"],
                            "batch_id": row["batch_id"],
                            "status": row["status"],
                        }
                    )
                groups.append(
                    {
                        "target": self._row_to_item(version_row).to_dict(),
                        "conflicts": conflicts,
                        "support_source_count": source_counts.get("support", 0),
                        "conflict_source_count": len(
                            {
                                str(row["source_id"])
                                for row in conflict_rows
                                if row["source_id"] is not None
                            }
                        ),
                    }
                )
        return groups

    async def apply_conflict_resolution(
        self,
        *,
        target_version_id: str,
        decision: str,
        summaries: list[dict[str, Any]],
        reason: str,
        epoch_id: int,
        resolution_key: str,
        min_abandon_sources: int = 2,
    ) -> dict[str, Any]:
        """Validate and atomically apply one Epoch-end conflict decision."""

        self._require_writable()
        action = str(decision or "").strip().lower()
        if action not in self.CONFLICT_RESOLUTION_ACTIONS:
            raise ValueError(f"Unsupported conflict resolution: {action or '<empty>'}")
        if action in {"restore", "defer", "abandon"} and summaries:
            raise ValueError(f"{action} must not return replacement summaries")
        if action == "update" and len(summaries) != 1:
            raise ValueError("conflict update requires exactly one summary")
        if action == "split" and len(summaries) < 2:
            raise ValueError("conflict split requires at least two summaries")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                """
                SELECT resolution_action, resolution_reason,
                       resolution_version_ids_json
                FROM memory_conflicts
                WHERE resolution_key = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (resolution_key,),
            ).fetchone()
            if replay is not None:
                return {
                    "target_version_id": target_version_id,
                    "decision": str(replay["resolution_action"]),
                    "reason": str(replay["resolution_reason"] or ""),
                    "version_ids": list(
                        json.loads(replay["resolution_version_ids_json"] or "[]")
                    ),
                    "reused": True,
                }

            target_row = connection.execute(
                """
                SELECT * FROM memory_versions
                WHERE version_id = ? AND version_state = 'current'
                  AND lifecycle_status = 'disputed'
                """,
                (target_version_id,),
            ).fetchone()
            if target_row is None:
                raise ValueError(
                    "Conflict resolution target is unknown, non-current, or not disputed"
                )
            conflict_rows = connection.execute(
                """
                SELECT * FROM memory_conflicts
                WHERE target_version_id = ?
                  AND status IN ('pending', 'deferred')
                ORDER BY created_at, conflict_id
                """,
                (target_version_id,),
            ).fetchall()
            if not conflict_rows:
                raise ValueError("Conflict resolution target has no open conflict records")
            conflict_source_ids = list(
                dict.fromkeys(
                    str(row["source_id"])
                    for row in conflict_rows
                    if row["source_id"] is not None
                )
            )
            if action == "abandon" and len(conflict_source_ids) < max(
                1, int(min_abandon_sources)
            ):
                raise ValueError(
                    "abandon requires at least "
                    f"{max(1, int(min_abandon_sources))} independent conflict sources"
                )

            method = str(target_row["method"])
            items: list[KnowledgeItem] = []
            for index, payload in enumerate(summaries):
                if not isinstance(payload, dict):
                    raise ValueError("Conflict resolution summaries must be objects")
                item = self._item_from_action_payload(
                    payload,
                    method=method,
                    source_id=None,
                    epoch_id=epoch_id,
                    batch_id=None,
                )
                if item is None:
                    raise ValueError("Conflict resolution contains an incomplete summary")
                item.metadata.update(
                    {
                        "lifecycle_status": "active",
                        "content_source_ids": conflict_source_ids,
                        "created_by": "epoch_conflict_resolution",
                        "checkpoint_key": f"{resolution_key}:summary:{index}",
                    }
                )
                items.append(item)
            if action == "split" and any(
                not str(item.condition or "").strip() for item in items
            ):
                raise ValueError("Every conflict split summary needs explicit conditions")

            audit_payload = {
                "resolution_action": action,
                "resolution_reason": reason,
                "resolution_key": resolution_key,
                "target_version_id": target_version_id,
                "conflict_ids": [str(row["conflict_id"]) for row in conflict_rows],
                "conflict_source_count": len(conflict_source_ids),
            }
            version_ids: list[str] = []
            if action == "restore":
                connection.execute(
                    "UPDATE memory_versions SET lifecycle_status = 'active' WHERE version_id = ?",
                    (target_version_id,),
                )
                version_ids = [target_version_id]
            elif action == "update":
                saved = self._update_tx(
                    connection,
                    target_version_id,
                    items[0],
                    epoch_id=epoch_id,
                    batch_id=None,
                    event_payload=audit_payload,
                )
                version_ids = [saved.id]
            elif action == "split":
                saved_items = self._split_tx(
                    connection,
                    [target_version_id],
                    items,
                    epoch_id=epoch_id,
                    batch_id=None,
                    event_payload=audit_payload,
                )
                version_ids = [item.id for item in saved_items]
            elif action == "abandon":
                connection.execute(
                    "UPDATE memory_versions SET lifecycle_status = 'abandoned' WHERE version_id = ?",
                    (target_version_id,),
                )
                version_ids = [target_version_id]

            final_status = "deferred" if action == "defer" else "resolved"
            now = utc_now_iso()
            placeholders = ",".join("?" for _ in conflict_rows)
            connection.execute(
                f"""
                UPDATE memory_conflicts
                SET status = ?, resolution_action = ?, resolution_reason = ?,
                    resolution_version_ids_json = ?, resolution_epoch_id = ?,
                    resolution_key = ?, updated_at = ?
                WHERE conflict_id IN ({placeholders})
                """,
                [
                    final_status,
                    action,
                    reason,
                    json.dumps(version_ids, ensure_ascii=False),
                    epoch_id,
                    resolution_key,
                    now,
                    *[str(row["conflict_id"]) for row in conflict_rows],
                ],
            )
            self._insert_event(
                connection,
                event_type=(
                    "conflict_deferred" if action == "defer" else "conflict_resolved"
                ),
                version_id=(version_ids[0] if version_ids else target_version_id),
                memory_id=str(target_row["memory_id"]),
                epoch_id=epoch_id,
                payload={**audit_payload, "version_ids": version_ids},
            )
            return {
                "target_version_id": target_version_id,
                "decision": action,
                "reason": reason,
                "version_ids": version_ids,
                "conflict_ids": [str(row["conflict_id"]) for row in conflict_rows],
                "conflict_source_count": len(conflict_source_ids),
                "reused": False,
            }

    async def conflict_resolutions_for_epoch(
        self,
        epoch_id: int,
    ) -> list[dict[str, Any]]:
        """Return one compact persisted result per Epoch-end resolution call."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT target_version_id, resolution_key, resolution_action,
                       resolution_reason, resolution_version_ids_json,
                       COUNT(*) AS conflict_count,
                       COUNT(DISTINCT source_id) AS conflict_source_count
                FROM memory_conflicts
                WHERE resolution_epoch_id = ? AND resolution_key IS NOT NULL
                GROUP BY target_version_id, resolution_key, resolution_action,
                         resolution_reason, resolution_version_ids_json
                ORDER BY MIN(updated_at), target_version_id
                """,
                (int(epoch_id),),
            ).fetchall()
        return [
            {
                "target_version_id": str(row["target_version_id"]),
                "resolution_key": str(row["resolution_key"]),
                "decision": str(row["resolution_action"]),
                "reason": str(row["resolution_reason"] or ""),
                "version_ids": list(
                    json.loads(row["resolution_version_ids_json"] or "[]")
                ),
                "conflict_count": int(row["conflict_count"]),
                "conflict_source_count": int(row["conflict_source_count"]),
            }
            for row in rows
        ]

    async def epoch_statistics(self, epoch_id: int | None = None) -> dict[str, Any]:
        """Return compact counts used by Epoch artifacts and debugging."""

        with self._connect() as connection:
            where = "WHERE epoch_id = ?" if epoch_id is not None else ""
            args: list[Any] = [epoch_id] if epoch_id is not None else []
            counts = connection.execute(
                f"SELECT event_type, COUNT(*) AS n FROM memory_events {where} GROUP BY event_type",
                args,
            ).fetchall()
            status_rows = connection.execute(
                "SELECT lifecycle_status, COUNT(*) AS n FROM memory_versions WHERE version_state = 'current' GROUP BY lifecycle_status"
            ).fetchall()
            current_count = connection.execute(
                "SELECT COUNT(*) AS n FROM memory_versions WHERE version_state = 'current'"
            ).fetchone()["n"]
            text_chars = connection.execute(
                "SELECT COALESCE(SUM(LENGTH(name) + LENGTH(content) + LENGTH(condition_text)), 0) AS n FROM memory_versions WHERE version_state = 'current'"
            ).fetchone()["n"]
            supported_count = connection.execute(
                """
                SELECT COUNT(*) AS n FROM memory_versions v
                WHERE v.version_state = 'current'
                AND EXISTS (
                    SELECT 1 FROM memory_sources s
                    WHERE s.version_id = v.version_id AND s.source_role = 'support'
                )
                """
            ).fetchone()["n"]
            conflict_count = connection.execute(
                """
                SELECT COUNT(*) AS n FROM memory_versions v
                WHERE v.version_state = 'current'
                AND (
                    v.lifecycle_status = 'disputed'
                    OR EXISTS (
                        SELECT 1 FROM memory_sources s
                        WHERE s.version_id = v.version_id AND s.source_role = 'conflict'
                    )
                )
                """
            ).fetchone()["n"]
            conflict_queue_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM memory_conflicts
                GROUP BY status
                """
            ).fetchall()
            label_rows = connection.execute(
                f"SELECT label, COUNT(*) AS n FROM memory_events {where} "
                "AND event_type = 'use' GROUP BY label"
                if where
                else "SELECT label, COUNT(*) AS n FROM memory_events WHERE event_type = 'use' GROUP BY label",
                args,
            ).fetchall()
        event_counts = {str(row["event_type"]): int(row["n"]) for row in counts}
        edit_count = sum(
            event_counts.get(name, 0) for name in ("add", "update", "merge", "split", "epoch_review_update")
        )
        current_count = int(current_count)
        return {
            "epoch_id": epoch_id,
            "event_counts": event_counts,
            "lifecycle_counts": {str(row["lifecycle_status"]): int(row["n"]) for row in status_rows},
            "use_label_counts": {
                str(row["label"]): int(row["n"]) for row in label_rows if row["label"]
            },
            "current_memory_versions": current_count,
            "memory_text_chars": int(text_chars),
            "estimated_memory_tokens": int(text_chars) // 4,
            "cross_source_support_ratio": (
                int(supported_count) / current_count if current_count else 0.0
            ),
            "conflict_ratio": int(conflict_count) / current_count if current_count else 0.0,
            "conflict_queue_counts": {
                str(row["status"]): int(row["n"])
                for row in conflict_queue_rows
            },
            "repository_edit_rate": edit_count / current_count if current_count else 0.0,
        }

    async def count_answer_exclusions(self, source_id: str) -> int:
        """Count visible current Memories hidden by same-source protection."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(DISTINCT v.version_id) AS n
                FROM memory_versions v
                JOIN memory_sources s ON s.version_id = v.version_id
                WHERE v.version_state = 'current'
                  AND v.lifecycle_status IN ('provisional', 'active', 'disputed')
                  AND s.source_role = 'content'
                  AND s.source_id = ?
                """,
                (source_id,),
            ).fetchone()
        return int(row["n"] if row is not None else 0)

    def clear(self) -> None:
        self._require_writable()
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='epoch_review_jobs'").fetchone():
                connection.execute("DELETE FROM epoch_review_jobs")
            connection.execute("DELETE FROM memory_conflicts")
            connection.execute("DELETE FROM memory_events")
            connection.execute("DELETE FROM memory_sources")
            connection.execute("DELETE FROM memory_fts")
            connection.execute("DELETE FROM memory_versions")

    def delete_ids(self, item_ids: list[str]) -> int:
        """Remove uncommitted replay writes while retaining committed history."""

        self._require_writable()
        if not item_ids:
            return 0
        placeholders = ",".join("?" for _ in item_ids)
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM memory_fts WHERE version_id IN ({placeholders})",
                item_ids,
            )
            cursor = connection.execute(
                f"DELETE FROM memory_versions WHERE version_id IN ({placeholders})",
                item_ids,
            )
        return max(int(cursor.rowcount), 0)

    def backup_to(self, destination: str | Path) -> Path:
        """Create a consistent SQLite snapshot for a validation checkpoint."""
        return snapshot_sqlite(self.path, destination)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_versions(
                    version_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    parent_version_id TEXT,
                    method TEXT NOT NULL,
                    name TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    condition_text TEXT,
                    category TEXT,
                    tools_json TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL DEFAULT 'provisional',
                    version_state TEXT NOT NULL DEFAULT 'current',
                    epoch_id INTEGER,
                    batch_id INTEGER,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memory_versions_current
                    ON memory_versions(version_state, lifecycle_status);
                CREATE INDEX IF NOT EXISTS idx_memory_versions_memory
                    ON memory_versions(memory_id, version_state);
                CREATE TABLE IF NOT EXISTS memory_sources(
                    version_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_role TEXT NOT NULL CHECK(source_role IN ('content', 'support', 'conflict')),
                    evidence TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(version_id, source_id, source_role),
                    FOREIGN KEY(version_id) REFERENCES memory_versions(version_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_sources_lookup
                    ON memory_sources(source_id, source_role, version_id);
                CREATE TABLE IF NOT EXISTS memory_events(
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    version_id TEXT,
                    memory_id TEXT,
                    source_id TEXT,
                    epoch_id INTEGER,
                    batch_id INTEGER,
                    label TEXT,
                    score_before REAL,
                    score_after REAL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(version_id) REFERENCES memory_versions(version_id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_epoch
                    ON memory_events(epoch_id, event_type);
                CREATE TABLE IF NOT EXISTS memory_conflicts(
                    conflict_id TEXT PRIMARY KEY,
                    target_version_id TEXT NOT NULL,
                    candidate_summary_json TEXT NOT NULL,
                    source_id TEXT,
                    reason TEXT,
                    epoch_id INTEGER,
                    batch_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'deferred', 'resolved')),
                    resolution_action TEXT,
                    resolution_reason TEXT,
                    resolution_version_ids_json TEXT NOT NULL DEFAULT '[]',
                    resolution_epoch_id INTEGER,
                    resolution_key TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(target_version_id)
                        REFERENCES memory_versions(version_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_conflicts_pending
                    ON memory_conflicts(status, target_version_id, epoch_id);
                CREATE INDEX IF NOT EXISTS idx_memory_conflicts_resolution
                    ON memory_conflicts(resolution_epoch_id, resolution_action);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    version_id UNINDEXED,
                    name,
                    content,
                    condition_text,
                    tokenize = 'unicode61'
                );
                """
            )

    def _is_initialized_database(self) -> bool:
        try:
            with sqlite3.connect(self.path, timeout=30) as connection:
                rows = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('memory_versions', 'memory_sources', 'memory_events', 'memory_fts')"
                ).fetchall()
        except sqlite3.DatabaseError:
            return False
        return {str(row[0]) for row in rows} == {
            "memory_versions",
            "memory_sources",
            "memory_events",
            "memory_fts",
        }

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro", uri=True, timeout=30
            )
        else:
            connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError(f"SQLite memory database is read-only: {self.path}")

    def _row_to_item(self, row: sqlite3.Row) -> KnowledgeItem:
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata.update(
            {
                "memory_id": row["memory_id"],
                "version_id": row["version_id"],
                "parent_version_id": row["parent_version_id"],
                "version_state": row["version_state"],
                "lifecycle_status": row["lifecycle_status"],
            }
        )
        with self._connect() as connection:
            source_rows = connection.execute(
                "SELECT source_id, source_role FROM memory_sources WHERE version_id = ? ORDER BY source_id",
                (row["version_id"],),
            ).fetchall()
        source_map: dict[str, list[str]] = {role: [] for role in self.SOURCE_ROLES}
        for source_row in source_rows:
            role = str(source_row["source_role"])
            source_id = str(source_row["source_id"])
            if source_id not in source_map[role]:
                source_map[role].append(source_id)
        for role, source_ids in source_map.items():
            metadata[f"{role}_source_ids"] = source_ids
        return KnowledgeItem.from_dict(
            {
                "id": row["version_id"],
                "method": row["method"],
                "name": row["name"],
                "key": row["key"],
                "content": row["content"],
                "condition": row["condition_text"] or None,
                "category": row["category"] or None,
                "tools": json.loads(row["tools_json"] or "[]"),
                "abandoned": row["lifecycle_status"] == "abandoned",
                "metadata": metadata,
            }
        )

    def _filter_sql(
        self,
        filters: dict[str, Any],
        *,
        table_alias: str,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        default_current = filters.get("include_superseded") is not True
        if default_current:
            clauses.append(f"{table_alias}.version_state = 'current'")
        if "version_state" in filters:
            clauses.append(f"{table_alias}.version_state = ?")
            parameters.append(str(filters["version_state"]))
        if "lifecycle_status" in filters:
            clauses.append(f"{table_alias}.lifecycle_status = ?")
            parameters.append(str(filters["lifecycle_status"]))
        elif "lifecycle_statuses" in filters:
            raw_statuses = filters["lifecycle_statuses"]
            statuses = (
                [str(raw_statuses)]
                if isinstance(raw_statuses, str)
                else [str(value) for value in raw_statuses or []]
            )
            statuses = list(dict.fromkeys(statuses))
            invalid = set(statuses) - self.LIFECYCLE_STATUSES
            if invalid:
                raise ValueError(
                    f"Unknown lifecycle statuses: {sorted(invalid)}"
                )
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                clauses.append(
                    f"{table_alias}.lifecycle_status IN ({placeholders})"
                )
                parameters.extend(statuses)
            else:

                clauses.append("1 = 0")
        elif filters.get("abandoned") is not True:
            placeholders = ",".join("?" for _ in self.ACTIVE_LIFECYCLE_STATUSES)
            clauses.append(f"{table_alias}.lifecycle_status IN ({placeholders})")
            parameters.extend(sorted(self.ACTIVE_LIFECYCLE_STATUSES))
        elif filters.get("abandoned") is True:
            clauses.append(f"{table_alias}.lifecycle_status = 'abandoned'")
        for key in ("method", "category"):
            if key in filters:
                clauses.append(f"{table_alias}.{key} = ?")
                parameters.append(str(filters[key]))
        if filters.get("exclude_content_source_id"):
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM memory_sources s "
                f"WHERE s.version_id = {table_alias}.version_id "
                "AND s.source_role = 'content' AND s.source_id = ?)"
            )
            parameters.append(str(filters["exclude_content_source_id"]))
        if filters.get("content_source_id"):
            clauses.append(
                "EXISTS (SELECT 1 FROM memory_sources s "
                f"WHERE s.version_id = {table_alias}.version_id "
                "AND s.source_role = 'content' AND s.source_id = ?)"
            )
            parameters.append(str(filters["content_source_id"]))
        for key, value in filters.items():
            if key.startswith("metadata."):
                metadata_key = key.split(".", 1)[1]
                if not re.fullmatch(r"[A-Za-z0-9_]+", metadata_key):
                    raise ValueError(f"Unsupported metadata filter: {key}")
                clauses.append(f"json_extract({table_alias}.metadata_json, ?) = ?")
                parameters.extend([f"$.{metadata_key}", value])
        return clauses, parameters

    def _insert_sources(
        self,
        connection: sqlite3.Connection,
        version_id: str,
        metadata: dict[str, Any],
    ) -> None:
        for role in ("content", "support", "conflict"):
            for source_id in source_ids_from_metadata(metadata, role):
                connection.execute(
                    "INSERT OR IGNORE INTO memory_sources(version_id, source_id, source_role, evidence, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        version_id,
                        source_id,
                        role,
                        self._source_evidence(metadata, role, source_id),
                        utc_now_iso(),
                    ),
                )

    async def _source_map_for_versions(
        self, version_ids: list[str]
    ) -> dict[str, list[str]]:
        result = {role: [] for role in ("content", "support", "conflict")}
        if not version_ids:
            return result
        placeholders = ",".join("?" for _ in version_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT source_id, source_role FROM memory_sources WHERE version_id IN ({placeholders}) ORDER BY source_id",
                version_ids,
            ).fetchall()
        for row in rows:
            role = str(row["source_role"])
            source_id = str(row["source_id"])
            if source_id not in result[role]:
                result[role].append(source_id)
        return result

    def _copy_sources(
        self,
        connection: sqlite3.Connection,
        old_version_id: str,
        new_version_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO memory_sources(version_id, source_id, source_role, evidence, created_at)
            SELECT ?, source_id, source_role, evidence, ?
            FROM memory_sources WHERE version_id = ?
            """,
            (new_version_id, utc_now_iso(), old_version_id),
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        version_id: str | None = None,
        memory_id: str | None = None,
        source_id: str | None = None,
        epoch_id: int | None = None,
        batch_id: int | None = None,
        label: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(payload or {})
        connection.execute(
            """
            INSERT INTO memory_events(
                event_id, event_type, version_id, memory_id, source_id,
                epoch_id, batch_id, label, score_before, score_after,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                event_type,
                version_id,
                memory_id,
                source_id,
                epoch_id,
                batch_id,
                label,
                payload.get("score_before"),
                payload.get("score_after"),
                json.dumps(payload, ensure_ascii=False),
                utc_now_iso(),
            ),
        )

    @staticmethod
    def _source_evidence(metadata: dict[str, Any], role: str, source_id: str) -> str | None:
        evidence = metadata.get(f"{role}_evidence")
        if isinstance(evidence, dict):
            value = evidence.get(source_id)
            return str(value) if value is not None else None
        return str(evidence) if evidence is not None else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _match_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", query.lower())
        return " OR ".join(f'"{token}"' for token in tokens)
