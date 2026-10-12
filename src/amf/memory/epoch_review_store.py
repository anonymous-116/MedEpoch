"""Small persistence adapter: review audit and version acceptance share a transaction.

The ordinary Epoch version/FTS/source schema is unchanged. One extra table
stores resumable jobs and per-action results; proposals use immutable versions.
"""
from __future__ import annotations

import json
import difflib

from amf.events import utc_now_iso
from amf.memory.item import KnowledgeItem
from amf.memory.epoch_review_input import text_fields

class ReviewStore:
    def __init__(self, store):
        store._require_writable()
        self.store = store
        with store._connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS epoch_review_jobs (
                job_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL)""")

    def get(self, job_id):
        with self.store._connect() as con:
            row = con.execute("SELECT payload_json FROM epoch_review_jobs WHERE job_id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, job):
        with self.store._connect() as con:
            self._save_tx(con, job)

    @staticmethod
    def _save_tx(con, job):
        con.execute("INSERT INTO epoch_review_jobs VALUES (?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at",
                    (job["job_id"], json.dumps(job, ensure_ascii=False, allow_nan=False), utc_now_iso()))

    def _row(self, con, version):
        row = con.execute("SELECT * FROM memory_versions WHERE version_id=?", (version,)).fetchone()
        if row is None:
            raise ValueError("Unresolved target: missing immutable version")
        return row

    def _head(self, con, logical_id):
        rows = con.execute("SELECT * FROM memory_versions WHERE memory_id=? AND version_state='current'", (logical_id,)).fetchall()
        if len(rows) != 1 or rows[0]["lifecycle_status"] in {"abandoned", "disputed"}:

            raise ValueError("Unresolved target: missing, ambiguous, disputed or abandoned head")
        return rows[0]

    @staticmethod
    def row_text(row):
        return text_fields(row["content"], row["condition_text"])

    def head(self, logical_id):
        with self.store._connect() as con:
            row = self._head(con, logical_id)
            return {"version_id": row["version_id"], **self.row_text(row)}

    def validate_snapshot(self, item):
        with self.store._connect() as con:
            row = self._row(con, item["version_id"])
            if row["memory_id"] != item["logical_id"] or self.row_text(row) != text_fields(item["content"], item["applicability"]):
                raise ValueError("Historical snapshot conflicts with immutable stored version")

    def _new_item(self, con, base, text, parents, source, epoch_id, job_id, state):
        sources = self.store._source_lists_tx(con, parents)

        return KnowledgeItem.create(method=base["method"], name=text["content"][:120],
            key=text["content"][:240], content=text["content"], condition=text["applicability"] or None,
            metadata={"memory_id": base["memory_id"], "parent_version_id": parents[0],
                      "parent_version_ids": parents, "version_state": state,
                      "lifecycle_status": "provisional", "source_id": source,
                      "content_source_ids": list(dict.fromkeys(sources["content"] + [source])),
                      "support_source_ids": [], "conflict_source_ids": sources["conflict"],
                      "epoch_id": epoch_id, "review_job_id": job_id})

    def stage(self, job, alias, target, action):
        """Persist a candidate before reconciliation, but never expose it to retrieval."""
        with self.store._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            saved = con.execute("SELECT payload_json FROM epoch_review_jobs WHERE job_id=?", (job["job_id"],)).fetchone()
            if saved:
                state = json.loads(saved[0]).get("execution", {}).get(alias)
                if state and state.get("proposal_id"):
                    job["execution"][alias] = state
                    return state["proposal_id"]
            base = self._row(con, target["version_id"])
            item = self._new_item(con, base, action["replacement"], [target["version_id"]],
                                  job["source_id"], job["epoch_id"], job["job_id"], "superseded")
            self.store._insert_item_tx(con, item, event_type="review_proposal")
            job["execution"][alias] = {"status": "pending", "proposal_id": item.id}
            self._save_tx(con, job)
        return item.id

    def accept(self, job, alias, target, proposal_id, expected, replacement, merge_note=None):
        """Compare-and-swap plus audit/idempotency in one short transaction."""
        with self.store._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            saved = con.execute("SELECT payload_json FROM epoch_review_jobs WHERE job_id=?", (job["job_id"],)).fetchone()
            previous = json.loads(saved[0]).get("execution", {}).get(alias) if saved else None
            if previous and previous.get("status") in {"accepted", "noop", "keep"}:
                job["execution"][alias] = previous
                return True
            current = self._head(con, target["logical_id"])
            if current["version_id"] != expected:
                return False
            parents = [expected, proposal_id]
            if self.row_text(current) == text_fields(replacement["content"], replacement["applicability"]):
                outcome = {"status": "noop", "version_id": expected, "parents": parents,
                           "proposal_id": proposal_id, "merge_note": merge_note}
            else:
                item = self._new_item(con, current, replacement, parents, job["source_id"],
                                      job["epoch_id"], job["job_id"], "current")
                con.execute("UPDATE memory_versions SET version_state='superseded', updated_at=? WHERE version_id=? AND version_state='current'",
                            (utc_now_iso(), expected))
                self.store._insert_item_tx(con, item, event_type="epoch_review_update",
                                           event_payload={"job_id": job["job_id"], "alias": alias,
                                                          "parents": parents, "merge_note": merge_note,
                                                          "item": item.to_dict()})
                outcome = {"status": "accepted", "version_id": item.id, "parents": parents,
                           "proposal_id": proposal_id, "merge_note": merge_note,
                           "diff": "\n".join(difflib.unified_diff(
                               (current["content"] + "\n" + (current["condition_text"] or "")).splitlines(),
                               (item.content + "\n" + (item.condition or "")).splitlines(),
                               fromfile=expected, tofile=item.id, lineterm=""))}
            job["execution"][alias] = outcome
            self._save_tx(con, job)
            return True
