from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from amf.events import utc_now_iso
from amf.llm import BaseLLMClient

DEFAULT_EPOCH_SKILL = (
    "Answer the current medical question. Use available tools when they are "
    "relevant, check important conditions and exceptions, and make the final "
    "answer complete."
)

class EpochSkillStore:
    """Small file-backed version chain for the global Epoch skill prompt."""

    def __init__(self, root: str | Path, *, max_tokens: int = 3000) -> None:
        self.root = Path(root)
        self.max_tokens = max_tokens
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "skill_versions.jsonl"

    def load_current(self) -> tuple[str, dict[str, Any] | None]:
        records = self._read_manifest()
        if not records:
            return DEFAULT_EPOCH_SKILL, None
        record = records[-1]
        path = self.root / str(record["path"])
        if not path.exists():
            return DEFAULT_EPOCH_SKILL, None
        return path.read_text(encoding="utf-8"), record

    def save(
        self,
        content: str,
        *,
        epoch_id: int,
        parent_version: str | None,
        selected_count: int,
    ) -> dict[str, Any]:
        content = self.trim(content)
        records = self._read_manifest()
        version_number = len(records)
        version = f"v{version_number:03d}"
        filename = f"skill_{version}.txt"
        (self.root / filename).write_text(content, encoding="utf-8")
        record = {
            "version": version,
            "path": filename,
            "epoch_id": epoch_id,
            "parent_version": parent_version,
            "selected_count": selected_count,
            "token_count": self.count_tokens(content),
            "created_at": utc_now_iso(),
        }
        with self.manifest_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def trim(self, content: str) -> str:
        """Keep the global prompt below the configured model-context budget."""

        text = self._clean(content)
        if self.count_tokens(text) <= self.max_tokens:
            return text

        max_chars = max(1, self.max_tokens * 4)
        return text[:max_chars].rstrip()

    @staticmethod
    def count_tokens(text: str) -> int:
        try:
            import tiktoken

            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            return max(1, len(text) // 4) if text else 0

    def _read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    @staticmethod
    def _clean(content: str) -> str:
        lines = [line.rstrip() for line in str(content).splitlines()]
        return "\n".join(lines).strip()

class EpochSkillUpdater:
    """Summarize cross-epoch score changes into one global skill prompt."""

    def __init__(
        self,
        llm: BaseLLMClient,
        store: EpochSkillStore,
        *,
        max_batch_size: int = 5,
        max_retries: int = 2,
    ) -> None:
        self.llm = llm
        self.store = store
        self.max_batch_size = max(1, max_batch_size)
        self.max_retries = max(1, max_retries)

    async def update(
        self,
        records: Iterable[dict[str, Any]],
        *,
        epoch_id: int,
    ) -> dict[str, Any]:
        selected = [dict(record) for record in records]
        previous_skill, previous_record = self.store.load_current()
        if not selected:
            return {
                "skill": previous_skill,
                "version": previous_record.get("version") if previous_record else None,
                "selected_count": 0,
                "batch_count": 0,
                "suggestions": [],
            }

        suggestions: list[str] = []
        for start in range(0, len(selected), self.max_batch_size):
            batch = selected[start : start + self.max_batch_size]
            suggestions.append(await self._analyze_batch(batch, epoch_id=epoch_id))

        new_skill = await self._rewrite(previous_skill, suggestions, epoch_id=epoch_id)
        record = self.store.save(
            new_skill,
            epoch_id=epoch_id,
            parent_version=previous_record.get("version") if previous_record else None,
            selected_count=len(selected),
        )
        return {
            "skill": new_skill,
            "version": record["version"],
            "selected_count": len(selected),
            "batch_count": len(suggestions),
            "suggestions": suggestions,
            "record": record,
        }

    async def _analyze_batch(self, records: list[dict[str, Any]], *, epoch_id: int) -> str:
        evidence = []
        for record in records:
            evidence.append(
                {
                    "score_change": record.get("score_change"),
                    "previous_score": record.get("previous_score"),
                    "current_score": record.get("current_score"),
                    "missed_positive_criteria": record.get("missed_positive_criteria", []),
                    "triggered_negative_criteria": record.get("triggered_negative_criteria", []),
                    "tool_names": [
                        item.get("tool")
                        for item in record.get("tool_queries", [])
                        if isinstance(item, dict)
                    ],
                }
            )
        prompt = (
            "You are analyzing a small batch of score changes from two training "
            "epochs. Extract only general agent-solving behaviors that could help "
            "future cases. Do not mention case IDs, reproduce patient facts, give "
            "medical dosages, or copy an answer. Return concise skill suggestions "
            "as plain text.\n\n"
            f"Epoch: {epoch_id}\nEvidence:\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}"
        )
        raw = ""
        for _ in range(self.max_retries):
            raw = await self.llm.generate(prompt)
            if str(raw).strip():
                return str(raw).strip()
        return ""

    async def _rewrite(self, old_skill: str, suggestions: list[str], *, epoch_id: int) -> str:
        prompt = (
            "You maintain a global skill prompt for a medical question-solving "
            "agent. Rewrite the old skill using the suggestions. Keep only general "
            "reasoning, verification, communication, and tool-use behaviors. "
            "Never include a case-specific question, patient fact, diagnosis, "
            "dose, named benchmark example, or copied answer. Return only the new "
            "skill text, without a preamble or Markdown code fence. Keep it under "
            f"{self.store.max_tokens} tokens.\n\n"
            f"Epoch: {epoch_id}\nOld skill:\n{old_skill}\n\n"
            f"Suggestions:\n{json.dumps(suggestions, ensure_ascii=False, indent=2)}"
        )
        raw = ""
        for _ in range(self.max_retries):
            raw = await self.llm.generate(prompt)
            cleaned = self._clean_model_text(raw)
            if cleaned:
                return self.store.trim(cleaned)
        return self.store.trim(old_skill)

    @staticmethod
    def _clean_model_text(raw: Any) -> str:
        text = str(raw or "").strip()
        text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("skill"), str):
                text = payload["skill"].strip()
        return text
