from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from amf.memory.item import KnowledgeItem
from amf.memory.store import BaseKnowledgeStore

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        store: BaseKnowledgeStore,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[KnowledgeItem]:
        ...

class EpochSQLiteFTSRetriever(BaseRetriever):
    """BM25 adapter for the versioned Epoch SQLite repository."""

    async def retrieve(
        self,
        query: str,
        store: BaseKnowledgeStore,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[KnowledgeItem]:
        from amf.memory.epoch_sqlite_store import EpochSQLiteMemoryStore

        if not isinstance(store, EpochSQLiteMemoryStore):
            raise TypeError("EpochSQLiteFTSRetriever requires EpochSQLiteMemoryStore.")
        return await store.search(query, top_k=top_k, filters=filters)

class SimilarityServicePostFilterRetriever(BaseRetriever):
    """Filter a small BM25 candidate set through the local similarity API.

    This is intentionally not a vector index. SQLite FTS5 remains the first-stage
    retriever shared by the experiment; the service only rejects weak candidates
    and reranks survivors. Model/GPU lifecycle is isolated from experiment workers.
    """

    _THRESHOLD_FILTER = "_bge_similarity_threshold"
    _CANDIDATE_K_FILTER = "_bm25_candidate_k"
    _FINAL_K_FILTER = "_bge_final_top_k"

    def __init__(
        self,
        base_retriever: BaseRetriever,
        *,
        service_url: str,
        similarity_threshold: float,
        bm25_candidate_k: int = 10,
        final_top_k: int = 3,
        timeout_seconds: float = 120,
    ) -> None:
        if bm25_candidate_k <= 0 or final_top_k <= 0:
            raise ValueError("BM25 candidate_k and BGE final_top_k must be positive.")
        self.base_retriever = base_retriever
        self.service_url = service_url.rstrip("/")
        if not self.service_url:
            raise ValueError("similarity service URL must not be empty")
        self.similarity_threshold = float(similarity_threshold)
        self.bm25_candidate_k = int(bm25_candidate_k)
        self.final_top_k = int(final_top_k)
        self.timeout_seconds = float(timeout_seconds)

    async def retrieve(
        self,
        query: str,
        store: BaseKnowledgeStore,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[KnowledgeItem]:
        retrieval_filters = dict(filters or {})
        threshold = float(
            retrieval_filters.pop(
                self._THRESHOLD_FILTER,
                self.similarity_threshold,
            )
        )
        candidate_k = int(
            retrieval_filters.pop(
                self._CANDIDATE_K_FILTER,
                self.bm25_candidate_k,
            )
        )
        requested_final_k = int(
            retrieval_filters.pop(
                self._FINAL_K_FILTER,
                self.final_top_k,
            )
        )
        final_k = min(max(top_k, 0), max(requested_final_k, 0))
        if final_k <= 0 or not query.strip():
            return []

        candidates = await self.base_retriever.retrieve(
            query=query,
            store=store,
            top_k=max(candidate_k, final_k),
            filters=retrieval_filters,
        )
        if not candidates:
            return []

        response = await asyncio.to_thread(
            self._call_service,
            {
                "query": query,
                "candidates": [
                    {"id": item.id, "text": self.memory_text(item)}
                    for item in candidates
                ],
                "threshold": threshold,
                "top_k": final_k,
            },
        )
        candidate_map = {item.id: item for item in candidates}
        bm25_ranks = {item.id: rank for rank, item in enumerate(candidates, start=1)}
        service_results = response.get("results")
        if not isinstance(service_results, list):
            raise RuntimeError("Similarity service response has no results list.")

        results: list[KnowledgeItem] = []
        for result in service_results:
            if not isinstance(result, dict):
                continue
            item_id = str(result.get("id") or "")
            item = candidate_map.get(item_id)
            if item is None:
                raise RuntimeError(
                    f"Similarity service returned an unknown candidate ID: {item_id}"
                )
            similarity = float(result["similarity"])
            bge_rank = int(result.get("rank") or len(results) + 1)
            payload = item.to_dict()
            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "bm25_rank": bm25_ranks[item.id],
                    "bm25_score": metadata.get("retrieval_score"),
                    "bge_similarity": similarity,
                    "bge_rank": bge_rank,
                    "bge_similarity_threshold": threshold,
                    "similarity_service_url": self.service_url,
                    "similarity_model_path": response.get("model_path"),
                    "similarity_device": response.get("device"),
                }
            )

            metadata["retrieval_score"] = similarity
            payload["metadata"] = metadata
            results.append(KnowledgeItem.from_dict(payload))
        return results

    def _call_service(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.service_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Similarity service returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Cannot reach similarity service at {self.service_url}: {exc}"
            ) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Similarity service returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Similarity service returned a non-object response.")
        return parsed

    def check_health(self) -> dict[str, Any]:
        """Fail before an experiment starts if the local service is absent."""

        base_url = self.service_url.rsplit("/", 1)[0]
        request = Request(f"{base_url}/health", method="GET")
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(
                request,
                timeout=min(self.timeout_seconds, 10.0),
            ) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Similarity service is unavailable. Start the configured local "
                f"service before running Epoch. Endpoint: {base_url}/health. "
                f"Error: {exc}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Similarity service health response is invalid JSON.") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise RuntimeError(f"Similarity service is unhealthy: {payload!r}")
        return payload

    @staticmethod
    def memory_text(item: KnowledgeItem) -> str:
        """Use the retrieval representation fixed by the experiment protocol."""

        return "\n".join(
            value
            for value in (
                item.name.strip(),
                item.content.strip(),
                (item.condition or "").strip(),
            )
            if value
        )
