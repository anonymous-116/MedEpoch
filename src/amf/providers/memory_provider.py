from __future__ import annotations

import sqlite3

from amf.memory.item import KnowledgeItem
from amf.memory.retrieval import BaseRetriever
from amf.memory.store import BaseKnowledgeStore
from amf.providers.base import BaseProvider, ProviderRequest, ProviderResponse

class MemoryProvider(BaseProvider):
    name = "memory"

    def __init__(self, store: BaseKnowledgeStore, retriever: BaseRetriever) -> None:
        self.store = store
        self.retriever = retriever

    async def call(self, request: ProviderRequest) -> ProviderResponse:
        if request.operation == "retrieve":
            top_k = int(request.payload.get("top_k", 5))
            filters = dict(request.payload.get("filters") or {})
            filters.setdefault("abandoned", False)
            items = await self.retriever.retrieve(
                query=request.query or "",
                store=self.store,
                top_k=top_k,
                filters=filters,
            )
            metadata = {}
            excluded_source = filters.get("exclude_content_source_id")
            if excluded_source and hasattr(self.store, "count_answer_exclusions"):
                metadata["same_source_excluded_count"] = (
                    await self.store.count_answer_exclusions(str(excluded_source))
                )
            return ProviderResponse(
                ok=True,
                data=[item.to_dict() for item in items],
                metadata=metadata,
            )

        if request.operation == "apply_actions":

            if not hasattr(self.store, "apply_actions"):
                return ProviderResponse(
                    ok=False,
                    error="This memory store does not support transactional actions.",
                )
            try:
                result = await self.store.apply_actions(
                    list(request.payload.get("actions") or []),
                    method=str(request.payload.get("method") or "epoch"),
                    source_id=(
                        str(request.payload["source_id"])
                        if request.payload.get("source_id") is not None
                        else None
                    ),
                    epoch_id=request.payload.get("epoch_id"),
                    batch_id=request.payload.get("batch_id"),
                    candidate_summary=(
                        dict(request.payload["candidate_summary"])
                        if isinstance(request.payload.get("candidate_summary"), dict)
                        else None
                    ),
                    information_usage=(
                        dict(request.payload["information_usage"])
                        if isinstance(request.payload.get("information_usage"), dict)
                        else None
                    ),
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                PermissionError,
                sqlite3.DatabaseError,
            ) as exc:
                return ProviderResponse(
                    ok=False,
                    error=f"Transactional memory action rejected: {exc}",
                )
            return ProviderResponse(ok=True, data=result)

        if request.operation == "write":
            payload = dict(request.payload.get("item") or request.payload)
            item = KnowledgeItem.from_dict(payload) if "id" in payload else KnowledgeItem.create(**payload)
            checkpoint_key = item.metadata.get("checkpoint_key")
            if checkpoint_key:

                existing = await self.store.list(
                    filters={"metadata.checkpoint_key": checkpoint_key}
                )
                if existing:
                    return ProviderResponse(
                        ok=True,
                        data=existing[0].to_dict(),
                        metadata={"idempotent_reuse": True},
                    )
            saved = await self.store.add(item)
            return ProviderResponse(ok=True, data=saved.to_dict())

        if request.operation == "list":
            filters = dict(request.payload.get("filters") or {})
            items = await self.store.list(filters=filters)
            return ProviderResponse(ok=True, data=[item.to_dict() for item in items])

        if request.operation == "update":
            item_id = str(request.payload["id"])
            patch = dict(request.payload.get("patch") or {})
            checkpoint_key = dict(patch.get("metadata") or {}).get("checkpoint_key")
            if checkpoint_key:
                existing = await self.store.list(
                    filters={"metadata.checkpoint_key": checkpoint_key}
                )
                if existing:
                    return ProviderResponse(
                        ok=True,
                        data=existing[0].to_dict(),
                        metadata={"idempotent_reuse": True},
                    )
            item = await self.store.update(item_id, patch)
            return ProviderResponse(ok=item is not None, data=item.to_dict() if item else None)

        if request.operation == "confirm" and hasattr(self.store, "confirm"):
            item = await self.store.confirm(
                str(request.payload["id"]),
                str(request.payload["source_id"]),
                evidence=request.payload.get("evidence"),
                epoch_id=request.payload.get("epoch_id"),
                batch_id=request.payload.get("batch_id"),
            )
            return ProviderResponse(ok=item is not None, data=item.to_dict() if item else None)

        if request.operation == "dispute" and hasattr(self.store, "dispute"):
            items = await self.store.dispute(
                [str(value) for value in request.payload.get("ids") or []],
                str(request.payload["source_id"]),
                evidence=request.payload.get("evidence"),
                epoch_id=request.payload.get("epoch_id"),
                batch_id=request.payload.get("batch_id"),
            )
            return ProviderResponse(ok=bool(items), data=[item.to_dict() for item in items])

        if request.operation == "merge" and hasattr(self.store, "merge"):
            item_payload = dict(request.payload["item"])
            checkpoint_key = dict(item_payload.get("metadata") or {}).get("checkpoint_key")
            if checkpoint_key:
                existing = await self.store.list(
                    filters={"metadata.checkpoint_key": checkpoint_key}
                )
                if existing:
                    return ProviderResponse(
                        ok=True,
                        data=existing[0].to_dict(),
                        metadata={"idempotent_reuse": True},
                    )
            item = KnowledgeItem.from_dict(item_payload)
            saved = await self.store.merge(
                [str(value) for value in request.payload.get("ids") or []], item
            )
            return ProviderResponse(ok=True, data=saved.to_dict())

        if request.operation == "split" and hasattr(self.store, "split"):
            item_payloads = [dict(value) for value in request.payload.get("items") or []]
            checkpoint_keys = [
                dict(value.get("metadata") or {}).get("checkpoint_key")
                for value in item_payloads
            ]
            if checkpoint_keys and all(checkpoint_keys):
                existing_items = []
                for checkpoint_key in checkpoint_keys:
                    matches = await self.store.list(
                        filters={"metadata.checkpoint_key": checkpoint_key}
                    )
                    if matches:
                        existing_items.append(matches[0])
                if len(existing_items) == len(item_payloads):
                    return ProviderResponse(
                        ok=True,
                        data=[item.to_dict() for item in existing_items],
                        metadata={"idempotent_reuse": True},
                    )
            items = [
                KnowledgeItem.from_dict(dict(value))
                for value in item_payloads
            ]
            target_ids = [str(value) for value in request.payload.get("ids") or []]
            if not target_ids and request.payload.get("id") is not None:
                target_ids = [str(request.payload["id"])]
            saved = await self.store.split(target_ids, items)
            return ProviderResponse(ok=True, data=[item.to_dict() for item in saved])

        return ProviderResponse(ok=False, error=f"Unsupported memory operation: {request.operation}")
