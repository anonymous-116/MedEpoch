from __future__ import annotations

"""Deterministic semantic windows for Epoch training.

Clustering is performed once from question text only.  The resulting manifest
is an experiment artifact; every Epoch reuses the same clusters but shuffles
members and window order with ``seed + epoch_id``.
"""

import asyncio
import json
import random
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Sequence

from amf.datasets import DatasetExample
from amf.memory.provenance import source_id_for_case

def question_text(example: DatasetExample) -> str:
    messages = example.case.get("messages")
    if isinstance(messages, list) and messages:
        parts = [
            f"{str(message.get('role') or 'user').capitalize()}: "
            f"{str(message.get('content') or '')}"
            for message in messages
            if isinstance(message, dict)
        ]
        if parts:
            return "\n\n".join(parts)
    return str(example.case.get("question") or "").strip()

def example_key(example: DatasetExample) -> str:

    return str(example.row_index)

async def cluster_examples(
    examples: Sequence[DatasetExample],
    *,
    service_url: str,
    similarity_threshold: float | None = None,
    target_multi_question_ratio_min: float | None = None,
    target_multi_question_ratio_max: float | None = None,
    linkage: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Ask the local BGE service for one fixed semantic partition."""

    if not examples:
        raise ValueError("cannot cluster an empty training set")
    payload: dict[str, Any] = {
        "items": [
            {"id": example_key(example), "text": question_text(example)}
            for example in examples
        ],
        "linkage": str(linkage),
    }
    if similarity_threshold is not None:
        payload["similarity_threshold"] = float(similarity_threshold)
    else:
        payload["target_multi_question_ratio_min"] = float(
            target_multi_question_ratio_min
        )
        payload["target_multi_question_ratio_max"] = float(
            target_multi_question_ratio_max
        )

    def request_cluster() -> dict[str, Any]:
        request = urllib.request.Request(
            service_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"evidence clustering service returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"evidence clustering service failed: {exc}") from exc
        if not isinstance(body, dict):
            raise RuntimeError("evidence clustering service returned a non-object")
        return body

    response = await asyncio.to_thread(request_cluster)
    raw_assignments = response.get("assignments")
    if not isinstance(raw_assignments, list):
        raise RuntimeError("evidence clustering response has no assignments")
    by_key = {example_key(example): example for example in examples}
    assignments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_assignments:
        if not isinstance(raw, dict):
            raise RuntimeError("invalid evidence clustering assignment")
        key = str(raw.get("id") or "")
        if key not in by_key or key in seen:
            raise RuntimeError(f"unexpected or duplicate clustering item ID: {key}")
        seen.add(key)
        example = by_key[key]
        assignments.append(
            {
                "example_key": key,
                "row_index": example.row_index,
                "source_id": source_id_for_case(example.case),
                "cluster_id": int(raw["cluster_id"]),
                "similarity_to_centroid": float(raw["similarity_to_centroid"]),
            }
        )
    if seen != set(by_key):
        raise RuntimeError("evidence clustering did not return every training question")
    assignments.sort(key=lambda item: int(item["row_index"]))
    return {
        "strategy": "bge_agglomerative_threshold",
        "question_only": True,
        "num_items": len(examples),
        "num_clusters": int(response.get("num_clusters", 0)),
        "similarity_threshold": float(response["similarity_threshold"]),
        "threshold_selection": str(response.get("threshold_selection") or "fixed"),
        "multi_question_ratio": float(response.get("multi_question_ratio") or 0.0),
        "num_multi_question_items": int(
            response.get("num_multi_question_items") or 0
        ),
        "target_multi_question_ratio_min": target_multi_question_ratio_min,
        "target_multi_question_ratio_max": target_multi_question_ratio_max,
        "linkage": str(linkage),
        "service_url": service_url,
        "model_path": response.get("model_path"),
        "assignments": assignments,
    }

def build_epoch_windows(
    examples: Sequence[DatasetExample],
    manifest: dict[str, Any],
    *,
    window_size: int,
    epoch_seed: int,
) -> list[list[DatasetExample]]:
    """Shuffle fixed clusters and split them into balanced, non-overlapping windows."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    by_key = {example_key(example): example for example in examples}
    clusters: dict[int, list[DatasetExample]] = defaultdict(list)
    for assignment in manifest.get("assignments") or []:
        key = str(assignment.get("example_key") or "")
        if key not in by_key:
            raise ValueError(f"cluster manifest refers to unknown example: {key}")
        clusters[int(assignment["cluster_id"])].append(by_key[key])
    if sum(len(values) for values in clusters.values()) != len(examples):
        raise ValueError("cluster manifest does not cover the selected training set")

    windows: list[list[DatasetExample]] = []
    for cluster_id in sorted(clusters):
        members = list(clusters[cluster_id])

        random.Random(f"{epoch_seed}:{cluster_id}").shuffle(members)
        window_count = (len(members) + window_size - 1) // window_size
        small_size, remainder = divmod(len(members), window_count)
        offset = 0
        for index in range(window_count):
            size = small_size + (1 if index < remainder else 0)
            windows.append(members[offset : offset + size])
            offset += size
    random.Random(epoch_seed).shuffle(windows)
    return windows

def window_schedule_record(
    windows: Sequence[Sequence[DatasetExample]],
    *,
    epoch_id: int,
    seed: int,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cluster_by_key = {
        str(item.get("example_key") or ""): int(item["cluster_id"])
        for item in (manifest or {}).get("assignments") or []
    }
    return {
        "epoch_id": epoch_id,
        "seed": seed,
        "num_windows": len(windows),
        "windows": [
            {
                "batch_id": batch_id,
                "cluster_id": (
                    cluster_by_key.get(example_key(window[0]))
                    if window
                    else None
                ),
                "single_case_fallback": len(window) == 1,
                "row_indices": [example.row_index for example in window],
                "source_ids": [source_id_for_case(example.case) for example in window],
            }
            for batch_id, window in enumerate(windows)
        ],
    }
