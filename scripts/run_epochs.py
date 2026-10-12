from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from amf.datasets import SUPPORTED_DATASET_TYPES, load_dataset
from amf.checkpoint import RunCheckpoint
from amf.epoch import EpochRunner
from amf.memory.epoch_review import EpochSummaryReviewer
from amf.evidence_windows import (
    build_epoch_windows,
    cluster_examples,
    question_text,
    window_schedule_record,
)
from amf.evaluators import (
    build_dataset_evaluator,
    collect_evaluation_failures,
    evaluator_requires_llm,
    resolve_evaluator_type,
)
from amf.llm import LLMUsageTracker
from amf.main import (
    build_llm_client,
    build_runner,
    ensure_method_active,
    sqlite_memory_store_path,
)
from amf.memory import (
    EpochConflictResolver,
    EpochMemoryUtilityJudge,
    EpochMemoryWriter,
    snapshot_sqlite,
)
from amf.skills import DEFAULT_EPOCH_SKILL, EpochSkillStore, EpochSkillUpdater
from amf.utils.config import load_config, merge_config
from amf.utils.paths import nested_artifact_dir, run_namespace, sanitize_path_part

def utc_run_id() -> str:

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

def validate_final_review_fork(
    *,
    enabled: bool,
    num_epochs: int,
    test_each_epoch: bool,
    has_test_examples: bool,
    summary_review_enabled: bool,
) -> None:
    if not enabled:
        return
    if num_epochs != 2:
        raise ValueError(
            "--fork-final-review is a temporary two-epoch diagnostic and "
            f"requires --num-epochs 2 (got {num_epochs})."
        )
    if test_each_epoch:
        raise ValueError(
            "--fork-final-review cannot be combined with --test-each-epoch; "
            "it creates the two final branch tests instead."
        )
    if not summary_review_enabled:
        raise ValueError(
            "--fork-final-review requires epoch summary review to be enabled. "
            "Use --summary-review with this diagnostic."
        )
    if not has_test_examples:
        raise ValueError(
            "--fork-final-review requires an enabled, non-empty test dataset."
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train epochs with versioned Memory updates, global Skill updates, "
            "frozen Validation, and an optional frozen Test phase."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="datasets/healthbench/healthbench_hard_train.jsonl",
    )
    parser.add_argument("--config", default="configs/epoch.yaml")
    parser.add_argument(
        "--dataset-config",
        default=None,
        help=(
            "Dataset-specific config. Defaults to "
            "configs/datasets/<dataset-type>.yaml when present."
        ),
    )
    parser.add_argument("--llm-config", default="configs/nv30a3_release.yaml")
    parser.add_argument(
        "--judge-llm-config",
        default="configs/nv30a3_release.yaml",
    )
    parser.add_argument(
        "--dataset-type",
        default="healthbench",
        choices=SUPPORTED_DATASET_TYPES,
    )

    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--summary-review",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override epoch-end summary review without changing the YAML config.",
    )
    parser.add_argument(
        "--fork-final-review",
        action="store_true",
        help=(
            "For exactly 2 epochs, preserve the final pre-review Memory, apply "
            "the final review to a branch, and test both branches."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Maximum semantic evidence-window size. Use 1 to restore the "
            "original online single-case writer."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=None,
        help=(
            "Override the answering/writer model seed. The Judge keeps the "
            "seed from --judge-llm-config."
        ),
    )
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--memory-db-name",
        default=None,
        help=(
            "Resume this existing named database. If omitted, create a new "
            "database named after this run."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--initial-memory-snapshot", type=Path, default=None,
        help="Initialize a separate run from an existing Epoch SQLite snapshot."
    )
    parser.add_argument(
        "--review-previous-epoch", type=Path, default=None,
        help="For a one-round diagnostic, compare against this saved training Epoch directory."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing epoch run. Requires the original --run-id.",
    )
    parser.add_argument("--output-dir", default="data/runs")
    parser.add_argument("--validation-dataset", default=None)
    parser.add_argument("--test-dataset", default=None)
    parser.add_argument(
        "--run-validation", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--run-test", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--test-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run a frozen read-only Test evaluation for every completed Epoch. "
            "The default is the existing final-Epoch-only Test."
        ),
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help=(
            "Batch/concurrency size for frozen Test only. This does not change "
            "the training Evidence Window batch_size."
        ),
    )
    parser.add_argument(
        "--response",
        action="append",
        default=None,
        help="Scripted agent response; repeat for all model turns across all epochs.",
    )
    parser.add_argument(
        "--grader-response",
        action="append",
        default=None,
        help="Scripted rubric grader response; repeat for every rubric call.",
    )
    parser.add_argument(
        "--memory-response",
        action="append",
        default=None,
        help="Scripted epoch memory-writer response; repeat for every episode.",
    )
    parser.add_argument(
        "--skill-response",
        action="append",
        default=None,
        help="Scripted skill-analysis/update response for scripted runs.",
    )
    parser.add_argument(
        "--utility-response",
        action="append",
        default=None,
        help="Scripted Memory utility-judge response for scripted runs.",
    )
    parser.add_argument(
        "--conflict-response",
        action="append",
        default=None,
        help="Scripted Epoch-end conflict-resolution response for scripted runs.",
    )
    return parser.parse_args()

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def evidence_cluster_cache_identity(
    examples: list[Any],
    *,
    dataset_path: str,
    dataset_type: str,
    start_index: int,
    end_index: int,
    model_id: str,
    linkage: str,
    similarity_threshold: float | None,
    target_ratio_min: float,
    target_ratio_max: float,
) -> dict[str, Any]:
    """Describe every input that can change a dataset clustering result."""

    question_payload = [
        {"row_index": example.row_index, "text": question_text(example)}
        for example in examples
    ]
    question_digest = hashlib.sha256(
        json.dumps(
            question_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selection = (
        {"mode": "fixed", "similarity_threshold": similarity_threshold}
        if similarity_threshold is not None
        else {
            "mode": "target_multi_question_ratio",
            "minimum": target_ratio_min,
            "maximum": target_ratio_max,
        }
    )
    return {
        "cache_version": 1,
        "dataset_path": dataset_path,
        "dataset_type": dataset_type,
        "start_index": start_index,
        "end_index": end_index,
        "num_items": len(examples),
        "question_text_sha256": question_digest,
        "model_id": model_id,
        "linkage": linkage,
        "threshold_selection": selection,
    }

def evidence_cluster_cache_path(
    cache_root: Path,
    identity: dict[str, Any],
) -> Path:
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    dataset = sanitize_path_part(str(identity["dataset_type"]))
    source = sanitize_path_part(Path(str(identity["dataset_path"])).stem)
    start, end = identity["start_index"], identity["end_index"]
    return cache_root / f"{dataset}__{source}__{start}-{end}__{identity_digest}.json"

async def main() -> None:
    args = parse_args()
    config = merge_config(
        load_config(ROOT / "configs/base.yaml"),
        load_config(ROOT / args.config),
    )
    dataset_config_path = (
        ROOT / args.dataset_config
        if args.dataset_config
        else ROOT / "configs" / "datasets" / f"{args.dataset_type}.yaml"
    )
    if dataset_config_path.exists():
        config = merge_config(config, load_config(dataset_config_path))
    if args.llm_config:
        config = merge_config(config, load_config(ROOT / args.llm_config))
    if args.llm_seed is not None:
        llm_config = dict(config.get("llm") or {})
        llm_config["seed"] = args.llm_seed
        config["llm"] = llm_config
    ensure_method_active(config)
    config["dataset_name"] = args.dataset_type

    epoch_config = dict(config.get("epoch") or {})
    if args.summary_review is not None:
        summary_review_config = dict(epoch_config.get("summary_review") or {})
        summary_review_config["enabled"] = bool(args.summary_review)
        epoch_config["summary_review"] = summary_review_config

        config["epoch"] = epoch_config
    writer_config = dict(epoch_config.get("memory_writer") or {})
    configured_prompt_dataset = writer_config.get(
        "prompt_dataset", epoch_config.get("prompt_dataset")
    )
    normalized_dataset_type = args.dataset_type.lower().replace("-", "_")
    expected_prompt_dataset = normalized_dataset_type
    prompt_aliases = {
        "mmmu_pro_medical": "mmmu_medical",
    }
    prompt_dataset = str(
        configured_prompt_dataset
        if configured_prompt_dataset is not None
        else prompt_aliases.get(expected_prompt_dataset, expected_prompt_dataset)
    )
    normalized_prompt_dataset = prompt_dataset.lower().replace("-", "_")
    allowed_prompt_datasets = {
        expected_prompt_dataset,
        prompt_aliases.get(expected_prompt_dataset, expected_prompt_dataset),
    }
    if normalized_prompt_dataset not in allowed_prompt_datasets:

        raise ValueError(
            "Epoch Memory Writer prompts are dataset-specific: "
            f"dataset_type={args.dataset_type!r}, prompt_dataset={prompt_dataset!r}. "
            f"Expected prompt bundle: prompts/{expected_prompt_dataset}/. "
            "Only explicitly declared prompt aliases are allowed."
        )
    start_index = (
        args.start_index
        if args.start_index is not None
        else int(epoch_config.get("start_index", 0))
    )
    configured_end_index = epoch_config.get("end_index")
    end_index = (
        args.end_index
        if args.end_index is not None
        else (
            int(configured_end_index)
            if configured_end_index is not None
            else None
        )
    )
    num_epochs = (
        args.num_epochs
        if args.num_epochs is not None
        else int(epoch_config.get("num_epochs", 2))
    )
    fork_final_review = bool(args.fork_final_review)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(epoch_config.get("batch_size", 1))
    )
    seed = args.seed if args.seed is not None else int(epoch_config.get("seed", 0))
    shuffle_each_epoch = (
        args.shuffle
        if args.shuffle is not None
        else bool(epoch_config.get("shuffle_each_epoch", False))
    )
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    evidence_window_config = dict(epoch_config.get("evidence_windows") or {})
    evidence_windows_enabled = bool(
        evidence_window_config.get("enabled", True)
    ) and batch_size > 1
    cluster_threshold_value = evidence_window_config.get("similarity_threshold")
    cluster_similarity_threshold = (
        float(cluster_threshold_value) if cluster_threshold_value is not None else None
    )
    if cluster_similarity_threshold is not None and not (
        -1.0 <= cluster_similarity_threshold <= 1.0
    ):
        raise ValueError(
            "epoch.evidence_windows.similarity_threshold must be between -1 and 1"
        )
    cluster_target_ratio_min = float(
        evidence_window_config.get("target_multi_question_ratio_min", 0.30)
    )
    cluster_target_ratio_max = float(
        evidence_window_config.get("target_multi_question_ratio_max", 0.40)
    )
    if not (
        0.0 <= cluster_target_ratio_min <= cluster_target_ratio_max <= 1.0
    ):
        raise ValueError(
            "Evidence-window target ratios must satisfy 0 <= min <= max <= 1"
        )
    cluster_linkage = str(
        evidence_window_config.get("linkage", "complete")
    ).strip().lower()
    if cluster_linkage not in {"complete", "average", "single"}:
        raise ValueError(
            "epoch.evidence_windows.linkage must be complete, average, or single"
        )
    cluster_model_id = str(
        evidence_window_config.get("clustering_model_id") or "BAAI/bge-m3"
    )
    cluster_cache_root = ROOT / str(
        evidence_window_config.get("cache_dir") or "data/evidence_clusters"
    )

    dataset = load_dataset(args.dataset_type, ROOT / args.dataset)
    examples = list(
        dataset.iter_examples(start_index=start_index, end_index=end_index)
    )
    if not examples:
        raise ValueError("The selected dataset slice is empty.")

    validation_config = dict(epoch_config.get("validation") or {})
    test_config = dict(epoch_config.get("test") or {})
    if args.run_validation is not None:
        validation_config["enabled"] = args.run_validation
    if args.run_test is not None:
        test_config["enabled"] = args.run_test
    validation_path_value = args.validation_dataset or validation_config.get("dataset")
    test_path_value = args.test_dataset or test_config.get("dataset")

    def load_phase_examples(
        path_value: Any,
        phase_config: dict[str, Any],
    ) -> list[Any]:
        if not path_value or not bool(phase_config.get("enabled", True)):
            return []
        phase_path = ROOT / str(path_value)
        if not phase_path.exists():
            raise FileNotFoundError(f"Configured phase dataset does not exist: {phase_path}")
        phase_dataset = load_dataset(args.dataset_type, phase_path)
        phase_start = int(phase_config.get("start_index", 0))
        phase_end_value = phase_config.get("end_index")
        phase_end = int(phase_end_value) if phase_end_value is not None else None
        return list(phase_dataset.iter_examples(start_index=phase_start, end_index=phase_end))

    validation_examples = load_phase_examples(validation_path_value, validation_config)
    test_examples = load_phase_examples(test_path_value, test_config)
    test_each_epoch = (
        args.test_each_epoch
        if args.test_each_epoch is not None
        else bool((epoch_config or {}).get("test_each_epoch", False))
    )
    epoch_config["test_each_epoch"] = test_each_epoch
    validate_final_review_fork(
        enabled=fork_final_review,
        num_epochs=num_epochs,
        test_each_epoch=test_each_epoch,
        has_test_examples=bool(test_examples),
        summary_review_enabled=bool(
            (epoch_config.get("summary_review") or {}).get("enabled", False)
        ),
    )
    config["epoch"] = epoch_config

    scripted = args.response is not None
    method_name, model_name = run_namespace(config, scripted=scripted)
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id.")
    run_id = args.run_id or utc_run_id()
    raw_db_name = str(args.memory_db_name or run_id)
    if raw_db_name.endswith(".sqlite"):
        raw_db_name = raw_db_name[: -len(".sqlite")]
    memory_db_name = sanitize_path_part(raw_db_name)

    initial_snapshot = args.initial_memory_snapshot.resolve() if args.initial_memory_snapshot else None
    if initial_snapshot is not None and not initial_snapshot.is_file():
        raise FileNotFoundError(initial_snapshot)
    previous_review_dir = args.review_previous_epoch.resolve() if args.review_previous_epoch else None
    if previous_review_dir is not None:
        if num_epochs != 1 or not (epoch_config.get("summary_review") or {}).get("enabled", False):
            raise ValueError("External review history requires num_epochs=1 and summary_review.enabled=true")
        if initial_snapshot != (previous_review_dir / "memory.sqlite").resolve():
            raise ValueError("Initial snapshot must come from the same Epoch as the review history")
        for filename in ("learning_records.jsonl", "evaluations.jsonl", "metrics.json"):
            if not (previous_review_dir / filename).is_file():
                raise FileNotFoundError(previous_review_dir / filename)
        config["review_previous_epoch"] = str(previous_review_dir)
    memory_db_action = "resume" if args.resume or initial_snapshot else "create"
    config["memory_db_name"] = memory_db_name
    config["memory_store_mode"] = memory_db_action
    if initial_snapshot:
        config["initial_memory_snapshot"] = str(initial_snapshot)

    identity_prompt_root = str(
        writer_config.get(
            "prompt_root",
            epoch_config.get("prompt_root", config.get("prompt_root", "prompts")),
        )
    )
    run_dir = ROOT / nested_artifact_dir(
        args.output_dir,
        config,
        scripted=scripted,
    ) / run_id
    checkpoint = RunCheckpoint(
        run_dir,
        run_id=run_id,
        resume=args.resume,
        identity={
            "kind": "epochs",
            **({"review_previous_epoch": str(previous_review_dir)} if previous_review_dir else {}),
            **({"summary_review": epoch_config["summary_review"]}
               if (epoch_config.get("summary_review") or {}).get("enabled", False) else {}),
            **({"initial_memory_snapshot": str(initial_snapshot)} if initial_snapshot else {}),
            "method": method_name,
            "model": model_name,
            "dataset_path": str(args.dataset),
            "dataset_type": args.dataset_type,
            "start_index": start_index,
            "end_index": end_index,
            "num_epochs": num_epochs,
            "test_each_epoch": test_each_epoch,
            "fork_final_review": fork_final_review,
            "batch_size": batch_size,
            "parallel_episodes": bool(epoch_config.get("parallel_episodes", True)),
            "seed": seed,
            "shuffle_each_epoch": shuffle_each_epoch,
            "evidence_windows": {
                "enabled": evidence_windows_enabled,
                "similarity_threshold": cluster_similarity_threshold,
                "target_multi_question_ratio_min": cluster_target_ratio_min,
                "target_multi_question_ratio_max": cluster_target_ratio_max,
                "linkage": cluster_linkage,
                "clustering_service_url": evidence_window_config.get(
                    "clustering_service_url"
                ),
                "clustering_service_timeout": evidence_window_config.get(
                    "clustering_service_timeout"
                ),
                "clustering_model_id": cluster_model_id,
                "cache_dir": str(cluster_cache_root),
            },
            "method_config": str(args.config),
            "dataset_config": (
                str(dataset_config_path) if dataset_config_path.exists() else None
            ),
            "prompt_dataset": prompt_dataset,
            "prompt_root": identity_prompt_root,
            "llm_config": str(args.llm_config),
            "judge_llm_config": str(args.judge_llm_config),
            "memory_db_name": memory_db_name,

            "memory_retrieval": {
                key: dict(config.get("memory_retrieval") or {}).get(key)
                for key in (
                    "bm25_candidate_k",
                    "similarity_service_url",
                    "similarity_service_timeout",
                    "writer_bge_similarity_threshold",
                    "answer_bge_similarity_threshold",
                    "bge_final_top_k",
                )
            },
            "validation_dataset": str(validation_path_value) if validation_path_value else None,
            "test_dataset": str(test_path_value) if test_path_value else None,
        },
    )
    config["log_dir"] = str(run_dir / "trajectories")

    if args.resume:
        checkpoint.write_json(run_dir / "config.resume.json", config)
    else:
        checkpoint.write_json(run_dir / "config.json", config)

    cluster_manifest: dict[str, Any] | None = None
    cluster_manifest_path = run_dir / "evidence_clusters.json"
    if evidence_windows_enabled:
        if args.resume:
            if not cluster_manifest_path.exists():
                raise FileNotFoundError(
                    "Cannot resume an evidence-window run without "
                    f"{cluster_manifest_path}"
                )
            cluster_manifest = checkpoint.read_json(cluster_manifest_path)
        else:
            cache_identity = evidence_cluster_cache_identity(
                examples,
                dataset_path=str(args.dataset),
                dataset_type=args.dataset_type,
                start_index=start_index,
                end_index=end_index,
                model_id=cluster_model_id,
                linkage=cluster_linkage,
                similarity_threshold=cluster_similarity_threshold,
                target_ratio_min=cluster_target_ratio_min,
                target_ratio_max=cluster_target_ratio_max,
            )
            cluster_cache_path = evidence_cluster_cache_path(
                cluster_cache_root,
                cache_identity,
            )
            if cluster_cache_path.exists():
                cached = checkpoint.read_json(cluster_cache_path)
                if cached.get("cache_identity") != cache_identity:
                    raise ValueError(
                        f"Evidence cluster cache identity mismatch: {cluster_cache_path}"
                    )
                cluster_manifest = cached
                print(
                    "[amf] action=evidence_window.cluster_cache_hit "
                    f"path={cluster_cache_path}",
                    flush=True,
                )
            else:
                clustering_url = str(
                    evidence_window_config.get("clustering_service_url")
                    or "http://127.0.0.1:9000/cluster"
                )
                print(
                    f"[amf] action=evidence_window.cluster questions={len(examples)} "
                    + (
                        f"threshold={cluster_similarity_threshold} "
                        if cluster_similarity_threshold is not None
                        else (
                            "target_multi_ratio="
                            f"{cluster_target_ratio_min:.2f}-"
                            f"{cluster_target_ratio_max:.2f} "
                        )
                    )
                    + f"linkage={cluster_linkage}",
                    flush=True,
                )
                cluster_manifest = await cluster_examples(
                    examples,
                    service_url=clustering_url,
                    similarity_threshold=cluster_similarity_threshold,
                    target_multi_question_ratio_min=(
                        cluster_target_ratio_min
                        if cluster_similarity_threshold is None
                        else None
                    ),
                    target_multi_question_ratio_max=(
                        cluster_target_ratio_max
                        if cluster_similarity_threshold is None
                        else None
                    ),
                    linkage=cluster_linkage,
                    timeout_seconds=float(
                        evidence_window_config.get("clustering_service_timeout", 300)
                    ),
                )
                cluster_manifest["cache_identity"] = cache_identity
                checkpoint.write_json(cluster_cache_path, cluster_manifest)
                print(
                    "[amf] action=evidence_window.cluster_cache_write "
                    f"path={cluster_cache_path}",
                    flush=True,
                )
            print(
                "[amf] action=evidence_window.cluster_complete "
                f"clusters={cluster_manifest['num_clusters']} "
                f"selected_threshold="
                f"{cluster_manifest['similarity_threshold']:.6f} "
                f"multi_question_ratio="
                f"{cluster_manifest['multi_question_ratio']:.3f}",
                flush=True,
            )
            checkpoint.write_json(cluster_manifest_path, cluster_manifest)
        if int(cluster_manifest.get("num_items", -1)) != len(examples):
            raise ValueError(
                "Evidence cluster manifest does not match the selected training set"
            )

    epoch_windows: dict[int, list[list[Any]]] = {}
    for epoch_id in range(num_epochs):
        if cluster_manifest is not None:
            windows = build_epoch_windows(
                examples,
                cluster_manifest,
                window_size=batch_size,
                epoch_seed=seed + epoch_id,
            )
        else:
            ordered = list(examples)
            if shuffle_each_epoch:
                random.Random(seed + epoch_id).shuffle(ordered)
            windows = [
                ordered[index : index + batch_size]
                for index in range(0, len(ordered), batch_size)
            ]
        epoch_windows[epoch_id] = windows

    usage_path = run_dir / "usage.jsonl"
    usage_tracker = LLMUsageTracker(checkpoint.read_records(usage_path))
    if initial_snapshot and not args.resume:
        target = sqlite_memory_store_path(config, method=method_name, scripted=scripted)
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("xb"):
            pass
        with sqlite3.connect(initial_snapshot.as_uri() + "?mode=ro", uri=True) as source:
            if source.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memory_versions'").fetchone()[0] != 1:
                raise ValueError("Initial snapshot is not an Epoch memory database")
            with sqlite3.connect(target) as destination:
                source.backup(destination)
        checkpoint.write_json(run_dir / "initial_memory.json", {
            "source_snapshot": str(initial_snapshot), "destination": str(target),
            "note": "Independent diagnostic run; local epoch numbering starts at zero.",
        })
    runner = build_runner(
        config,
        scripted_responses=args.response,
        usage_tracker=usage_tracker,
    )
    evaluation_config = dict(config.get("evaluation") or {})
    if not bool(evaluation_config.get("enabled", False)):
        raise ValueError("Epoch runs require dataset evaluation to be enabled.")
    evaluator_type = resolve_evaluator_type(args.dataset_type, evaluation_config)
    grader = None
    if evaluator_requires_llm(evaluator_type):
        judge_config = merge_config(
            load_config(ROOT / "configs/base.yaml"),
            load_config(ROOT / args.judge_llm_config),
        )
        grader = build_llm_client(
            judge_config,
            scripted_responses=args.grader_response,
            usage_tracker=usage_tracker,
            usage_role="healthbench_judge",
        )
    evaluator = build_dataset_evaluator(
        evaluator_type,
        evaluation_config,
        grader=grader,
    )

    retrieval_config = dict(config.get("memory_retrieval") or {})
    writer_llm_config = config if writer_config.get("provider", "same_llm") == "same_llm" else {
        "llm": writer_config
    }

    prompt_root_value = writer_config.get(
        "prompt_root",
        epoch_config.get("prompt_root", config.get("prompt_root", "prompts")),
    )
    prompt_root = Path(str(prompt_root_value))
    if not prompt_root.is_absolute():
        prompt_root = ROOT / prompt_root
    memory_writer = EpochMemoryWriter(
        build_llm_client(
            writer_llm_config,
            scripted_responses=args.memory_response,
            usage_tracker=usage_tracker,
            usage_role="epoch_memory_writer",
        ),
        method=method_name,
        max_retries=int(writer_config.get("max_retries", 3)),
        strict_json=bool(writer_config.get("strict_json", False)),
        use_json_response_format=bool(
            writer_config.get("use_json_response_format", False)
        ),
        prompt_root=prompt_root,
        prompt_dataset=prompt_dataset,
    )
    review_config = dict(epoch_config.get("summary_review") or {})
    summary_reviewer = None
    if bool(review_config.get("enabled", False)):
        summary_reviewer = EpochSummaryReviewer(
            build_llm_client(config, usage_tracker=usage_tracker,
                             usage_role="epoch_summary_review"),
            runner.providers.get("memory").store,
            prompt_root / prompt_dataset,
            max_retries=int(review_config.get("max_retries", 3)),
            version_retries=int(review_config.get("version_retries", 3)),
            max_input_tokens=int(review_config.get("max_input_tokens", 15000)),
            max_output_tokens=int(review_config.get("max_output_tokens", 5000)),
            context_tokens=int(review_config.get("context_tokens", 18000)),
            safety_margin_tokens=int(review_config.get("safety_margin_tokens", 500)),
            trace=bool(config.get("trace_actions", False)),
            save_usage=lambda: checkpoint.write_records(usage_path, usage_tracker.records),
        )
    conflict_config = dict(epoch_config.get("conflict_resolution") or {})
    conflict_llm_config = (
        config
        if conflict_config.get("provider", "same_llm") == "same_llm"
        else {"llm": conflict_config}
    )
    conflict_resolver = (
        EpochConflictResolver(
            build_llm_client(
                conflict_llm_config,
                scripted_responses=args.conflict_response,
                usage_tracker=usage_tracker,
                usage_role="epoch_conflict_resolver",
            ),
            max_retries=int(conflict_config.get("max_retries", 3)),
            use_json_response_format=bool(
                conflict_config.get("use_json_response_format", True)
            ),
            min_abandon_sources=int(
                conflict_config.get("min_abandon_sources", 2)
            ),
            max_input_memories=int(
                conflict_config.get("max_input_memories", 5)
            ),
            prompt_root=str(prompt_root),
            prompt_dataset=prompt_dataset,
        )
        if bool(conflict_config.get("enabled", True))
        else None
    )
    skill_config = dict(epoch_config.get("skill") or {})
    dynamic_skill_update_enabled = bool(
        skill_config.get("dynamic_update_enabled", False)
    )
    static_skill = str(
        skill_config.get("static_prompt") or DEFAULT_EPOCH_SKILL
    ).strip()
    skill_store = EpochSkillStore(
        run_dir / "skills",
        max_tokens=int(skill_config.get("max_tokens", 3000)),
    )
    stored_skill, initial_skill_record = skill_store.load_current()
    initial_skill = stored_skill if dynamic_skill_update_enabled else static_skill
    if (
        initial_skill_record is None
        or (
            not dynamic_skill_update_enabled
            and stored_skill.strip() != static_skill
        )
    ):
        initial_skill_record = skill_store.save(
            initial_skill,
            epoch_id=-1,
            parent_version=(
                initial_skill_record.get("version")
                if initial_skill_record is not None
                else None
            ),
            selected_count=0,
        )
    skill_llm_config = config if skill_config.get("provider", "same_llm") == "same_llm" else {
        "llm": skill_config
    }
    skill_updater = (
        EpochSkillUpdater(
            build_llm_client(
                skill_llm_config,
                scripted_responses=args.skill_response,
                usage_tracker=usage_tracker,
                usage_role="epoch_skill_updater",
            ),
            skill_store,
            max_batch_size=int(skill_config.get("max_batch_size", 5)),
            max_retries=int(skill_config.get("max_retries", 2)),
        )
        if dynamic_skill_update_enabled
        else None
    )

    runner.set_global_skill(initial_skill)
    utility_config = dict(epoch_config.get("memory_utility_judge") or {})
    utility_llm_config = config if utility_config.get("provider", "same_llm") == "same_llm" else {
        "llm": utility_config
    }
    memory_utility_judge = (
        EpochMemoryUtilityJudge(
            build_llm_client(
                utility_llm_config,
                scripted_responses=args.utility_response,
                usage_tracker=usage_tracker,
                usage_role="epoch_memory_utility_judge",
            ),
            max_retries=int(utility_config.get("max_retries", 2)),
            use_json_response_format=bool(
                utility_config.get("use_json_response_format", True)
            ),
        )
        if bool(utility_config.get("enabled", True))
        else None
    )
    epoch_runner = EpochRunner(
        runner,
        evaluator,
        memory_writer,
        batch_size=batch_size,
        max_new_memories=(
            int(epoch_config["max_new_memories"])
            if epoch_config.get("max_new_memories") is not None
            else None
        ),
        checkpoint_run_id=run_id,
        memory_utility_judge=memory_utility_judge,
        writer_top_k=int(
            retrieval_config.get(
                "bge_final_top_k",
                writer_config.get("top_k", 3),
            )
        ),
        writer_similarity_threshold=float(
            retrieval_config.get("writer_bge_similarity_threshold", 0.71)
        ),
        parallel_episodes=bool(epoch_config.get("parallel_episodes", True)),
    )

    if args.resume:
        provider = runner.providers.get("memory")
        store = getattr(provider, "store", None)
        if (
            store is not None
            and hasattr(store, "delete_ids")
            and not getattr(store, "supports_versioning", False)
        ):
            checkpoint_items = await store.list(
                filters={"metadata.checkpoint_run_id": run_id}
            )
            uncommitted_ids = [
                item.id
                for item in checkpoint_items
                if str(item.metadata.get("checkpoint_batch_unit") or "")
                not in checkpoint.completed_units
            ]
            store.delete_ids(uncommitted_ids)

    epoch_summaries: list[dict[str, Any]] = []
    previous_train_records: dict[str, dict[str, Any]] = {}
    final_validation_score: float | None = None
    final_epoch_id: int | None = None
    final_memory_snapshot: Path | None = None
    final_pre_review_snapshot: Path | None = None
    final_skill_record: dict[str, Any] | None = initial_skill_record

    resumed_epoch_artifacts: dict[int, dict[str, Any]] = {}
    if args.resume:
        for prior_epoch_id in range(num_epochs):
            prior_dir = run_dir / f"epoch_{prior_epoch_id:03d}"
            if f"epoch:{prior_epoch_id}:complete" not in checkpoint.completed_units:
                continue
            total_prior_batches = len(epoch_windows[prior_epoch_id])
            if any(
                f"epoch:{prior_epoch_id}:batch:{batch_id}"
                not in checkpoint.completed_units
                for batch_id in range(total_prior_batches)
            ):

                continue
            learning_records = checkpoint.read_records(
                prior_dir / "learning_records.jsonl"
            )
            prior_predictions = checkpoint.read_records(
                prior_dir / "predictions.jsonl"
            )
            metrics = checkpoint.read_json(prior_dir / "metrics.json") if (
                prior_dir / "metrics.json"
            ).exists() else {}
            skill_update = checkpoint.read_json(prior_dir / "skill_update.json") if (
                prior_dir / "skill_update.json"
            ).exists() else {}
            resumed_epoch_artifacts[prior_epoch_id] = {
                "learning_records": learning_records,
                "metrics": metrics,
                "skill_update": skill_update,
                "artifact_dir": str(prior_dir),
                "row_order": [
                    record.get("row_index")
                    for record in prior_predictions
                    if record.get("row_index") is not None
                ],
            }
        if resumed_epoch_artifacts:
            latest_epoch_id = max(resumed_epoch_artifacts)
            latest_skill_update = resumed_epoch_artifacts[latest_epoch_id]["skill_update"]
            runner.set_global_skill(
                str(latest_skill_update.get("skill") or initial_skill)
                if dynamic_skill_update_enabled
                else initial_skill
            )
            latest_records = resumed_epoch_artifacts[latest_epoch_id]["learning_records"]
            previous_train_records = {
                str(record.get("source_id") or record.get("case_id") or record.get("row_index")): record
                for record in latest_records
                if record.get("score") is not None
            }
        final_binding_path = run_dir / "final_checkpoint.json"
        if final_binding_path.exists():
            final_binding = checkpoint.read_json(final_binding_path)
            value = final_binding.get("validation_score")
            if isinstance(value, int | float):
                final_validation_score = float(value)
            if final_binding.get("epoch_id") is not None:
                final_epoch_id = int(final_binding["epoch_id"])
            candidate_snapshot = Path(
                str(final_binding.get("memory_snapshot") or "")
            )
            if candidate_snapshot.is_file():
                final_memory_snapshot = candidate_snapshot
            candidate_pre_review = Path(
                str(final_binding.get("pre_review_memory_snapshot") or "")
            )
            if candidate_pre_review.is_file():
                final_pre_review_snapshot = candidate_pre_review

    for prior_epoch_id in sorted(resumed_epoch_artifacts):
        artifact = resumed_epoch_artifacts[prior_epoch_id]
        epoch_summaries.append(
            {
                **dict(artifact["metrics"]),
                "artifact_dir": artifact["artifact_dir"],
                "row_order": artifact["row_order"],
                "resumed": True,
            }
        )

    for epoch_id in range(num_epochs):
        if epoch_id in resumed_epoch_artifacts:

            continue
        windows = epoch_windows[epoch_id]
        ordered_examples = [example for window in windows for example in window]
        epoch_dir = run_dir / f"epoch_{epoch_id:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        schedule_path = epoch_dir / "evidence_windows.json"
        schedule = window_schedule_record(
            windows,
            epoch_id=epoch_id,
            seed=seed + epoch_id,
            manifest=cluster_manifest,
        )
        if args.resume and schedule_path.exists():
            if checkpoint.read_json(schedule_path) != schedule:
                raise ValueError(
                    f"Evidence-window schedule changed while resuming Epoch {epoch_id}"
                )
        else:
            checkpoint.write_json(schedule_path, schedule)
        artifact_paths = {
            "predictions": epoch_dir / "predictions.jsonl",
            "evaluations": epoch_dir / "evaluations.jsonl",
            "learning_records": epoch_dir / "learning_records.jsonl",
            "memory_writes": epoch_dir / "memory_writes.jsonl",
        }
        evaluation_failures_path = epoch_dir / "evaluation_failures.jsonl"
        accumulated = {
            name: checkpoint.read_records(path)
            for name, path in artifact_paths.items()
        }
        if args.resume:

            for name, records in accumulated.items():
                accumulated[name] = [
                    record
                    for record in records
                    if (
                        f"epoch:{epoch_id}:batch:{int(record.get('batch_id', -1))}"
                        in checkpoint.completed_units
                    )
                ]
                checkpoint.write_records(artifact_paths[name], accumulated[name])
        total_batches = len(windows)
        completed_batch_ids = {
            batch_id
            for batch_id in range(total_batches)
            if f"epoch:{epoch_id}:batch:{batch_id}" in checkpoint.completed_units
        }
        start_batch_id = 0
        while start_batch_id in completed_batch_ids:
            start_batch_id += 1

        async def save_batch(
            batch_id: int,
            batch_artifacts: dict[str, list[dict[str, Any]]],
        ) -> None:
            """Commit one batch's artifacts, then expose it as completed."""

            for name, records in batch_artifacts.items():
                accumulated[name].extend(records)
                checkpoint.write_records(artifact_paths[name], accumulated[name])
            checkpoint.write_records(
                evaluation_failures_path,
                collect_evaluation_failures(
                    accumulated["evaluations"],
                    phase="train",
                ),
            )
            checkpoint.write_records(usage_path, usage_tracker.records)
            checkpoint.mark_completed(
                f"epoch:{epoch_id}:batch:{batch_id}",
                epoch_id=epoch_id,
                batch_id=batch_id,
            )

        existing_written = 0
        for write_record in accumulated["memory_writes"]:
            for committed in write_record.get("written_items") or []:
                if not isinstance(committed, dict):
                    continue
                if committed.get("action") == "add":
                    created = committed.get("created_version_ids")
                    if isinstance(created, list):
                        existing_written += len(created)
                    elif committed.get("version_ids"):
                        existing_written += len(committed["version_ids"])
                elif committed.get("id"):

                    existing_written += 1
        if start_batch_id < total_batches:

            await epoch_runner.run_epoch(
                ordered_examples,
                epoch_id=epoch_id,
                phase="train",
                previous_scores={
                    key: float(record["score"])
                    for key, record in previous_train_records.items()
                    if isinstance(record.get("score"), int | float)
                },
                start_batch_id=start_batch_id,
                initial_num_written=existing_written,
                on_batch_complete=save_batch,
                batches=windows,
            )

        metric_values: dict[str, list[float]] = {}
        for record in accumulated["evaluations"]:
            for name, value in dict(record.get("evaluation", {})).get(
                "metrics", {}
            ).items():
                if isinstance(value, int | float):
                    metric_values.setdefault(name, []).append(float(value))
        metrics: dict[str, Any] = {
            "epoch_id": epoch_id,
            "phase": "train",
            "batch_size": batch_size,
            "num_batches": total_batches,
            "num_predictions": len(accumulated["predictions"]),
            "num_memory_writes": sum(
                len(record.get("written_items") or [])
                for record in accumulated["memory_writes"]
            ),
            "evidence_windows_enabled": evidence_windows_enabled,
            "num_evidence_clusters": (
                int(cluster_manifest.get("num_clusters", 0))
                if cluster_manifest is not None
                else None
            ),
        }
        train_evaluation_failures = collect_evaluation_failures(
            accumulated["evaluations"],
            phase="train",
        )
        checkpoint.write_records(
            evaluation_failures_path,
            train_evaluation_failures,
        )
        metrics.update(
            {
                "num_valid_evaluations": (
                    len(accumulated["evaluations"])
                    - len(train_evaluation_failures)
                ),
                "num_evaluation_failures": len(train_evaluation_failures),
                "evaluation_complete": not train_evaluation_failures,
                "evaluation_failure_rows": [
                    record.get("row_index")
                    for record in train_evaluation_failures
                    if record.get("row_index") is not None
                ],
            }
        )
        for name, values in metric_values.items():
            metrics[name] = sum(values) / len(values)
            metrics[f"{name}:n_samples"] = len(values)

        store = getattr(runner.providers.get("memory"), "store", None)
        conflict_resolution_path = epoch_dir / "conflict_resolutions.json"
        if conflict_resolution_path.exists() and args.resume:
            conflict_resolution = checkpoint.read_json(conflict_resolution_path)
        elif (
            conflict_resolver is not None
            and hasattr(store, "pending_conflict_groups")
            and hasattr(store, "apply_conflict_resolution")
        ):
            usage_tracker.set_context(
                run_id=run_id,
                epoch_id=epoch_id,
                phase="epoch_conflict_resolution",
            )
            attempts = await conflict_resolver.resolve_epoch(
                store,
                epoch_id=epoch_id,
                run_id=run_id,
                trace=bool(config.get("trace_actions", False)),
            )
            persisted = (
                await store.conflict_resolutions_for_epoch(epoch_id)
                if hasattr(store, "conflict_resolutions_for_epoch")
                else []
            )
            conflict_resolution = {
                "epoch_id": epoch_id,
                "attempts": attempts,
                "persisted_resolutions": persisted,
                "num_groups": len(attempts),
                "num_committed": sum(
                    1 for record in attempts if record.get("status") == "committed"
                ),
                "num_errors": sum(
                    1 for record in attempts if record.get("status") == "error"
                ),
            }
            checkpoint.write_json(conflict_resolution_path, conflict_resolution)
        else:
            conflict_resolution = {
                "epoch_id": epoch_id,
                "attempts": [],
                "persisted_resolutions": [],
                "num_groups": 0,
                "num_committed": 0,
                "num_errors": 0,
                "enabled": conflict_resolver is not None,
            }
            checkpoint.write_json(conflict_resolution_path, conflict_resolution)

        pre_review_snapshot: Path | None = None
        if fork_final_review and epoch_id == num_epochs - 1:

            pre_review_snapshot = epoch_dir / "memory_before_review.sqlite"
            if not (args.resume and pre_review_snapshot.exists()):
                if hasattr(store, "backup_to"):
                    store.backup_to(pre_review_snapshot)
                else:
                    snapshot_sqlite(
                        Path(str(getattr(store, "path"))),
                        pre_review_snapshot,
                    )
            final_pre_review_snapshot = pre_review_snapshot
        if summary_reviewer is not None and not (
            fork_final_review and epoch_id == num_epochs - 1
        ):
            review_path = epoch_dir / "summary_review.json"
            if args.resume and review_path.exists():
                summary_review = checkpoint.read_json(review_path)
            else:
                summary_review = await summary_reviewer.review_epoch(
                    run_dir, epoch_id, run_id=run_id,
                    previous_epoch_dir=previous_review_dir,
                )
                checkpoint.write_json(review_path, summary_review)
            metrics["summary_review"] = summary_review

        current_train_records = {
            str(record.get("source_id") or record.get("case_id") or record.get("row_index")): record
            for record in accumulated["learning_records"]
            if record.get("score") is not None
        }
        skill_candidates: list[dict[str, Any]] = []
        if dynamic_skill_update_enabled and previous_train_records:
            for source_key, current in current_train_records.items():
                previous = previous_train_records.get(source_key)
                if previous is None:
                    continue
                previous_score = previous.get("score")
                current_score = current.get("score")
                if not isinstance(previous_score, int | float) or not isinstance(
                    current_score, int | float
                ):
                    continue
                if abs(float(current_score) - float(previous_score)) <= 1e-12:
                    continue
                candidate = dict(current)
                candidate.update(
                    {
                        "previous_score": float(previous_score),
                        "current_score": float(current_score),
                        "score_change": float(current_score) - float(previous_score),
                    }
                )
                skill_candidates.append(candidate)
            max_skill_candidates = max(1, int(len(examples) * 0.10))
            skill_candidates.sort(
                key=lambda record: abs(float(record.get("score_change") or 0.0)),
                reverse=True,
            )
            skill_candidates = skill_candidates[:max_skill_candidates]

        skill_update_path = epoch_dir / "skill_update.json"
        if skill_update_path.exists() and args.resume:
            skill_update = checkpoint.read_json(skill_update_path)
            runner.set_global_skill(str(skill_update.get("skill") or initial_skill))
            final_skill_record = dict(
                skill_update.get("record") or final_skill_record or {}
            )
        elif skill_candidates and skill_updater is not None:
            usage_tracker.set_context(
                run_id=run_id,
                epoch_id=epoch_id,
                phase="epoch_skill_update",
            )
            skill_update = await skill_updater.update(skill_candidates, epoch_id=epoch_id)
            checkpoint.write_json(skill_update_path, skill_update)
            runner.set_global_skill(str(skill_update.get("skill") or initial_skill))
            final_skill_record = dict(
                skill_update.get("record") or final_skill_record or {}
            )
        else:
            skill_update = {
                "skill": runner.global_skill,
                "version": (final_skill_record or {}).get("version"),
                "selected_count": 0,
                "batch_count": 0,
                "suggestions": [],
                "mode": (
                    "dynamic_no_update"
                    if dynamic_skill_update_enabled
                    else "static"
                ),
            }
            checkpoint.write_json(skill_update_path, skill_update)

        validation_metrics: dict[str, Any] = {"num_evaluated": 0}
        validation_result = None
        if validation_examples:
            validation_result = await epoch_runner.run_epoch(
                validation_examples,
                epoch_id=epoch_id,
                phase="validation",
                start_batch_id=0,
            )
            validation_dir = epoch_dir / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            checkpoint.write_records(
                validation_dir / "predictions.jsonl", validation_result.predictions
            )
            checkpoint.write_records(
                validation_dir / "evaluations.jsonl", validation_result.evaluations
            )
            checkpoint.write_records(
                validation_dir / "evaluation_failures.jsonl",
                collect_evaluation_failures(
                    validation_result.evaluations,
                    phase="validation",
                ),
            )
            checkpoint.write_records(
                validation_dir / "learning_records.jsonl",
                validation_result.learning_records,
            )
            checkpoint.write_json(validation_dir / "metrics.json", validation_result.metrics)
            validation_metrics = dict(validation_result.metrics)

        memory_statistics = (
            await store.epoch_statistics(epoch_id)
            if hasattr(store, "epoch_statistics")
            else {}
        )
        memory_snapshot = epoch_dir / "memory.sqlite"
        if hasattr(store, "backup_to"):
            store.backup_to(memory_snapshot)
        skill_record = dict(skill_update.get("record") or final_skill_record or {})
        binding = {
            "epoch_id": epoch_id,
            "memory_snapshot": str(memory_snapshot),
            "skill_version": skill_record.get("version") or skill_update.get("version"),
            "skill_path": str(
                skill_store.root / str(skill_record.get("path") or "")
            ),
            "validation_score": validation_metrics.get("overall_score"),
        }
        checkpoint.write_json(epoch_dir / "checkpoint.json", binding)
        validation_score = validation_metrics.get("overall_score")

        final_validation_score = (
            float(validation_score)
            if isinstance(validation_score, int | float)
            else None
        )
        final_epoch_id = epoch_id
        final_memory_snapshot = run_dir / "final_memory.sqlite"
        snapshot_sqlite(memory_snapshot, final_memory_snapshot)
        final_skill_record = skill_record
        checkpoint.write_json(
            run_dir / "final_checkpoint.json",
            {
                **binding,
                "memory_snapshot": str(final_memory_snapshot),
                **(
                    {"pre_review_memory_snapshot": str(final_pre_review_snapshot)}
                    if final_pre_review_snapshot is not None
                    else {}
                ),
                "validation_score": final_validation_score,
                "selection_policy": "last_epoch",
            },
        )

        metrics["skill_selected_count"] = len(skill_candidates)
        metrics["skill_version"] = skill_update.get("version")
        metrics["dynamic_skill_update_enabled"] = dynamic_skill_update_enabled
        metrics["validation_score"] = validation_score
        metrics["validation_num_evaluated"] = validation_metrics.get("num_evaluated", 0)
        metrics["memory_statistics"] = memory_statistics
        metrics["conflict_resolution"] = {
            "num_groups": int(conflict_resolution.get("num_groups") or 0),
            "num_committed": int(conflict_resolution.get("num_committed") or 0),
            "num_errors": int(conflict_resolution.get("num_errors") or 0),
        }
        metrics["same_source_excluded_total"] = sum(
            int(record.get("same_source_excluded_count") or 0)
            for record in accumulated["learning_records"]
        )
        metrics["same_source_excluded_episodes"] = sum(
            1
            for record in accumulated["learning_records"]
            if int(record.get("same_source_excluded_count") or 0) > 0
        )
        previous_train_records = current_train_records
        checkpoint.write_json(epoch_dir / "metrics.json", metrics)
        epoch_summaries.append(
            {
                **metrics,
                "artifact_dir": str(epoch_dir),
                "row_order": [example.row_index for example in ordered_examples],
            }
        )
        checkpoint.mark_completed(
            f"epoch:{epoch_id}:complete",
            epoch_id=epoch_id,
            phase="train_complete",
        )
        print(json.dumps(epoch_summaries[-1], ensure_ascii=False))

    if fork_final_review:

        review_path = run_dir / f"epoch_{num_epochs - 1:03d}" / "summary_review.json"
        if args.resume and review_path.exists():
            summary_review = checkpoint.read_json(review_path)
        else:
            summary_review = await summary_reviewer.review_epoch(
                run_dir,
                num_epochs - 1,
                run_id=run_id,
            )
            checkpoint.write_json(review_path, summary_review)
        final_epoch_dir = run_dir / f"epoch_{num_epochs - 1:03d}"
        final_metrics_path = final_epoch_dir / "metrics.json"
        if final_metrics_path.exists():
            final_metrics = checkpoint.read_json(final_metrics_path)
            final_metrics["summary_review"] = summary_review
            checkpoint.write_json(final_metrics_path, final_metrics)
        if final_memory_snapshot is None:
            final_memory_snapshot = run_dir / "final_memory.sqlite"
        store = getattr(runner.providers.get("memory"), "store", None)
        if store is not None and hasattr(store, "backup_to"):
            store.backup_to(final_memory_snapshot)
        else:
            raise RuntimeError("Forked final review requires an Epoch SQLite store")
        final_binding_path = run_dir / "final_checkpoint.json"
        final_binding = checkpoint.read_json(final_binding_path)
        final_binding.update(
            {
                "memory_snapshot": str(final_memory_snapshot),
                "pre_review_memory_snapshot": str(final_pre_review_snapshot),
            }
        )
        checkpoint.write_json(final_binding_path, final_binding)

    async def run_frozen_test(
        *,
        binding: dict[str, Any],
        test_dir: Path,
        completion_marker: str,
        database_suffix: str,
    ) -> dict[str, Any] | None:
        """Evaluate one frozen Epoch snapshot without mutating training Memory."""

        metrics_path = test_dir / "metrics.json"
        if completion_marker in checkpoint.completed_units and metrics_path.exists():
            return checkpoint.read_json(metrics_path)

        source_snapshot = Path(str(binding.get("memory_snapshot") or ""))
        if not source_snapshot.is_absolute():
            source_snapshot = ROOT / source_snapshot
        if not source_snapshot.is_file():
            raise FileNotFoundError(source_snapshot)

        test_dir.mkdir(parents=True, exist_ok=True)
        frozen_database_name = sanitize_path_part(
            f"{memory_db_name}__{database_suffix}"
        )
        frozen_config = dict(config)
        frozen_config["memory_db_name"] = frozen_database_name
        frozen_config["memory_store_mode"] = "read_only"
        frozen_config["run_phase"] = "test"
        frozen_config["log_dir"] = str(test_dir / "trajectories")
        frozen_memory_path = sqlite_memory_store_path(
            frozen_config,
            method=method_name,
            scripted=scripted,
        )
        if not frozen_memory_path.is_absolute():
            frozen_memory_path = ROOT / frozen_memory_path
        frozen_memory_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_sqlite(source_snapshot, frozen_memory_path)

        frozen_runner = build_runner(
            frozen_config,
            scripted_responses=args.response,
            usage_tracker=usage_tracker,
        )
        skill_path = Path(str(binding.get("skill_path") or ""))
        if not skill_path.is_absolute():
            skill_path = ROOT / skill_path
        frozen_runner.set_global_skill(
            skill_path.read_text(encoding="utf-8")
            if skill_path.is_file()
            else initial_skill
        )
        frozen_epoch_runner = EpochRunner(
            frozen_runner,
            evaluator,
            memory_writer,
            batch_size=(args.test_batch_size or batch_size),
            checkpoint_run_id=run_id,
            parallel_episodes=bool(epoch_config.get("parallel_episodes", True)),
        )
        test_result = await frozen_epoch_runner.run_epoch(
            test_examples,
            epoch_id=int(binding.get("epoch_id", 0)),
            phase="test",
        )
        checkpoint.write_records(test_dir / "predictions.jsonl", test_result.predictions)
        checkpoint.write_records(test_dir / "evaluations.jsonl", test_result.evaluations)
        checkpoint.write_records(
            test_dir / "evaluation_failures.jsonl",
            collect_evaluation_failures(test_result.evaluations, phase="test"),
        )
        checkpoint.write_records(
            test_dir / "learning_records.jsonl", test_result.learning_records
        )
        checkpoint.write_json(metrics_path, test_result.metrics)
        checkpoint.mark_completed(
            completion_marker,
            epoch_id=int(binding.get("epoch_id", 0)),
            phase="test_complete",
        )
        return dict(test_result.metrics)

    test_metrics: dict[str, Any] | None = None
    if test_each_epoch and test_examples:

        for epoch_id in range(num_epochs):
            epoch_dir = run_dir / f"epoch_{epoch_id:03d}"
            binding_path = epoch_dir / "checkpoint.json"
            if not binding_path.exists():
                continue
            epoch_metrics = await run_frozen_test(
                binding=checkpoint.read_json(binding_path),
                test_dir=epoch_dir / "test",
                completion_marker=f"test:epoch:{epoch_id}:complete",
                database_suffix=f"epoch_{epoch_id}_test",
            )
            if epoch_id == num_epochs - 1:
                test_metrics = epoch_metrics
            if epoch_metrics is not None:
                epoch_metrics_path = epoch_dir / "metrics.json"
                saved_epoch_metrics = checkpoint.read_json(epoch_metrics_path)
                saved_epoch_metrics["test_metrics"] = epoch_metrics
                checkpoint.write_json(epoch_metrics_path, saved_epoch_metrics)
                if epoch_id < len(epoch_summaries):
                    epoch_summaries[epoch_id]["test_metrics"] = epoch_metrics

        if test_metrics is not None:
            source_test_dir = run_dir / f"epoch_{num_epochs - 1:03d}" / "test"
            final_test_dir = run_dir / "test"
            final_test_dir.mkdir(parents=True, exist_ok=True)
            for filename in (
                "predictions.jsonl",
                "evaluations.jsonl",
                "evaluation_failures.jsonl",
                "learning_records.jsonl",
            ):
                source_file = source_test_dir / filename
                if source_file.exists():
                    checkpoint.write_records(
                        final_test_dir / filename,
                        checkpoint.read_records(source_file),
                    )
            checkpoint.write_json(final_test_dir / "metrics.json", test_metrics)
            checkpoint.mark_completed(
                "test:complete",
                epoch_id=num_epochs - 1,
                phase="test_complete",
            )

    fork_test_metrics: dict[str, Any] | None = None
    fork_test_metrics_before: dict[str, Any] | None = None
    if (
        (not test_each_epoch)
        and (not fork_final_review)
        and test_examples
        and final_memory_snapshot is not None
        and final_memory_snapshot.exists()
    ):
        test_dir = run_dir / "test"
        if args.resume and "test:complete" in checkpoint.completed_units:

            test_metrics_path = test_dir / "metrics.json"
            if test_metrics_path.exists():
                test_metrics = checkpoint.read_json(test_metrics_path)
        if test_metrics is None:

            final_database_name = sanitize_path_part(f"{memory_db_name}__final")
            frozen_config = dict(config)
            frozen_config["memory_db_name"] = final_database_name
            frozen_config["memory_store_mode"] = "read_only"
            frozen_config["run_phase"] = "test"
            frozen_config["log_dir"] = str(test_dir / "trajectories")
            frozen_memory_path = sqlite_memory_store_path(
                frozen_config,
                method=method_name,
                scripted=scripted,
            )
            if not frozen_memory_path.is_absolute():
                frozen_memory_path = ROOT / frozen_memory_path
            frozen_memory_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_sqlite(final_memory_snapshot, frozen_memory_path)

            frozen_runner = build_runner(
                frozen_config,
                scripted_responses=args.response,
                usage_tracker=usage_tracker,
            )
            final_binding = checkpoint.read_json(run_dir / "final_checkpoint.json")
            final_skill_path = Path(str(final_binding.get("skill_path") or ""))
            final_skill = (
                final_skill_path.read_text(encoding="utf-8")
                if final_skill_path.is_file()
                else initial_skill
            )
            frozen_runner.set_global_skill(final_skill)
            frozen_epoch_runner = EpochRunner(
                frozen_runner,
                evaluator,
                memory_writer,

                batch_size=(args.test_batch_size or batch_size),
                checkpoint_run_id=run_id,
                parallel_episodes=bool(epoch_config.get("parallel_episodes", True)),
            )
            test_result = await frozen_epoch_runner.run_epoch(
                test_examples,
                epoch_id=int(final_binding.get("epoch_id", final_epoch_id or 0)),
                phase="test",
            )
            test_dir.mkdir(parents=True, exist_ok=True)
            checkpoint.write_records(test_dir / "predictions.jsonl", test_result.predictions)
            checkpoint.write_records(test_dir / "evaluations.jsonl", test_result.evaluations)
            checkpoint.write_records(
                test_dir / "evaluation_failures.jsonl",
                collect_evaluation_failures(
                    test_result.evaluations,
                    phase="test",
                ),
            )
            checkpoint.write_records(
                test_dir / "learning_records.jsonl", test_result.learning_records
            )
            checkpoint.write_json(test_dir / "metrics.json", test_result.metrics)
            test_metrics = dict(test_result.metrics)
            checkpoint.mark_completed("test:complete", phase="test_complete")

    if (
        fork_final_review
        and test_examples
        and final_memory_snapshot is not None
        and final_memory_snapshot.exists()
        and final_pre_review_snapshot is not None
        and final_pre_review_snapshot.exists()
    ):
        final_binding = checkpoint.read_json(run_dir / "final_checkpoint.json")
        before_binding = {
            **final_binding,
            "memory_snapshot": str(final_pre_review_snapshot),
        }
        fork_test_metrics_before = await run_frozen_test(
            binding=before_binding,
            test_dir=run_dir / "test_before_review",
            completion_marker="test:before_review:complete",
            database_suffix="test_before_review",
        )
        fork_test_metrics = await run_frozen_test(
            binding=final_binding,
            test_dir=run_dir / "test_after_review",
            completion_marker="test:after_review:complete",
            database_suffix="test_after_review",
        )
        test_metrics = fork_test_metrics
        checkpoint.write_json(
            run_dir / "fork_final_review.json",
            {
                "enabled": True,
                "num_epochs": num_epochs,
                "before_review_memory": str(final_pre_review_snapshot),
                "after_review_memory": str(final_memory_snapshot),
                "before_review_test_dir": str(run_dir / "test_before_review"),
                "after_review_test_dir": str(run_dir / "test_after_review"),
                "before_review_metrics": fork_test_metrics_before,
                "after_review_metrics": fork_test_metrics,
            },
        )

    memory_path = sqlite_memory_store_path(
        config,
        method=method_name,
        scripted=scripted,
    )
    if not memory_path.is_absolute():
        memory_path = ROOT / memory_path
    summary = {
        "run_id": run_id,
        "method": method_name,
        "model": model_name,
        "dataset_path": args.dataset,
        "dataset_type": args.dataset_type,
        "dataset_config_path": (
            str(dataset_config_path) if dataset_config_path.exists() else None
        ),
        "start_index": start_index,
        "end_index": end_index,
        "num_examples": len(examples),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "memory_db_name": memory_db_name,
        "memory_db_action": memory_db_action,
        "shuffle_each_epoch": shuffle_each_epoch,
        "seed": seed,
        "memory_path": str(memory_path),
        "epochs": epoch_summaries,
        "checkpoint_selection": "last_epoch",
        "final_epoch_id": final_epoch_id,
        "test_each_epoch": test_each_epoch,
        "fork_final_review": fork_final_review,
        "fork_final_review_path": (
            str(run_dir / "fork_final_review.json") if fork_final_review else None
        ),
        "test_metrics_before_review": fork_test_metrics_before,
        "test_metrics_after_review": fork_test_metrics,
        "final_validation_score": final_validation_score,
        "final_checkpoint_path": str(run_dir / "final_checkpoint.json"),
        "test_metrics": test_metrics,
        "usage_path": str(usage_path),
        "llm_usage": usage_tracker.summary(),
    }
    checkpoint.write_records(usage_path, usage_tracker.records)
    checkpoint.write_json(run_dir / "summary.json", summary)
    checkpoint.mark_run_complete(**summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
