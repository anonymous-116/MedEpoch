from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from tqdm import tqdm

from amf.datasets import DatasetExample
from amf.evaluators.base import (
    BaseEvaluator,
    EvaluationResult,
    aggregate_evaluation_records,
    collect_evaluation_failures,
    invalid_evaluation_result,
)
from amf.evaluators.registry import evaluate_dataset_answer
from amf.memory import (
    EpochMemoryUtilityJudge,
    EpochMemoryWriter,
    LLMMemoryWriter,
    MemoryWriteAction,
)
from amf.memory.provenance import source_id_for_case
from amf.providers.base import ProviderRequest
from amf.runner import EpisodeResult, Runner

@dataclass(slots=True)
class EpisodeLearningRecord:
    """Compact, serializable post-answer signal used at a batch boundary."""

    epoch_id: int
    batch_id: int
    batch_position: int
    row_index: int
    case_id: str | None
    source_id: str
    episode_id: str
    status: str
    question: str
    final_answer: str
    score: float | None
    previous_score: float | None
    score_change: float | None
    metrics: dict[str, float | int | None]
    positive_rubrics: list[dict[str, Any]] = field(default_factory=list)
    negative_rubrics: list[dict[str, Any]] = field(default_factory=list)
    observed_criteria: list[str] = field(default_factory=list)
    missed_positive_criteria: list[str] = field(default_factory=list)
    triggered_negative_criteria: list[str] = field(default_factory=list)
    tool_queries: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memory_ids: list[str] = field(default_factory=list)
    same_source_excluded_count: int = 0

    evaluation_feedback: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def writer_feedback(self) -> dict[str, Any]:
        """Return only the post-answer evidence the memory writer may use."""

        payload = {
            "score": self.score,
            "previous_score": self.previous_score,
            "score_change": self.score_change,
            "metrics": self.metrics,
            "summary": self._summary(),

            "rubrics": {
                "positive": self.positive_rubrics,
                "negative": self.negative_rubrics,
            },
            "rubric_feedback": {
                "observed_criteria": self.observed_criteria,
                "missed_positive_criteria": self.missed_positive_criteria,
                "triggered_negative_criteria": self.triggered_negative_criteria,
            },
            "tool_queries": self.tool_queries,
            "retrieved_memory_ids": self.retrieved_memory_ids,
            "same_source_excluded_count": self.same_source_excluded_count,
        }
        if self.evaluation_feedback is not None:
            payload["evaluation_feedback"] = dict(self.evaluation_feedback)
        return payload

    def _summary(self) -> str:
        parts = [
            f"epoch={self.epoch_id}",
            f"batch={self.batch_id}",
            f"score={self.score}",
        ]
        if self.missed_positive_criteria:
            parts.append(f"missed_positive={len(self.missed_positive_criteria)}")
        if self.triggered_negative_criteria:
            parts.append(f"triggered_negative={len(self.triggered_negative_criteria)}")
        return ", ".join(parts)

@dataclass(slots=True)
class EpochResult:
    epoch_id: int
    phase: str
    predictions: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]
    learning_records: list[dict[str, Any]]
    memory_writes: list[dict[str, Any]]
    metrics: dict[str, Any]

class EpochRunner:
    """Run one train, validation, or test epoch with deterministic commits.

    Episodes in a batch solve against the same pre-commit repository snapshot.
    A multi-example training batch is an evidence window: candidates are made
    independently, resolved jointly, and committed in one transaction.
    Validation and Test execute the same evaluation path but never call the writer.
    """

    def __init__(
        self,
        runner: Runner,
        evaluator: BaseEvaluator,
        memory_writer: LLMMemoryWriter,
        *,
        batch_size: int = 1,
        max_new_memories: int | None = None,
        checkpoint_run_id: str | None = None,
        memory_utility_judge: EpochMemoryUtilityJudge | None = None,
        writer_top_k: int = 3,
        writer_similarity_threshold: float = 0.71,
        parallel_episodes: bool = True,
    ) -> None:
        if not runner.providers.has("memory"):
            raise ValueError("EpochRunner requires a registered memory provider.")
        self.runner = runner
        self.evaluator = evaluator
        self.memory_writer = memory_writer
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.max_new_memories = max_new_memories
        self.checkpoint_run_id = checkpoint_run_id
        self.memory_utility_judge = memory_utility_judge
        self.writer_top_k = max(1, writer_top_k)
        self.writer_similarity_threshold = float(writer_similarity_threshold)
        self.parallel_episodes = parallel_episodes

    async def run_epoch(
        self,
        examples: Sequence[DatasetExample],
        *,
        epoch_id: int,
        phase: str = "train",
        previous_scores: dict[str, float] | None = None,
        start_batch_id: int = 0,
        initial_num_written: int = 0,
        on_batch_complete: Callable[[int, dict[str, list[dict[str, Any]]]], Awaitable[None]]
        | None = None,
        batches: Sequence[Sequence[DatasetExample]] | None = None,
    ) -> EpochResult:
        if phase not in {"train", "validation", "test"}:
            raise ValueError("phase must be train, validation, or test")
        allow_memory_writes = phase == "train"
        previous_scores = dict(previous_scores or {})
        predictions: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        learning_records: list[EpisodeLearningRecord] = []
        memory_writes: list[dict[str, Any]] = []
        num_written = initial_num_written

        if batches is None:
            batch_specs = [
                (batch_id, batch_start, list(examples[batch_start : batch_start + self.batch_size]))
                for batch_id, batch_start in enumerate(
                    range(0, len(examples), self.batch_size)
                )
            ]
        else:
            batch_specs = []
            batch_start = 0
            for batch_id, raw_batch in enumerate(batches):
                batch = list(raw_batch)
                if not batch:
                    raise ValueError("evidence windows must not be empty")
                if len(batch) > self.batch_size:
                    raise ValueError("an evidence window exceeds batch_size")
                batch_specs.append((batch_id, batch_start, batch))
                batch_start += len(batch)
            if batch_start != len(examples):
                raise ValueError("evidence windows must cover every example exactly once")
        for batch_id, batch_start, batch in tqdm(
            batch_specs[start_batch_id:],
            desc=f"epoch {epoch_id} batches",
            unit="batch",
            dynamic_ncols=True,
        ):
            prediction_start = len(predictions)
            evaluation_start = len(evaluations)
            learning_start = len(learning_records)
            write_start = len(memory_writes)
            completed: list[
                tuple[DatasetExample, EpisodeResult, dict[str, Any], int]
            ] = []

            async def solve_one(
                batch_position: int,
                example: DatasetExample,
            ) -> tuple[DatasetExample, EpisodeResult, dict[str, Any], int]:
                epoch_position = batch_start + batch_position
                episode_runner = (
                    self.runner.fork_for_episode()
                    if self.parallel_episodes and len(batch) > 1
                    else self.runner
                )
                episode_runner.usage_tracker.set_context(
                    run_id=self.checkpoint_run_id,
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                    row_index=example.row_index,
                    phase=f"{phase}_solve",
                )
                result = await episode_runner.run_episode(
                    example.case,
                    hidden_context=example.evaluation if phase == "train" else None,
                    runtime_metadata={

                        "epoch_id": epoch_id,
                        "epoch_position": epoch_position,
                        "batch_id": batch_id,
                        "batch_position": batch_position,
                        "batch_size": self.batch_size,
                        "row_index": example.row_index,
                        "source_id": source_id_for_case(example.case),
                        "memory_update_phase": "batch_end",
                        "run_phase": phase,
                        "run_id": self.checkpoint_run_id,
                        "checkpoint_key": (
                            f"{self.checkpoint_run_id}:epoch:{epoch_id}:"
                            f"batch:{batch_id}:row:{example.row_index}"
                            if self.checkpoint_run_id
                            else None
                        ),
                    },
                )
                prediction = self._prediction(
                    example,
                    result,
                    epoch_id,
                    epoch_position,
                    batch_id,
                    batch_position,
                )

                return example, result, prediction, batch_position

            solve_coroutines = [
                solve_one(batch_position, example)
                for batch_position, example in enumerate(batch)
            ]
            if self.parallel_episodes and len(solve_coroutines) > 1:
                completed = list(await asyncio.gather(*solve_coroutines))
            else:
                completed = [await coroutine for coroutine in solve_coroutines]
            predictions.extend(record[2] for record in completed)

            batch_records: list[tuple[EpisodeResult, EpisodeLearningRecord]] = []
            evaluation_by_episode: dict[str, dict[str, Any]] = {}
            reference_by_episode: dict[str, dict[str, Any]] = {}
            async def evaluate_one(
                example: DatasetExample,
                result: EpisodeResult,
                prediction: dict[str, Any],
                batch_position: int,
            ) -> tuple[DatasetExample, EpisodeResult, dict[str, Any], int, EvaluationResult]:
                result_runner = self.runner
                result_runner.usage_tracker.set_context(
                    run_id=self.checkpoint_run_id,
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                    row_index=example.row_index,
                    phase=f"{phase}_evaluate",
                )

                result_runner.trace_action(
                    result.state,
                    "evaluate.start",
                    phase=phase,
                )

                episode_evaluator = self.evaluator.fork()
                try:
                    evaluation = await evaluate_dataset_answer(
                        episode_evaluator,
                        example,
                        result.final_answer or "",
                        answer_extraction=result.state.metadata.get(
                            "answer_extraction"
                        ),
                    )
                except Exception as exc:

                    evaluation = invalid_evaluation_result(
                        f"{type(exc).__name__}: {exc}"
                    )
                evaluation_status = str(
                    evaluation.details.get("evaluation_status")
                    or ("OK" if evaluation.score is not None else "INVALID")
                )
                result_runner.trace_action(
                    result.state,
                    "evaluate.done",
                    phase=phase,
                    score=evaluation.score,
                    status=evaluation_status,
                )
                return example, result, prediction, batch_position, evaluation

            evaluation_coroutines = [
                evaluate_one(example, result, prediction, batch_position)
                for example, result, prediction, batch_position in completed
            ]
            if self.parallel_episodes and len(evaluation_coroutines) > 1:
                evaluated = list(await asyncio.gather(*evaluation_coroutines))
            else:
                evaluated = [await coroutine for coroutine in evaluation_coroutines]

            for example, result, prediction, batch_position, evaluation in evaluated:
                evaluations.append({**prediction, "evaluation": evaluation.to_dict()})
                record = self._learning_record(
                    example,
                    result,
                    evaluation,
                    epoch_id,
                    batch_id,
                    batch_position,
                    previous_score=previous_scores.get(
                        source_id_for_case(example.case)
                        or str(example.case.get("case_id") or example.row_index)
                    ),
                )
                result.state.metadata["previous_epoch_score"] = record.previous_score
                result.state.metadata["score_change"] = record.score_change
                learning_records.append(record)
                batch_records.append((result, record))
                evaluation_by_episode[result.episode_id] = evaluation.to_dict()
                reference_by_episode[result.episode_id] = dict(example.evaluation)

            if self.memory_utility_judge is not None and phase == "train":
                async def judge_utility(
                    result: EpisodeResult,
                    record: EpisodeLearningRecord,
                ) -> tuple[EpisodeResult, list[Any]]:
                    self.runner.usage_tracker.set_context(
                        run_id=self.checkpoint_run_id,
                        epoch_id=epoch_id,
                        batch_id=batch_id,
                        row_index=record.row_index,
                        phase="memory_utility_judge",
                    )
                    episode_utility_judge = self.memory_utility_judge.fork()
                    utility_decisions = await episode_utility_judge.judge(
                        result.state,
                        evaluation=evaluation_by_episode[result.episode_id],
                        reference=reference_by_episode[result.episode_id],
                    )
                    return result, utility_decisions

                utility_coroutines = [
                    judge_utility(result, record) for result, record in batch_records
                ]
                if self.parallel_episodes and len(utility_coroutines) > 1:
                    utility_results = list(await asyncio.gather(*utility_coroutines))
                else:
                    utility_results = [await coroutine for coroutine in utility_coroutines]
                store = getattr(self.runner.providers.get("memory"), "store", None)
                for result, utility_decisions in utility_results:
                    result.state.metadata["memory_utility"] = [
                        decision.to_dict() for decision in utility_decisions
                    ]
                    record = next(
                        record
                        for candidate_result, record in batch_records
                        if candidate_result is result
                    )
                    if hasattr(store, "record_use"):
                        for decision in utility_decisions:
                            await store.record_use(
                                decision.version_id,
                                source_id_for_case(result.state.case) or result.episode_id,
                                decision.label,
                                evidence=decision.evidence,
                                epoch_id=epoch_id,
                                batch_id=batch_id,
                                score_before=record.previous_score,
                                score_after=record.score,
                            )

            if not allow_memory_writes:
                for result, record in batch_records:
                    memory_writes.append(
                        {
                            "episode_id": result.episode_id,
                            "row_index": record.row_index,
                            "batch_id": batch_id,
                            "action": "read_only",
                            "reason": f"{phase} phase does not write memory",
                        }
                    )
                if on_batch_complete is not None:
                    await on_batch_complete(
                        batch_id,
                        {
                            "predictions": predictions[prediction_start:],
                            "evaluations": evaluations[evaluation_start:],
                            "learning_records": [
                                record.to_dict()
                                for record in learning_records[learning_start:]
                            ],
                            "memory_writes": memory_writes[write_start:],
                        },
                    )
                continue

            if not isinstance(self.memory_writer, EpochMemoryWriter):
                raise TypeError(
                    "EpochRunner requires EpochMemoryWriter for the two-stage "
                    "candidate and relation workflow."
                )
            if len(batch_records) == 1:

                write_records, newly_added = await self._write_single_episode_memory(
                    batch_records[0][0],
                    batch_records[0][1],
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                    num_written=num_written,
                )
            else:
                write_records, newly_added = await self._write_evidence_window(
                    batch_records,
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                    num_written=num_written,
                )
            memory_writes.extend(write_records)
            num_written += newly_added

            if on_batch_complete is not None:

                await on_batch_complete(
                    batch_id,
                    {
                        "predictions": predictions[prediction_start:],
                        "evaluations": evaluations[evaluation_start:],
                        "learning_records": [
                            record.to_dict()
                            for record in learning_records[learning_start:]
                        ],
                        "memory_writes": memory_writes[write_start:],
                    },
                )

        await self.runner.flush()
        metrics = self._aggregate_metrics(evaluations)
        metrics.update(
            {
                "epoch_id": epoch_id,
                "phase": phase,
                "batch_size": self.batch_size,
                "num_batches": len(batch_specs),
                "num_predictions": len(predictions),
                "num_memory_writes": num_written,
            }
        )
        return EpochResult(
            epoch_id=epoch_id,
            phase=phase,
            predictions=predictions,
            evaluations=evaluations,
            learning_records=[record.to_dict() for record in learning_records],
            memory_writes=memory_writes,
            metrics=metrics,
        )

    async def _write_single_episode_memory(
        self,
        result: EpisodeResult,
        record: EpisodeLearningRecord,
        *,
        epoch_id: int,
        batch_id: int,
        num_written: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Preserve the pre-window batch_size=1 writer path exactly."""

        if record.score is None:
            return [
                {
                    "episode_id": result.episode_id,
                    "row_index": record.row_index,
                    "batch_id": batch_id,
                    "action": "ignore",
                    "reason": "evaluation did not produce a valid score",
                }
            ], 0
        writer = self.memory_writer
        assert isinstance(writer, EpochMemoryWriter)
        self.runner.usage_tracker.set_context(
            run_id=self.checkpoint_run_id,
            epoch_id=epoch_id,
            batch_id=batch_id,
            row_index=record.row_index,
            phase="write_memory",
        )
        feedback = record.writer_feedback()
        answer_references = self._answer_memory_references(result.state)
        result.state.metadata["answer_memory_references"] = answer_references
        result.state.metadata["answer_memory_prompt_id_map"] = {
            f"Information {index}": str(reference["id"])
            for index, reference in enumerate(answer_references, 1)
            if reference.get("id")
        }
        candidate, candidate_raw, candidate_error = await writer.generate_candidate(
            result.state,
            feedback,
            information=answer_references,
        )
        if candidate is not None:
            self._attach_information_usage(result.state, candidate)

        candidate_query = ""
        references: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []
        if candidate is None:
            decision = writer.invalid_candidate_decision(candidate_raw, candidate_error)
        else:
            summary = str(candidate.get("summary") or "").strip()
            candidate_query = "\n".join(
                value
                for value in (
                    summary.split(".", 1)[0].strip(),
                    summary,
                    str(candidate.get("applicable_conditions") or "").strip(),
                )
                if value
            )
            response = await self.runner.providers.call(
                ProviderRequest(
                    provider="memory",
                    operation="retrieve",
                    query=candidate_query,
                    payload={
                        "top_k": self.writer_top_k,
                        "filters": {
                            "method": "epoch",
                            "_bge_similarity_threshold": self.writer_similarity_threshold,
                        },
                    },
                )
            )
            if not response.ok:
                decision = writer.invalid_candidate_decision(
                    "",
                    "writer retrieval failed: " + str(response.error or "unknown error"),
                )
            else:
                writer_references = [
                    self._writer_memory_reference(item)
                    for item in (
                        response.data if isinstance(response.data, list) else []
                    )
                    if isinstance(item, dict)
                ]
                references = self._merge_memory_references(
                    answer_references,
                    writer_references,
                )
                source_id = source_id_for_case(result.state.case) or result.episode_id
                history, other = self._classify_memory_references(
                    references,
                    source_id,
                )
                result.state.metadata["writer_memory_references"] = references
                result.state.metadata["writer_history_references"] = history
                result.state.metadata["writer_other_references"] = other
                result.state.metadata["writer_summary_prompt_id_map"] = {
                    f"Summary {index}": str(reference["id"])
                    for index, reference in enumerate([*history, *other], 1)
                    if reference.get("id")
                }
                decision = await writer.decide_candidate(
                    result.state,
                    feedback,
                    candidate_memory=candidate,
                    candidate_raw_output=candidate_raw,
                    candidate_memories=references,
                )

        write_record: dict[str, Any] = {
            "episode_id": result.episode_id,
            "row_index": record.row_index,
            "batch_id": batch_id,
            "candidate_query": candidate_query,
            "writer_retrieval": {
                "similarity_threshold": self.writer_similarity_threshold,
                "top_k": self.writer_top_k,
                "results": references,
                "answer_results": answer_references,
                "history_results": history,
                "other_results": other,
            },
            "information_usage": result.state.metadata.get("memory_information_usage"),
            "decision": decision.to_dict(),
            "written_items": [],
        }
        operations = list(decision.actions)
        if not operations and decision.action not in {"ignore", "invalid"}:
            operations = [
                MemoryWriteAction(
                    action=decision.action,
                    items=list(decision.items),
                    target_ids=list(decision.target_ids),
                    model_target_ids=list(decision.model_target_ids),
                    reason=str(decision.reason or ""),
                )
            ]
        source_id = source_id_for_case(result.state.case) or result.episode_id
        action_payloads: list[dict[str, Any]] = []
        pending_adds = 0
        item_counter = 0
        for operation_index, operation in enumerate(operations):
            provenance = {
                "epoch_id": epoch_id,
                "batch_id": batch_id,
                "source_id": source_id,
                "content_source_ids": [source_id],
                "source_episode_ids": [result.episode_id],
                "source_row_indices": [record.row_index],
                "source_score": record.score,
                "retrieved_memory_ids": record.retrieved_memory_ids,
                "created_by": "episode_memory_writer",
            }
            for item in operation.items:
                item.metadata.update(provenance)
                if self.checkpoint_run_id:
                    item.metadata.update(
                        {
                            "checkpoint_run_id": self.checkpoint_run_id,
                            "checkpoint_batch_unit": f"epoch:{epoch_id}:batch:{batch_id}",
                            "checkpoint_key": (
                                f"{self.checkpoint_run_id}:epoch:{epoch_id}:batch:{batch_id}:"
                                f"row:{record.row_index}:memory:{item_counter}"
                            ),
                        }
                    )
                item_counter += 1
            if (
                self.max_new_memories is not None
                and operation.action == "add"
                and num_written + pending_adds >= self.max_new_memories
            ):
                continue
            if operation.action == "add":
                pending_adds += len(operation.items)
            action_payloads.append(
                {
                    "action": operation.action,
                    "target_ids": list(operation.target_ids),
                    "reason": operation.reason,
                    "checkpoint_key": (
                        f"{self.checkpoint_run_id}:epoch:{epoch_id}:batch:{batch_id}:"
                        f"row:{record.row_index}:action:{operation_index}"
                        if self.checkpoint_run_id
                        else None
                    ),
                    "items": [item.to_dict() for item in operation.items],
                }
            )
        if not action_payloads:
            write_record["action"] = "ignore"
            write_record["reason"] = (
                "epoch max_new_memories limit reached"
                if self.max_new_memories is not None
                else decision.reason or "no valid memory actions"
            )
            if decision.error:
                write_record["error"] = decision.error
            return [write_record], 0

        response = await self.runner.providers.call(
            ProviderRequest(
                provider="memory",
                operation="apply_actions",
                payload={
                    "actions": action_payloads,
                    "method": "epoch",
                    "source_id": source_id,
                    "epoch_id": epoch_id,
                    "batch_id": batch_id,
                    "candidate_summary": decision.candidate_memory,
                    "information_usage": result.state.metadata.get(
                        "memory_information_usage"
                    ),
                },
            )
        )
        write_record["action"] = "committed" if response.ok else "error"
        write_record["response"] = response.to_dict()
        if not response.ok:
            write_record["error"] = response.error or "transactional memory action failed"
            raise RuntimeError(write_record["error"])
        added = 0
        if isinstance(response.data, dict):
            committed_actions = list(response.data.get("actions") or [])
            write_record["written_items"] = committed_actions
            for committed in committed_actions:
                if isinstance(committed, dict) and committed.get("action") == "add":
                    added += len(list(committed.get("created_version_ids") or []))
        return [write_record], added

    async def _write_evidence_window(
        self,
        batch_records: list[tuple[EpisodeResult, EpisodeLearningRecord]],
        *,
        epoch_id: int,
        batch_id: int,
        num_written: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Generate per-case candidates, then jointly commit one window."""

        writer = self.memory_writer
        assert isinstance(writer, EpochMemoryWriter)
        ignored: list[dict[str, Any]] = []
        valid_records = [
            (result, record)
            for result, record in batch_records
            if record.score is not None
        ]
        for result, record in batch_records:
            if record.score is None:
                ignored.append(
                    {
                        "episode_id": result.episode_id,
                        "row_index": record.row_index,
                        "batch_id": batch_id,
                        "action": "ignore",
                        "reason": "evaluation did not produce a valid score",
                    }
                )
        if not valid_records:
            return ignored, 0
        self.runner.trace_action(
            valid_records[0][0].state,
            "memory.window.candidates",
            batch_id=batch_id,
            size=len(valid_records),
        )

        async def generate_one(
            candidate_index: int,
            result: EpisodeResult,
            record: EpisodeLearningRecord,
        ) -> dict[str, Any]:
            candidate_id = chr(ord("A") + candidate_index)
            self.runner.usage_tracker.set_context(
                run_id=self.checkpoint_run_id,
                epoch_id=epoch_id,
                batch_id=batch_id,
                row_index=record.row_index,
                phase="write_memory_candidate",
            )
            feedback = record.writer_feedback()
            answer_references = self._answer_memory_references(result.state)
            result.state.metadata["answer_memory_references"] = answer_references
            result.state.metadata["answer_memory_prompt_id_map"] = {
                f"Information {index}": str(reference["id"])
                for index, reference in enumerate(answer_references, 1)
                if reference.get("id")
            }

            memory, raw, error = await writer.generate_candidate(
                result.state,
                feedback,
                information=answer_references,
            )
            if memory is not None:
                self._attach_information_usage(result.state, memory)
            return {
                "id": candidate_id,
                "state": result.state,
                "result": result,
                "record": record,
                "feedback": feedback,
                "question": record.question,
                "memory": memory,
                "raw_output": raw,
                "error": error,
                "answer_references": answer_references,
                "information_usage": result.state.metadata.get(
                    "memory_information_usage"
                ),
            }

        generation_calls = [
            generate_one(index, result, record)
            for index, (result, record) in enumerate(valid_records)
        ]
        if self.parallel_episodes and len(generation_calls) > 1:
            generated = list(await asyncio.gather(*generation_calls))
        else:
            generated = [await call for call in generation_calls]

        candidates: list[dict[str, Any]] = []
        for generated_candidate in generated:
            if generated_candidate["memory"] is None:
                ignored.append(
                    {
                        "episode_id": generated_candidate["result"].episode_id,
                        "row_index": generated_candidate["record"].row_index,
                        "batch_id": batch_id,
                        "candidate_id": generated_candidate["id"],
                        "action": "ignore",
                        "reason": "invalid candidate summary",
                        "candidate_raw_output": generated_candidate["raw_output"],
                        "error": generated_candidate["error"],
                    }
                )
            else:
                candidates.append(generated_candidate)
        if not candidates:
            return ignored, 0

        async def retrieve_for_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
            record = candidate["record"]
            memory = candidate["memory"]
            summary = str(memory.get("summary") or "").strip()
            name = summary.split(".", 1)[0].strip()
            query = "\n".join(
                value
                for value in (
                    name,
                    summary,
                    str(memory.get("applicable_conditions") or "").strip(),
                )
                if value
            )
            self.runner.usage_tracker.set_context(
                run_id=self.checkpoint_run_id,
                epoch_id=epoch_id,
                batch_id=batch_id,
                row_index=record.row_index,
                phase="write_memory_retrieve",
            )
            response = await self.runner.providers.call(
                ProviderRequest(
                    provider="memory",
                    operation="retrieve",
                    query=query,
                    payload={
                        "top_k": self.writer_top_k,
                        "filters": {
                            "method": "epoch",
                            "_bge_similarity_threshold": self.writer_similarity_threshold,
                        },
                    },
                )
            )
            return {"candidate": candidate, "query": query, "response": response}

        retrievals = list(
            await asyncio.gather(
                *(retrieve_for_candidate(candidate) for candidate in candidates)
            )
        )
        retrieval_errors = [
            retrieval for retrieval in retrievals if not retrieval["response"].ok
        ]
        if retrieval_errors:
            return [
                *ignored,
                {
                    "batch_id": batch_id,
                    "row_indices": [candidate["record"].row_index for candidate in candidates],
                    "candidate_ids": [candidate["id"] for candidate in candidates],
                    "action": "ignore",
                    "reason": "writer retrieval failed; no unconditional ADD was performed",
                    "errors": [
                        str(retrieval["response"].error or "unknown error")
                        for retrieval in retrieval_errors
                    ],
                },
            ], 0

        union: dict[str, dict[str, Any]] = {}
        union_order: list[str] = []
        retrieval_audit: list[dict[str, Any]] = []
        for retrieval in retrievals:
            candidate = retrieval["candidate"]
            candidate_id = candidate["id"]
            raw_data = retrieval["response"].data
            writer_references = [
                self._writer_memory_reference(item)
                for item in (raw_data if isinstance(raw_data, list) else [])
                if isinstance(item, dict)
            ]
            per_candidate = self._merge_memory_references(
                candidate["answer_references"],
                writer_references,
            )
            retrieval_audit.append(
                {
                    "candidate_id": candidate_id,
                    "row_index": candidate["record"].row_index,
                    "query": retrieval["query"],
                    "answer_results": candidate["answer_references"],
                    "writer_results": writer_references,
                    "union_results": per_candidate,
                }
            )
            for reference in per_candidate:
                real_id = str(reference.get("id") or "").strip()
                if not real_id:
                    continue
                if real_id not in union:
                    union[real_id] = dict(reference)
                    union[real_id]["retrieved_for_candidate_ids"] = []
                    union[real_id]["usage_assessments"] = {}
                    union_order.append(real_id)
                if candidate_id not in union[real_id]["retrieved_for_candidate_ids"]:
                    union[real_id]["retrieved_for_candidate_ids"].append(candidate_id)
                assessment = reference.get("usage_assessment")
                if assessment:
                    union[real_id]["usage_assessments"][candidate_id] = assessment
        old_summaries = [union[real_id] for real_id in union_order]

        self.runner.usage_tracker.set_context(
            run_id=self.checkpoint_run_id,
            epoch_id=epoch_id,
            batch_id=batch_id,
            phase="write_memory_window_action",
        )
        self.runner.trace_action(
            candidates[0]["state"],
            "memory.window.action.start",
            batch_id=batch_id,
            candidates=len(candidates),
            existing=len(old_summaries),
        )
        decision = await writer.decide_window(candidates, old_summaries)
        self.runner.trace_action(
            candidates[0]["state"],
            "memory.window.action.done",
            batch_id=batch_id,
            actions=",".join(action.action for action in decision.actions)
            or decision.action,
        )
        window_record: dict[str, Any] = {
            "batch_id": batch_id,
            "episode_ids": [candidate["result"].episode_id for candidate in candidates],
            "row_indices": [candidate["record"].row_index for candidate in candidates],
            "candidates": [
                {
                    "candidate_id": candidate["id"],
                    "episode_id": candidate["result"].episode_id,
                    "row_index": candidate["record"].row_index,
                    "source_id": source_id_for_case(candidate["state"].case),
                    "temporary_summary": candidate["memory"],
                    "candidate_raw_output": candidate["raw_output"],
                    "information_usage": candidate["information_usage"],
                }
                for candidate in candidates
            ],
            "writer_retrieval": {
                "similarity_threshold": self.writer_similarity_threshold,
                "top_k": self.writer_top_k,
                "per_candidate": retrieval_audit,
                "deduplicated_union": old_summaries,
            },
            "decision": decision.to_dict(),
            "written_items": [],
        }
        operations = list(decision.actions)
        if not operations:
            window_record["action"] = "ignore"
            window_record["reason"] = decision.reason or "no valid memory actions"
            if decision.error:
                window_record["error"] = decision.error
            return [*ignored, window_record], 0

        by_candidate_id = {candidate["id"]: candidate for candidate in candidates}
        action_payloads: list[dict[str, Any]] = []
        pending_adds = 0
        item_counter = 0
        for operation_index, operation in enumerate(operations):
            selected = [
                by_candidate_id[candidate_id]
                for candidate_id in operation.candidate_ids
                if candidate_id in by_candidate_id
            ] or candidates
            source_ids = list(
                dict.fromkeys(
                    source_id_for_case(candidate["state"].case)
                    or candidate["result"].episode_id
                    for candidate in selected
                )
            )
            provenance = {
                "epoch_id": epoch_id,
                "batch_id": batch_id,
                "source_id": source_ids[0],
                "content_source_ids": source_ids,
                "source_episode_ids": [candidate["result"].episode_id for candidate in selected],
                "source_row_indices": [candidate["record"].row_index for candidate in selected],
                "source_scores": [candidate["record"].score for candidate in selected],
                "retrieved_memory_ids": list(
                    dict.fromkeys(
                        memory_id
                        for candidate in selected
                        for memory_id in candidate["record"].retrieved_memory_ids
                    )
                ),
                "created_by": "evidence_window_memory_writer",
                "candidate_ids": list(operation.candidate_ids),
            }
            for item in operation.items:
                item.metadata.update(provenance)
                if self.checkpoint_run_id:
                    item.metadata.update(
                        {
                            "checkpoint_run_id": self.checkpoint_run_id,
                            "checkpoint_batch_unit": f"epoch:{epoch_id}:batch:{batch_id}",
                            "checkpoint_key": (
                                f"{self.checkpoint_run_id}:epoch:{epoch_id}:"
                                f"batch:{batch_id}:memory:{item_counter}"
                            ),
                        }
                    )
                item_counter += 1
            if (
                self.max_new_memories is not None
                and operation.action == "add"
                and num_written + pending_adds >= self.max_new_memories
            ):
                continue
            if operation.action == "add":
                pending_adds += len(operation.items)
            action_payloads.append(
                {
                    "action": operation.action,
                    "candidate_ids": list(operation.candidate_ids),
                    "source_ids": source_ids,
                    "target_ids": list(operation.target_ids),
                    "reason": operation.reason,
                    "checkpoint_key": (
                        f"{self.checkpoint_run_id}:epoch:{epoch_id}:"
                        f"batch:{batch_id}:action:{operation_index}"
                        if self.checkpoint_run_id
                        else None
                    ),
                    "items": [item.to_dict() for item in operation.items],
                }
            )
        if not action_payloads:
            window_record["action"] = "ignore"
            window_record["reason"] = "epoch max_new_memories limit reached"
            return [*ignored, window_record], 0

        all_source_ids = list(
            dict.fromkeys(
                source_id_for_case(candidate["state"].case)
                or candidate["result"].episode_id
                for candidate in candidates
            )
        )
        response = await self.runner.providers.call(
            ProviderRequest(
                provider="memory",
                operation="apply_actions",
                payload={
                    "actions": action_payloads,
                    "method": "epoch",

                    "source_id": all_source_ids[0],
                    "source_ids": all_source_ids,
                    "epoch_id": epoch_id,
                    "batch_id": batch_id,
                    "candidate_summary": decision.candidate_memory,
                    "information_usage": {
                        candidate["id"]: candidate["information_usage"]
                        for candidate in candidates
                    },
                },
            )
        )
        window_record["action"] = "committed" if response.ok else "error"
        window_record["response"] = response.to_dict()
        if not response.ok:
            window_record["error"] = response.error or "transactional memory action failed"
            raise RuntimeError(window_record["error"])
        added = 0
        if isinstance(response.data, dict):
            committed_actions = list(response.data.get("actions") or [])
            window_record["written_items"] = committed_actions
            for committed in committed_actions:
                if isinstance(committed, dict) and committed.get("action") == "add":
                    added += len(list(committed.get("created_version_ids") or []))
        return [*ignored, window_record], added

    @staticmethod
    def _answer_memory_references(state: Any) -> list[dict[str, Any]]:
        """Collect answer-stage memories and deduplicate them by stable ID.

        A strategy may retrieve memory more than once while building an
        episode.  The relation stage needs one stable record per repository
        version, while retaining enough metadata for auditing and Information
        label mapping.
        """

        references: list[dict[str, Any]] = []
        seen: set[str] = set()
        for context in getattr(state, "retrieved_contexts", []) or []:
            if not isinstance(context, dict) or context.get("source") != "memory":
                continue
            for item in context.get("results") or []:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id") or "").strip()
                if not item_id or item_id in seen:
                    continue
                seen.add(item_id)
                reference = EpochRunner._writer_memory_reference(item)
                usage_assessments = dict(
                    getattr(state, "metadata", {}).get(
                        "memory_usage_assessments"
                    )
                    or {}
                )
                information_usage = dict(
                    getattr(state, "metadata", {}).get(
                        "memory_information_usage"
                    )
                    or {}
                )
                usage_assessments.update(
                    dict(information_usage.get("effects") or {})
                )
                if item_id in usage_assessments:
                    reference["usage_assessment"] = usage_assessments[item_id]
                    reference["metadata"]["usage_assessment"] = (
                        usage_assessments[item_id]
                    )
                reference["retrieval_origin"] = "answer"
                references.append(reference)
        return references

    @staticmethod
    def _writer_memory_reference(memory: dict[str, Any]) -> dict[str, Any]:
        """Normalize a provider memory payload for the writer and audit log."""

        metadata = dict(memory.get("metadata") or {})
        content_source_ids = memory.get("content_source_ids")
        if content_source_ids is None:
            content_source_ids = metadata.get("content_source_ids")
        if content_source_ids is None:
            content_source_ids = metadata.get("source_question_ids")
        if content_source_ids is None:
            content_source_ids = metadata.get("source_id")
        if isinstance(content_source_ids, str):
            content_source_ids = [content_source_ids]
        if not isinstance(content_source_ids, list | tuple | set):
            content_source_ids = []
        normalized_sources = [
            str(value).strip()
            for value in content_source_ids
            if str(value).strip()
        ]
        return {
            "id": str(memory.get("id") or "").strip(),

            "name": memory.get("name"),
            "key": memory.get("key"),
            "content": memory.get("content"),

            "summary": memory.get("summary") or memory.get("content"),
            "condition": memory.get("condition"),
            "applicable_conditions": (
                memory.get("applicable_conditions") or memory.get("condition")
            ),
            "category": memory.get("category"),
            "lifecycle_status": str(
                memory.get("lifecycle_status")
                or metadata.get("lifecycle_status")
                or "active"
            ),
            "content_source_ids": normalized_sources,
            "metadata": metadata,
            "bm25_rank": memory.get("bm25_rank", metadata.get("bm25_rank")),
            "bm25_score": memory.get("bm25_score", metadata.get("bm25_score")),
            "bge_similarity": memory.get(
                "bge_similarity",
                metadata.get("bge_similarity"),
            ),
            "bge_rank": memory.get("bge_rank", metadata.get("bge_rank")),
            "usage_assessment": memory.get(
                "usage_assessment",
                metadata.get("usage_assessment"),
            ),
            "retrieval_origin": memory.get("retrieval_origin"),
        }

    @classmethod
    def _merge_memory_references(
        cls,
        answer_memories: list[dict[str, Any]],
        writer_memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Union answer and summary retrievals by stable repository ID."""

        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for incoming in [*answer_memories, *writer_memories]:
            item_id = str(incoming.get("id") or "").strip()
            if not item_id:
                continue
            if item_id not in merged:
                merged[item_id] = dict(incoming)
                merged[item_id]["retrieval_origin"] = (
                    incoming.get("retrieval_origin") or "writer"
                )
                order.append(item_id)
                continue
            existing = merged[item_id]
            origin = str(existing.get("retrieval_origin") or "")
            incoming_origin = str(incoming.get("retrieval_origin") or "")
            if incoming_origin and incoming_origin not in origin.split("+"):
                existing["retrieval_origin"] = (
                    f"{origin}+{incoming_origin}" if origin else incoming_origin
                )

            for key, value in incoming.items():
                if value is None or value == "":
                    continue
                if key == "usage_assessment" and existing.get(key):
                    continue
                if not existing.get(key):
                    existing[key] = value
            existing_metadata = dict(existing.get("metadata") or {})
            for key, value in dict(incoming.get("metadata") or {}).items():
                if value is not None and key not in existing_metadata:
                    existing_metadata[key] = value
            existing["metadata"] = existing_metadata
        return [merged[item_id] for item_id in order]

    @staticmethod
    def _classify_memory_references(
        references: list[dict[str, Any]],
        current_source_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separate current-question history from independent sources."""

        history: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []
        current_source_id = str(current_source_id or "").strip()
        for reference in references:
            source_values = reference.get("content_source_ids") or []
            if isinstance(source_values, str):
                source_values = [source_values]
            source_values = {str(value).strip() for value in source_values if value}
            if current_source_id and current_source_id in source_values:
                history.append(reference)
            else:
                other.append(reference)
        return history, other

    @staticmethod
    def _attach_information_usage(
        state: Any,
        candidate_memory: dict[str, Any],
    ) -> None:
        """Map Stage-1 ``Information N`` effects back to stable IDs.

        Strict schema validation remains the writer's responsibility.  This
        helper only creates an audit projection that Stage 2 and trajectory
        exporters can reuse.
        """

        usage = candidate_memory.get("information_usage")
        effects: list[Any] | None = None
        if isinstance(usage, dict):
            effects_value = usage.get("information_effects")
            if isinstance(effects_value, list):
                effects = effects_value

        normalized_usage = state.metadata.get("memory_information_usage")
        if isinstance(normalized_usage, dict):
            normalized_effects = normalized_usage.get("effects")
            if isinstance(normalized_effects, dict):
                mapped = {
                    str(key): value
                    for key, value in normalized_effects.items()
                    if key
                }
                if mapped:
                    state.metadata["memory_usage_assessments"] = mapped
                    for reference in list(
                        state.metadata.get("answer_memory_references") or []
                    ):
                        item_id = str(reference.get("id") or "")
                        if item_id not in mapped:
                            continue
                        reference["usage_assessment"] = mapped[item_id]
                        reference.setdefault("metadata", {})[
                            "usage_assessment"
                        ] = mapped[item_id]
                    return
        if effects is None:
            return
        id_map = dict(state.metadata.get("answer_memory_prompt_id_map") or {})
        by_id = {
            str(reference.get("id")): reference
            for reference in list(
                state.metadata.get("answer_memory_references") or []
            )
            if reference.get("id")
        }
        mapped: dict[str, Any] = {}
        for effect in effects:
            if not isinstance(effect, dict):
                continue
            prompt_id = str(effect.get("information_id") or "").strip()
            stable_id = id_map.get(prompt_id)
            if not stable_id:
                continue
            assessment = {
                key: effect.get(key)
                for key in ("analysis", "usage", "assessment", "evidence")
                if effect.get(key) is not None
            }
            if not assessment:
                assessment = {
                    key: value
                    for key, value in effect.items()
                    if key != "information_id"
                }
            mapped[stable_id] = assessment
            reference = by_id.get(stable_id)
            if reference is not None:
                reference["usage_assessment"] = assessment
                metadata = dict(reference.get("metadata") or {})
                metadata["usage_assessment"] = assessment
                reference["metadata"] = metadata
        if mapped:
            state.metadata["memory_usage_assessments"] = mapped

    def _prediction(
        self,
        example: DatasetExample,
        result: EpisodeResult,
        epoch_id: int,
        position: int,
        batch_id: int,
        batch_position: int,
    ) -> dict[str, Any]:
        return {
            "epoch_id": epoch_id,
            "epoch_position": position,
            "batch_id": batch_id,
            "batch_position": batch_position,
            "row_index": example.row_index,
            "case_id": example.case.get("case_id"),
            "dataset": example.case.get("dataset"),
            "metadata": example.metadata,
            "episode_id": result.episode_id,
            "status": result.status,
            "final_answer": result.final_answer,
            "answer_extraction": result.state.metadata.get("answer_extraction"),
            "trajectory_path": str(result.trajectory_path),
        }

    def _learning_record(
        self,
        example: DatasetExample,
        result: EpisodeResult,
        evaluation: EvaluationResult,
        epoch_id: int,
        batch_id: int,
        batch_position: int,
        previous_score: float | None = None,
    ) -> EpisodeLearningRecord:

        observed_criteria: list[str] = []
        missed_positive: list[str] = []
        triggered_negative: list[str] = []
        positive_rubrics: list[dict[str, Any]] = []
        negative_rubrics: list[dict[str, Any]] = []
        evaluation_feedback: dict[str, Any] | None = None
        if str(evaluation.details.get("evaluator_type") or "") == "multiple_choice":
            details = evaluation.details

            evaluation_feedback = {
                "predicted_label": details.get("predicted_label"),
                "parse_status": details.get("parse_status"),
                "reference_label": details.get("reference_label"),
                "reference_text": details.get("reference_text"),
                "score": evaluation.score,
            }
        for grade in evaluation.details.get("rubric_items", []):
            criterion = str(grade.get("criterion") or "")
            points = float(grade.get("points") or 0.0)

            observed = bool(grade.get("observed", grade.get("criteria_met", False)))
            explanation = str(grade.get("explanation") or "").strip()
            rubric_record = {
                "criterion": criterion,
                "observed": observed,
            }
            if points < 0:
                negative_rubrics.append(rubric_record)
            else:
                positive_rubrics.append(rubric_record)
            rendered = criterion if not explanation else f"{criterion} — {explanation}"
            if observed:
                observed_criteria.append(rendered)
                if points < 0:
                    triggered_negative.append(rendered)
            elif points > 0:
                missed_positive.append(rendered)

        tool_queries: list[dict[str, Any]] = []
        for action_entry in result.state.actions:
            action = dict(action_entry.get("action") or {})
            tool_queries.append(
                {
                    "tool": action.get("name"),
                    "arguments": dict(action.get("arguments") or {}),
                }
            )

        retrieved_ids: list[str] = []
        for context in result.state.retrieved_contexts:
            if context.get("source") != "memory":
                continue
            for item in context.get("results") or []:
                item_id = item.get("id")
                if item_id and str(item_id) not in retrieved_ids:
                    retrieved_ids.append(str(item_id))

        return EpisodeLearningRecord(
            epoch_id=epoch_id,
            batch_id=batch_id,
            batch_position=batch_position,
            row_index=example.row_index,
            case_id=(str(example.case["case_id"]) if example.case.get("case_id") else None),
            source_id=source_id_for_case(example.case) or str(example.row_index),
            episode_id=result.episode_id,
            status=result.status,
            question=str(example.case.get("question") or example.case),
            final_answer=result.final_answer or "",
            score=evaluation.score,
            previous_score=previous_score,
            score_change=(
                float(evaluation.score) - float(previous_score)
                if isinstance(evaluation.score, int | float)
                and isinstance(previous_score, int | float)
                else None
            ),
            metrics=evaluation.metrics,
            positive_rubrics=positive_rubrics,
            negative_rubrics=negative_rubrics,
            observed_criteria=observed_criteria,
            missed_positive_criteria=missed_positive,
            triggered_negative_criteria=triggered_negative,
            tool_queries=tool_queries,
            retrieved_memory_ids=retrieved_ids,
            same_source_excluded_count=int(
                result.state.metadata.get("same_source_excluded_count") or 0
            ),
            evaluation_feedback=evaluation_feedback,
        )

    def _aggregate_metrics(self, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
        return aggregate_evaluation_records(evaluations)
