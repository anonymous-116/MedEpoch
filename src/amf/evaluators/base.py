from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from amf.datasets import DatasetExample

@dataclass(slots=True)
class EvaluationResult:
    score: float | None
    metrics: dict[str, float | int | None]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class BaseEvaluator(ABC):
    def fork(self) -> "BaseEvaluator":
        """Return an episode-local evaluator for concurrent batches."""

        return copy.copy(self)

    @abstractmethod
    async def evaluate(self, example: DatasetExample, response_text: str) -> EvaluationResult:
        ...

def invalid_evaluation_result(error: str) -> EvaluationResult:
    """Represent a technical Judge failure without turning it into a zero score."""

    message = str(error or "unknown evaluation error")
    return EvaluationResult(
        score=None,
        metrics={"overall_score": None},
        details={
            "evaluation_status": "INVALID",
            "grader_error": message,
            "error": message,
        },
    )

def collect_evaluation_failures(
    records: list[dict[str, Any]],
    *,
    phase: str | None = None,
) -> list[dict[str, Any]]:
    """Extract compact, retry-friendly failure records from saved evaluations."""

    failures: list[dict[str, Any]] = []
    for record in records:
        evaluation = dict(record.get("evaluation") or {})
        details = dict(evaluation.get("details") or {})
        score = evaluation.get("score")
        status = str(details.get("evaluation_status") or "").upper()
        grader_error = details.get("grader_error")
        error = grader_error or details.get("error")
        if score is not None and status != "INVALID" and not grader_error:
            continue
        failure = {
            key: record.get(key)
            for key in (
                "epoch_id",
                "batch_id",
                "batch_position",
                "row_index",
                "case_id",
                "episode_id",
                "final_answer",
            )
            if record.get(key) is not None
        }
        failure.update(
            {
                "phase": phase or record.get("phase"),
                "evaluation_status": status or "INVALID",
                "error": str(error or "evaluation did not produce a valid score"),

                "grader_raw_output": details.get("grader_raw_output"),
            }
        )
        failures.append(failure)
    return failures

def aggregate_evaluation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate both rubric and deterministic multiple-choice evaluations."""

    failures = collect_evaluation_failures(records)
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for record in records:
        evaluation = dict(record.get("evaluation") or {})
        for name, value in dict(evaluation.get("metrics") or {}).items():
            if isinstance(value, int | float):
                values_by_metric[name].append(float(value))

    aggregated: dict[str, Any] = {
        "num_evaluated": len(records),
        "num_valid_evaluations": len(records) - len(failures),
        "num_evaluation_failures": len(failures),
        "evaluation_complete": not failures,
        "evaluation_failure_rows": [
            item.get("row_index")
            for item in failures
            if item.get("row_index") is not None
        ],
    }
    for name, values in values_by_metric.items():
        aggregated[name] = sum(values) / len(values)
        aggregated[f"{name}:n_samples"] = len(values)

    choice_records = [
        dict(record.get("evaluation", {}).get("details") or {})
        for record in records
        if record.get("evaluation", {}).get("details", {}).get("evaluator_type")
        == "multiple_choice"
        and record.get("evaluation", {}).get("score") is not None
    ]
    if choice_records:
        _add_multiple_choice_aggregates(aggregated, choice_records)
    return aggregated

def _add_multiple_choice_aggregates(
    aggregated: dict[str, Any],
    details: list[dict[str, Any]],
) -> None:
    labels = sorted(
        {
            str(item["reference_label"])
            for item in details
            if item.get("reference_label") is not None
        }
    )
    confusion: dict[str, dict[str, int]] = {}
    f1_values: list[float] = []
    for item in details:
        gold = str(item.get("reference_label"))
        predicted = (
            str(item.get("predicted_label"))
            if item.get("predicted_label") is not None
            else "<UNPARSED>"
        )
        confusion.setdefault(gold, {})[predicted] = (
            confusion.setdefault(gold, {}).get(predicted, 0) + 1
        )

    for label in labels:
        true_positive = sum(
            1
            for item in details
            if str(item.get("reference_label")) == label
            and str(item.get("predicted_label")) == label
        )
        false_positive = sum(
            1
            for item in details
            if str(item.get("reference_label")) != label
            and str(item.get("predicted_label")) == label
        )
        false_negative = sum(
            1
            for item in details
            if str(item.get("reference_label")) == label
            and str(item.get("predicted_label")) != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    aggregated["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else None
    aggregated["num_answer_parsed"] = sum(
        1 for item in details if item.get("predicted_label") is not None
    )
    aggregated["num_answer_unparsed"] = sum(
        1 for item in details if item.get("predicted_label") is None
    )
    aggregated["num_refusals"] = sum(
        1 for item in details if item.get("response_status") == "refusal"
    )
    aggregated["confusion_matrix"] = confusion
