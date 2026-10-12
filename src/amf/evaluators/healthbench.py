from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from amf.datasets import DatasetExample
from amf.evaluators.base import BaseEvaluator, EvaluationResult
from amf.llm import BaseLLMClient
from amf.prompt.healthbench import HEALTHBENCH_BATCH_GRADER_TEMPLATE

@dataclass(slots=True)
class RubricItem:
    criterion: str
    points: float
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RubricItem":
        return cls(
            criterion=str(payload["criterion"]),
            points=float(payload["points"]),
            tags=list(payload.get("tags") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_for_grader(self) -> str:

        return self.criterion

@dataclass(slots=True)
class RubricGrade:
    criterion: str
    points: float
    tags: list[str]
    observed: bool
    explanation: str
    awarded_points: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def rubric_response_format(count: int) -> dict[str, Any]:
    """Constrain structure only; rubric meanings/weights never enter the schema.

    Length and ID range cannot enforce unique IDs, so parse_batch_grades still
    checks exact coverage after generation.
    """
    if count < 1:
        raise ValueError("Rubric count must be positive")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "healthbench_rubric_results", "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "required": ["rubric_results"],
                "properties": {"rubric_results": {
                    "type": "array", "minItems": count, "maxItems": count,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["id", "observed", "evidence"],
                        "properties": {
                            "id": {"type": "integer", "enum": list(range(1, count + 1))},
                            "observed": {"type": "boolean"},
                            "evidence": {"type": "string"},
                        },
                    },
                }},
            },
        },
    }

class HealthBenchEvaluator(BaseEvaluator):
    """Rubric-based evaluator for HealthBench examples.

    The evaluated model never sees rubrics. This class is called after an
    episode finishes and uses one separate judge call to grade all rubric items.
    """

    def __init__(
        self,
        grader: BaseLLMClient,
        max_retries: int = 3,
        strict_json: bool = False,
        use_json_response_format: bool = False,
        length_adjustment_center: float | None = None,
        length_adjustment_penalty_per_500_chars: float | None = None,
        show_progress: bool = False,
        progress_desc: str | None = None,
        json_response_mode: str = "json_schema",
    ) -> None:
        self.grader = grader
        self.max_retries = max_retries
        self.strict_json = strict_json
        self.use_json_response_format = use_json_response_format
        if json_response_mode not in {"json_schema", "json_object"}:
            raise ValueError("json_response_mode must be json_schema or json_object")
        self.json_response_mode = json_response_mode
        self.length_adjustment_center = length_adjustment_center
        self.length_adjustment_penalty_per_500_chars = length_adjustment_penalty_per_500_chars
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        if (length_adjustment_center is None) != (length_adjustment_penalty_per_500_chars is None):
            raise ValueError("Length adjustment center and penalty must be set together.")

    def fork(self) -> "HealthBenchEvaluator":
        """Clone the Judge client so concurrent rubric calls cannot share state."""

        return HealthBenchEvaluator(
            self.grader.fork(),
            max_retries=self.max_retries,
            strict_json=self.strict_json,
            use_json_response_format=self.use_json_response_format,
            json_response_mode=self.json_response_mode,
            length_adjustment_center=self.length_adjustment_center,
            length_adjustment_penalty_per_500_chars=(
                self.length_adjustment_penalty_per_500_chars
            ),
            show_progress=self.show_progress,
            progress_desc=self.progress_desc,
        )

    async def evaluate(self, example: DatasetExample, response_text: str) -> EvaluationResult:
        rubric_items = [
            RubricItem.from_dict(item)
            for item in example.evaluation.get("rubrics", [])
        ]
        if not rubric_items:
            return EvaluationResult(
                score=None,
                metrics={"overall_score": None},
                details={"error": "HealthBench example has no rubrics."},
            )

        messages = list(example.case.get("messages") or [])
        if not messages:
            messages = [{"role": "user", "content": example.case.get("question", "")}]

        grades, grader_raw_output, grader_error = await self._grade_rubric_items(
            messages, response_text, rubric_items
        )

        overall_score = (
            None if grader_error is not None else calculate_score(rubric_items, grades)
        )
        metrics: dict[str, float | int | None] = {"overall_score": overall_score}
        if overall_score is not None and self.length_adjustment_center is not None:
            assert self.length_adjustment_penalty_per_500_chars is not None
            metrics["overall_score_length_adjusted"] = calculate_length_adjusted_score(
                overall_score,
                response_text,
                center=self.length_adjustment_center,
                penalty_per_500_chars=self.length_adjustment_penalty_per_500_chars,
            )

        if grader_error is None:
            example_tags = list(example.metadata.get("example_tags") or [])
            for tag in example_tags:
                metrics[tag] = overall_score
            metrics.update(calculate_rubric_tag_scores(rubric_items, grades))

        return EvaluationResult(
            score=overall_score,
            metrics=metrics,
            details={
                "rubric_items": [grade.to_dict() for grade in grades],
                "prompt": messages,
                "completion": [{"role": "assistant", "content": response_text}],
                "prompt_id": example.case.get("case_id"),
                "evaluation_status": "INVALID" if grader_error else "OK",
                "grader_raw_output": grader_raw_output,
                "grader_error": grader_error,
            },
        )

    async def _grade_rubric_items(
        self,
        messages: list[dict[str, Any]],
        response_text: str,
        rubric_items: list[RubricItem],
    ) -> tuple[list[RubricGrade], str, str | None]:
        prompt = self._build_batch_grader_prompt(messages, response_text, rubric_items)
        raw_output = ""
        last_error: str | None = None
        for _ in range(max(self.max_retries, 1)):
            request_kwargs: dict[str, Any] = {}
            if self.use_json_response_format:

                request_kwargs["response_format"] = (
                    rubric_response_format(len(rubric_items))
                    if self.json_response_mode == "json_schema"
                    else {"type": "json_object"}
                )
            raw_output = await self.grader.generate(prompt, **request_kwargs)
            parsed = parse_grader_json(raw_output)
            batch_results, last_error = parse_batch_grades(parsed, len(rubric_items))
            if batch_results is not None:
                grades = [
                    RubricGrade(
                        criterion=item.criterion,
                        points=item.points,
                        tags=item.tags,
                        observed=result["observed"],
                        explanation=result["evidence"],
                        awarded_points=item.points if result["observed"] else 0.0,
                    )
                    for item, result in zip(rubric_items, batch_results, strict=True)
                ]
                return grades, raw_output, None

        if self.strict_json:
            raise ValueError(f"Invalid HealthBench grader output: {raw_output}")

        grades = [
            RubricGrade(
                criterion=item.criterion,
                points=item.points,
                tags=item.tags,
                observed=False,
                explanation="Invalid batch grader output; rubric treated as not observed.",
                awarded_points=0.0,
            )
            for item in rubric_items
        ]
        return grades, raw_output, last_error or "invalid batch grader output"

    def _build_batch_grader_prompt(
        self,
        messages: list[dict[str, Any]],
        response_text: str,
        rubric_items: list[RubricItem],
    ) -> str:
        convo_str = "\n\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
        rubric_str = "\n".join(
            f"{index}. {item.format_for_grader()}"
            for index, item in enumerate(rubric_items, 1)
        )

        return (
            HEALTHBENCH_BATCH_GRADER_TEMPLATE
            .replace("<<conversation>>", convo_str)
            .replace("<<candidate_answer>>", response_text)
            .replace("<<rubric_items>>", rubric_str)
        )

def parse_grader_json(raw_output: str) -> dict[str, Any]:
    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}

def parse_batch_grades(
    payload: dict[str, Any],
    expected_count: int,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate a complete, one-result-per-rubric judge response."""

    raw_results = payload.get("rubric_results")
    if not isinstance(raw_results, list):
        return None, "grader output missing rubric_results list"

    results_by_id: dict[int, dict[str, Any]] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            return None, "rubric result is not an object"
        try:
            rubric_id = int(raw_result.get("id"))
        except (TypeError, ValueError):
            return None, "rubric result has an invalid id"
        if rubric_id in results_by_id:
            return None, f"duplicate rubric result id {rubric_id}"
        if type(raw_result.get("observed")) is not bool:
            return None, f"rubric result {rubric_id} has no boolean observed field"
        evidence = raw_result.get("evidence", "")
        if not isinstance(evidence, str):
            return None, f"rubric result {rubric_id} has non-string evidence"
        results_by_id[rubric_id] = {
            "id": rubric_id,
            "observed": raw_result["observed"],
            "evidence": evidence,
        }

    expected_ids = set(range(1, expected_count + 1))
    actual_ids = set(results_by_id)
    if actual_ids != expected_ids:
        return None, (
            f"rubric result ids mismatch: expected {sorted(expected_ids)}, "
            f"got {sorted(actual_ids)}"
        )
    return [results_by_id[index] for index in range(1, expected_count + 1)], None

def calculate_score(rubric_items: list[RubricItem], grades: list[RubricGrade]) -> float | None:
    total_possible_points = sum(item.points for item in rubric_items if item.points > 0)
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        item.points
        for item, grade in zip(rubric_items, grades, strict=True)
        if grade.observed
    )
    return achieved_points / total_possible_points

def calculate_rubric_tag_scores(
    rubric_items: list[RubricItem],
    grades: list[RubricGrade],
) -> dict[str, float]:
    tag_items_grades: dict[str, list[tuple[RubricItem, RubricGrade]]] = defaultdict(list)
    for item, grade in zip(rubric_items, grades, strict=True):
        seen_tags: set[str] = set()
        for tag in item.tags:
            if tag in seen_tags:
                continue
            tag_items_grades[tag].append((item, grade))
            seen_tags.add(tag)

    tag_scores: dict[str, float] = {}
    for tag, pairs in tag_items_grades.items():
        items = [pair[0] for pair in pairs]
        item_grades = [pair[1] for pair in pairs]
        score = calculate_score(items, item_grades)
        if score is not None:
            tag_scores[tag] = score
    return tag_scores

def calculate_length_adjusted_score(
    score: float,
    response_text: str,
    *,
    center: float,
    penalty_per_500_chars: float,
) -> float:
    return score - penalty_per_500_chars * ((len(response_text) - center) / 500.0)
