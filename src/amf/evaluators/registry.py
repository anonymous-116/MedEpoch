from __future__ import annotations

from typing import Any

from amf.evaluators.base import BaseEvaluator, EvaluationResult
from amf.datasets import DatasetExample
from amf.evaluators.healthbench import HealthBenchEvaluator
from amf.evaluators.multiple_choice import MultipleChoiceEvaluator
from amf.llm import BaseLLMClient
from amf.datasets.multiple_choice import MULTIPLE_CHOICE_PROFILES

RUBRIC_DATASET_TYPES = {"healthbench", "livemedbench"}
MULTIPLE_CHOICE_DATASET_TYPES = set(MULTIPLE_CHOICE_PROFILES)

def resolve_evaluator_type(
    dataset_type: str,
    evaluation_config: dict[str, Any],
) -> str:
    configured = str(
        evaluation_config.get("type")
        or evaluation_config.get("evaluator")
        or ""
    ).strip()
    if configured:
        return configured.lower().replace("-", "_")
    normalized_dataset = dataset_type.lower().replace("-", "_")
    if normalized_dataset in RUBRIC_DATASET_TYPES:
        return "healthbench_rubric"
    if normalized_dataset in MULTIPLE_CHOICE_DATASET_TYPES:
        return "multiple_choice"
    raise ValueError(
        f"Dataset {dataset_type!r} has evaluation enabled but no evaluator type."
    )

def evaluator_requires_llm(evaluator_type: str) -> bool:
    return evaluator_type in {"healthbench", "healthbench_rubric", "rubric"}

def build_dataset_evaluator(
    evaluator_type: str,
    evaluation_config: dict[str, Any],
    *,
    grader: BaseLLMClient | None = None,
    show_progress: bool = False,
) -> BaseEvaluator:
    normalized = evaluator_type.lower().replace("-", "_")
    if normalized == "multiple_choice":
        return MultipleChoiceEvaluator()
    if normalized in {"healthbench", "healthbench_rubric", "rubric"}:
        if grader is None:
            raise ValueError("The HealthBench rubric evaluator requires a Judge LLM.")
        return HealthBenchEvaluator(
            grader,
            use_json_response_format=bool(
                evaluation_config.get("use_json_response_format", False)
            ),
            json_response_mode=str(evaluation_config.get("json_response_mode", "json_schema")),
            show_progress=show_progress,
        )
    raise ValueError(f"Unknown evaluator type: {evaluator_type}")

async def evaluate_dataset_answer(
    evaluator: BaseEvaluator,
    example: DatasetExample,
    response_text: str,
    *,
    answer_extraction: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Use the post-answer JSON label when one was saved for an MCQ run.

    Older prediction files do not contain ``answer_extraction``. They retain
    the deterministic legacy parser so offline re-evaluation remains possible.
    """

    if isinstance(evaluator, MultipleChoiceEvaluator) and answer_extraction is not None:
        return await evaluator.evaluate_extracted(
            example,
            response_text,
            answer_extraction,
        )
    return await evaluator.evaluate(example, response_text)
