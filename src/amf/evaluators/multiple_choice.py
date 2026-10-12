from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from amf.datasets import DatasetExample
from amf.evaluators.base import BaseEvaluator, EvaluationResult

@dataclass(frozen=True, slots=True)
class ParsedChoice:
    label: str | None
    status: str
    method: str | None

class MultipleChoiceEvaluator(BaseEvaluator):
    """Deterministically score one answer against a multiple-choice label.

    This evaluator never calls an LLM. A solver-format failure is a genuine
    incorrect answer (score 0), while malformed dataset gold data is an INVALID
    technical evaluation (score None).
    """

    async def evaluate(
        self,
        example: DatasetExample,
        response_text: str,
    ) -> EvaluationResult:
        choices, allowed_labels, case_sensitive = self._choice_config(example)
        parsed = parse_choice_response(
            response_text,
            allowed_labels=allowed_labels,
            choices=choices,
            case_sensitive=case_sensitive,
        )
        return self._score_parsed_choice(example, response_text, parsed)

    async def evaluate_extracted(
        self,
        example: DatasetExample,
        response_text: str,
        extraction: dict[str, Any],
    ) -> EvaluationResult:
        """Score the explicit JSON extraction produced after answer generation."""

        if str(extraction.get("status") or "").lower() != "ok":
            error = str(extraction.get("error") or "answer extraction failed")
            return EvaluationResult(
                score=None,
                metrics={"overall_score": None, "accuracy": None},
                details={
                    "evaluation_status": "INVALID",
                    "evaluator_type": "multiple_choice",
                    "grader_error": error,
                    "error": error,
                    "answer_extraction": extraction,
                },
            )

        _, allowed_labels, case_sensitive = self._choice_config(example)
        extracted = extraction.get("answer")
        if extracted is None:
            parsed = ParsedChoice(None, "empty", "llm_answer_extractor")
        elif not isinstance(extracted, str):
            return self._invalid_extracted_label(extracted, extraction)
        else:
            label = extracted.strip()
            if not case_sensitive:
                matches = [item for item in allowed_labels if item.upper() == label.upper()]
                label = matches[0] if len(matches) == 1 else label
            if label not in allowed_labels:
                return self._invalid_extracted_label(label, extraction)
            parsed = ParsedChoice(label, "valid", "llm_answer_extractor")

        result = self._score_parsed_choice(example, response_text, parsed)
        result.details["answer_extraction"] = extraction
        return result

    @staticmethod
    def _choice_config(
        example: DatasetExample,
    ) -> tuple[dict[str, str], tuple[str, ...], bool]:
        choices = {
            str(label): str(text)
            for label, text in dict(example.evaluation.get("choices") or {}).items()
        }
        allowed_labels = tuple(
            str(label)
            for label in example.evaluation.get("allowed_labels") or choices.keys()
        )
        case_sensitive = bool(example.evaluation.get("case_sensitive_labels", False))
        return choices, allowed_labels, case_sensitive

    @staticmethod
    def _invalid_extracted_label(
        label: Any,
        extraction: dict[str, Any],
    ) -> EvaluationResult:
        error = f"Answer extractor returned an unsupported label: {label!r}."
        return EvaluationResult(
            score=None,
            metrics={"overall_score": None, "accuracy": None},
            details={
                "evaluation_status": "INVALID",
                "evaluator_type": "multiple_choice",
                "grader_error": error,
                "error": error,
                "answer_extraction": extraction,
            },
        )

    def _score_parsed_choice(
        self,
        example: DatasetExample,
        response_text: str,
        parsed: ParsedChoice,
    ) -> EvaluationResult:
        reference_label = str(example.evaluation.get("reference_label") or "").strip()
        reference_text = str(example.evaluation.get("reference_text") or "").strip()
        choices, allowed_labels, case_sensitive = self._choice_config(example)
        if not reference_label or reference_label not in allowed_labels:
            return EvaluationResult(
                score=None,
                metrics={"overall_score": None, "accuracy": None},
                details={
                    "evaluation_status": "INVALID",
                    "evaluator_type": "multiple_choice",
                    "error": "Missing or unsupported reference label.",
                },
            )

        is_correct = parsed.label == reference_label
        score = 1.0 if is_correct else 0.0
        refusal = looks_like_refusal(response_text)

        return EvaluationResult(
            score=score,
            metrics={
                "overall_score": score,
                "accuracy": score,
                "answer_parsed": 1 if parsed.label is not None else 0,
                "refusal": 1 if refusal else 0,
            },
            details={
                "evaluation_status": "OK",
                "evaluator_type": "multiple_choice",
                "is_correct": is_correct,
                "predicted_label": parsed.label,
                "predicted_text": choices.get(parsed.label) if parsed.label else None,
                "reference_label": reference_label,
                "reference_text": reference_text,
                "parse_status": parsed.status,
                "parse_method": parsed.method,
                "response_status": "refusal" if refusal else "answered",
                "case_sensitive_labels": case_sensitive,
            },
        )

def parse_choice_response(
    response_text: str,
    *,
    allowed_labels: tuple[str, ...],
    choices: dict[str, str],
    case_sensitive: bool,
) -> ParsedChoice:
    """Extract one intended option using deterministic, auditable rules."""

    response = str(response_text or "").strip()
    if not response:
        return ParsedChoice(None, "empty", None)

    def canonicalize(label: str) -> str | None:
        if case_sensitive:
            return label if label in allowed_labels else None
        upper = label.upper()
        return upper if upper in allowed_labels else None

    final_patterns = (
        r"(?i)final\s+answer\s*(?:is|:|=|-)?\s*[\(\[]?([A-Za-z])(?:[\)\]]|\b)",
        r"(?i)(?:answer|option|choice)\s*(?:is|:|=|-)?\s*[\(\[]?([A-Za-z])(?:[\)\]]|\b)",
    )
    for pattern_index, pattern in enumerate(final_patterns):
        matches = list(re.finditer(pattern, response))
        if matches:
            label = canonicalize(matches[-1].group(1))
            if label is not None:
                method = "final_answer_label" if pattern_index == 0 else "answer_label"
                return ParsedChoice(label, "valid", method)

    concise = re.match(
        r"^\s*[\(\[]?([A-Za-z])[\)\]]?(?:\s*$|[\s.。,:：-]+)",
        response,
    )
    if concise:
        label = canonicalize(concise.group(1))
        if label is not None:
            return ParsedChoice(label, "valid", "leading_label")

    normalized_response = normalize_answer_text(response)
    text_matches: list[str] = []
    for label, choice_text in choices.items():
        normalized_choice = normalize_answer_text(choice_text)
        if not normalized_choice:
            continue
        if (
            normalized_response == normalized_choice
            or contains_normalized_phrase(normalized_response, normalized_choice)
        ):
            text_matches.append(label)
    if len(text_matches) == 1:
        return ParsedChoice(text_matches[0], "valid", "option_text")
    if len(text_matches) > 1:
        return ParsedChoice(None, "ambiguous", "option_text")
    return ParsedChoice(None, "unparsed", None)

def normalize_answer_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())

def contains_normalized_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "

def looks_like_refusal(response_text: str) -> bool:
    normalized = normalize_answer_text(response_text)
    refusal_phrases = (
        "i don t know",
        "i do not know",
        "i cannot answer",
        "i can t answer",
        "unable to answer",
        "cannot provide an answer",
        "无法回答",
        "不能回答",
    )
    return any(phrase in normalized for phrase in refusal_phrases)
