from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from amf.llm import BaseLLMClient
from amf.prompt.answer_extraction import ANSWER_EXTRACTION_PROMPT

@dataclass(slots=True)
class AnswerExtractionResult:
    """Auditable result of converting one free-form answer to an option label."""

    status: str
    answer: str | None
    raw_output: str | None = None
    error: str | None = None
    llm_usage: dict[str, Any] | None = None

    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class LLMAnswerExtractor:
    """Extract a multiple-choice label without replacing the original answer.

    Extraction is an independent request: the model sees only the extraction
    prompt filled with the question and the candidate's final response.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        *,
        use_json_response_format: bool = True,
        max_completion_tokens: int = 128,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.use_json_response_format = use_json_response_format
        self.max_completion_tokens = max(1, int(max_completion_tokens))

        self.max_retries = max(0, int(max_retries))

    def fork(self) -> "LLMAnswerExtractor":
        return LLMAnswerExtractor(
            self.llm.fork(),
            use_json_response_format=self.use_json_response_format,
            max_completion_tokens=self.max_completion_tokens,
            max_retries=self.max_retries,
        )

    async def extract(
        self,
        *,
        question: str,
        candidate_response: str,
        allowed_labels: list[str] | tuple[str, ...] | None = None,
    ) -> AnswerExtractionResult:
        prompt = self.render_prompt(question, candidate_response)
        kwargs: dict[str, Any] = {
            "max_completion_tokens": self.max_completion_tokens,

            "fallback_on_error": False,
        }
        if self.use_json_response_format:
            labels = [str(label) for label in (allowed_labels or ())]
            answer_schema: dict[str, Any] = {
                "type": ["string", "null"],
            }

            if labels:
                answer_schema["enum"] = [*labels, None]

            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_extraction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": answer_schema},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }

        attempts: list[dict[str, Any]] = []
        last_raw: str | None = None
        last_error: str | None = None

        for attempt_index in range(self.max_retries + 1):
            try:
                raw_output = await self.llm.generate(prompt, **kwargs)
                last_raw = raw_output
            except Exception as exc:
                last_raw = None
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "invalid",
                    "raw_output": None,
                    "error": last_error,
                })
                continue

            try:
                payload = json.loads(raw_output)
                if not isinstance(payload, dict) or set(payload) != {"answer"}:
                    raise ValueError('JSON must contain exactly one key: "answer"')
                answer = payload["answer"]
                if answer is not None and not isinstance(answer, str):
                    raise ValueError('"answer" must be a string or null')
                if isinstance(answer, str):
                    answer = answer.strip()
                    if not answer:
                        answer = None

                    elif len(answer) != 1 or not answer.isalnum():
                        raise ValueError(
                            '"answer" must be one option-label character or null'
                        )
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "invalid",
                    "raw_output": raw_output,
                    "error": last_error,
                })
                continue

            attempts.append({
                "attempt": attempt_index + 1,
                "status": "ok",
                "raw_output": raw_output,
                "answer": answer,
            })
            return AnswerExtractionResult(
                status="ok",
                answer=answer,
                raw_output=raw_output,
                llm_usage=self.llm.last_usage,
                attempts=attempts,
            )

        return AnswerExtractionResult(
            status="invalid",
            answer=None,
            raw_output=last_raw,
            error=last_error or "answer extraction failed",
            llm_usage=self.llm.last_usage,
            attempts=attempts,
        )

    @staticmethod
    def render_prompt(question: str, candidate_response: str) -> str:
        return (
            ANSWER_EXTRACTION_PROMPT.replace("<<question>>", str(question))
            .replace("<<candidate_response>>", str(candidate_response))
        )
