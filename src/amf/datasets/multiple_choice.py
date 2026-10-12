from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from amf.datasets.base import DatasetExample
from amf.datasets.jsonl_dataset import JSONLDataset

@dataclass(frozen=True, slots=True)
class MultipleChoiceProfile:
    """Dataset-specific label rules for the shared multiple-choice adapter."""

    labels: tuple[str, ...]
    case_sensitive_labels: bool = False

MULTIPLE_CHOICE_PROFILES: dict[str, MultipleChoiceProfile] = {
    "bioasq": MultipleChoiceProfile(tuple("AB")),
    "ddxplus": MultipleChoiceProfile(
        tuple(chr(code) for code in range(ord("A"), ord("Z") + 1))
        + tuple(chr(code) for code in range(ord("a"), ord("x") + 1)),
        case_sensitive_labels=True,
    ),
    "med_qa": MultipleChoiceProfile(tuple("ABCDE")),
    "medmc_qa": MultipleChoiceProfile(tuple("ABCD")),
    "mmlu_med": MultipleChoiceProfile(tuple("ABCD")),
    "seer": MultipleChoiceProfile(tuple("ABCDEFG")),
    "mmmu_pro_medical": MultipleChoiceProfile(tuple("ABCDEFGHIJ")),
    "mediq": MultipleChoiceProfile(tuple("ABCD")),
}

class MultipleChoiceDataset(JSONLDataset):
    """Read the project's canonical prompt/answer multiple-choice JSONL.

    Gold labels and answer text are kept in ``DatasetExample.evaluation`` and
    never copied into the solving case. This preserves the same no-label-leak
    boundary used by HealthBench.
    """

    def __init__(self, path: str, *, dataset_name: str) -> None:
        super().__init__(path)
        if dataset_name not in MULTIPLE_CHOICE_PROFILES:
            raise ValueError(f"Unknown multiple-choice dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.profile = MULTIPLE_CHOICE_PROFILES[dataset_name]

    def normalize_row(self, raw: dict[str, Any], row_index: int) -> DatasetExample:
        messages = self._normalize_messages(raw.get("prompt"), raw.get("question"))

        original_question = self._messages_to_text(messages)
        choices = self._extract_choices(original_question)
        image_descriptions = self._normalize_image_descriptions(raw, row_index)
        if image_descriptions:

            messages = self._append_image_information(messages, image_descriptions)
        question = self._messages_to_text(messages)
        answer = str(raw.get("answer") or "").strip()
        if not answer:
            raise ValueError(
                f"{self.dataset_name} row {row_index} has no reference answer."
            )
        if answer not in self.profile.labels:
            raise ValueError(
                f"{self.dataset_name} row {row_index} has unsupported answer "
                f"label {answer!r}."
            )

        answer_text = str(raw.get("answer_text") or "").strip()
        if not answer_text:
            answer_text = choices.get(answer, "").strip()
        if not answer_text:
            raise ValueError(
                f"{self.dataset_name} row {row_index} has no text for reference "
                f"label {answer!r}."
            )

        raw_metadata = dict(raw.get("metadata") or {})
        external_id = raw_metadata.get("id")
        if external_id is None:
            external_id = raw.get("id", row_index)
        case_id = f"{self.dataset_name}-{external_id}"

        safe_metadata = {
            key: value
            for key, value in raw_metadata.items()
            if key not in {"answer", "answer_text", "answer_value", "label"}
        }
        safe_metadata.update(
            {
                "row_index": row_index,
                "source": self.dataset_name,
                "source_id": str(external_id),
                "image_count": len(image_descriptions),
            }
        )

        case = {
            "case_id": case_id,
            "question": question,
            "messages": messages,
            "dataset": self.dataset_name,

            "metadata": {
                **safe_metadata,
                "allowed_labels": list(self.profile.labels),
            },
        }
        evaluation = {
            "evaluation_type": "multiple_choice",
            "reference_label": answer,
            "reference_text": answer_text,
            "choices": choices,
            "allowed_labels": list(self.profile.labels),
            "case_sensitive_labels": self.profile.case_sensitive_labels,
        }
        return DatasetExample(
            row_index=row_index,
            case=case,
            metadata=safe_metadata,
            evaluation=evaluation,
        )

    def _normalize_image_descriptions(
        self,
        raw: dict[str, Any],
        row_index: int,
    ) -> list[dict[str, str]]:
        image_paths = raw.get("images") or []
        if not isinstance(image_paths, list):
            raise ValueError(
                f"{self.dataset_name} row {row_index} has a non-list images field."
            )
        descriptions = raw.get("image_descriptions") or []
        if not image_paths:
            return []
        if not isinstance(descriptions, list) or len(descriptions) != len(image_paths):
            raise ValueError(
                f"{self.dataset_name} row {row_index} references {len(image_paths)} "
                "images but does not contain one offline image description per image."
            )

        normalized: list[dict[str, str]] = []
        for position, item in enumerate(descriptions, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{self.dataset_name} row {row_index} image description "
                    f"{position} is not an object."
                )
            label = str(item.get("label") or "").strip()
            description = str(item.get("description") or "").strip()
            if not label or not description:
                raise ValueError(
                    f"{self.dataset_name} row {row_index} image description "
                    f"{position} is missing label or description."
                )
            normalized.append({"label": label, "description": description})
        return normalized

    @staticmethod
    def _append_image_information(
        messages: list[dict[str, str]],
        image_descriptions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        blocks = [
            "## Image Information",
            (
                "The following descriptions correspond to the images referenced "
                "in the question."
            ),
        ]
        blocks.extend(
            f'{item["label"]}:\n{item["description"]}'
            for item in image_descriptions
        )
        image_text = "\n\n".join(blocks)

        enriched = [dict(message) for message in messages]
        target_index = next(
            (
                index
                for index in range(len(enriched) - 1, -1, -1)
                if str(enriched[index].get("role") or "user").lower() == "user"
            ),
            None,
        )
        if target_index is None:
            enriched.append({"role": "user", "content": image_text})
        else:
            original = str(enriched[target_index].get("content") or "").rstrip()
            enriched[target_index]["content"] = f"{original}\n\n{image_text}"
        return enriched

    def _normalize_messages(
        self,
        prompt: Any,
        question: Any,
    ) -> list[dict[str, str]]:
        if isinstance(prompt, list) and prompt:
            messages = [
                {
                    "role": str(item.get("role") or "user"),
                    "content": str(item.get("content") or ""),
                }
                for item in prompt
                if isinstance(item, dict)
            ]
            if messages:
                return messages
        text = str(question or prompt or "")
        if not text.strip():
            raise ValueError(f"{self.dataset_name} row has no prompt text.")
        return [{"role": "user", "content": text}]

    def _messages_to_text(self, messages: list[dict[str, str]]) -> str:
        if len(messages) == 1 and messages[0].get("role") == "user":
            return messages[0].get("content", "")
        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )

    def _extract_choices(self, question: str) -> dict[str, str]:

        choices: dict[str, str] = {}
        label_set = set(self.profile.labels)
        for match in re.finditer(r"(?m)^([A-Za-z])\.\s*(.+?)\s*$", question):
            label = match.group(1)
            if label in label_set:
                choices[label] = match.group(2).strip()
        return choices
