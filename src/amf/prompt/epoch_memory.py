from __future__ import annotations

"""Dataset-scoped prompt loading for the Epoch memory workflow.

The wording of the Epoch prompts is an experiment artifact.  Keep it in
``prompts/<dataset>/`` and load it verbatim; this module only performs the
small placeholder substitutions required to insert the current case.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_PROMPT_FILES = (
    "stage1_no_information.txt",
    "stage1_with_information.txt",
    "action_common.txt",
    "action_example_1.txt",
    "action_example_2.txt",
    "history_compare.txt",
    "window_action.txt",
    "window_repair.txt",
    "action_repair.txt",
    "conflict_resolution.txt",
)

@dataclass(frozen=True, slots=True)
class EpochPromptBundle:
    """The fixed Epoch prompt fragments used by one dataset."""

    root: Path
    contents: dict[str, str]

    @classmethod
    def for_dataset(
        cls,
        dataset_name: str,
        *,
        prompt_root: str | Path | None = None,
        allow_healthbench_fallback: bool = False,
    ) -> "EpochPromptBundle":
        repo_root = Path(__file__).resolve().parents[3]
        root = Path(prompt_root) if prompt_root is not None else repo_root / "prompts"
        dataset = str(dataset_name or "").strip() or "healthbench"
        candidate = root / dataset
        if not candidate.is_dir() and allow_healthbench_fallback:
            fallback = root / "healthbench"
            if fallback.is_dir():
                candidate = fallback
        missing = [name for name in _PROMPT_FILES if not (candidate / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Epoch prompt directory is incomplete: "
                f"{candidate}; missing {', '.join(missing)}"
            )
        return cls(
            candidate,
            {
                name: (candidate / name).read_text(encoding="utf-8")
                for name in _PROMPT_FILES
            },
        )

    def read(self, filename: str) -> str:
        if filename not in _PROMPT_FILES:
            raise ValueError(f"Unknown Epoch prompt fragment: {filename}")
        return self.contents[filename]

    @property
    def stage1_no_information(self) -> str:
        return self.read("stage1_no_information.txt")

    @property
    def stage1_with_information(self) -> str:
        return self.read("stage1_with_information.txt")

    @property
    def action_common(self) -> str:
        return self.read("action_common.txt")

    @property
    def action_example_1(self) -> str:
        return anonymize_example_ids(self.read("action_example_1.txt"))

    @property
    def action_example_2(self) -> str:
        return anonymize_example_ids(self.read("action_example_2.txt"))

    @property
    def history_compare(self) -> str:
        return self.read("history_compare.txt")

    @property
    def window_action(self) -> str:
        return self.read("window_action.txt")

    @property
    def window_repair(self) -> str:
        return self.read("window_repair.txt")

    @property
    def conflict_resolution(self) -> str:
        return self.read("conflict_resolution.txt")

def replace_prompt_placeholders(template: str, **values: str) -> str:
    """Replace only named placeholders, preserving JSON braces in the text."""

    result = template
    for name, value in values.items():
        result = result.replace("{{" + name + "}}", value)
    return result

def anonymize_example_ids(example: str) -> str:
    """Replace descriptive IDs in a fixed shot with local numeric labels.

    The files under ``prompts/<dataset>/`` remain verbatim copies of the
    experiment Prompt.  Before a shot is sent to the model, however, its
    illustrative identifiers follow the same ``1, 2, 3`` convention as the
    current database records.  This prevents the shot from teaching the model
    to emit descriptive names while keeping all real repository IDs private.
    """

    identifiers = list(dict.fromkeys(re.findall(r"ID: `([^`]+)`", example)))
    result = example
    for index, identifier in enumerate(identifiers, start=1):
        numeric_id = str(index)
        result = result.replace(f"ID: `{identifier}`", f"ID: `{numeric_id}`")
        result = result.replace(f'"{identifier}"', f'"{numeric_id}"')
    return result
