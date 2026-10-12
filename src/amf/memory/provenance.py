from __future__ import annotations

from typing import Any

def source_id_for_case(case: dict[str, Any]) -> str | None:
    """Return the stable dataset/source identifier used for same-source filtering."""

    dataset = case.get("dataset") or case.get("dataset_name")
    case_id = case.get("case_id")
    if case_id is None:
        case_id = case.get("id")
    if case_id is None:
        metadata = case.get("metadata")
        if isinstance(metadata, dict):
            case_id = metadata.get("row_index")
            if case_id is None:
                case_id = metadata.get("prompt_id")
        if case_id is None:
            case_id = case.get("row_index")
    if case_id is None:
        return None
    case_text = str(case_id).strip()
    if not case_text:
        return None
    return f"{dataset}:{case_text}" if dataset else case_text

def source_ids_from_metadata(metadata: dict[str, Any], role: str) -> list[str]:
    """Read normalized source lists while accepting the writer's compact fields."""

    role_key = f"{role}_source_ids"
    values = metadata.get(role_key)
    if values is None and role == "content":
        values = metadata.get("source_ids")
    if values is None and role == "content":
        values = metadata.get("source_id")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list | tuple | set):
        return []
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result
