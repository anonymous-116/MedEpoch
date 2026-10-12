from __future__ import annotations

import re
from pathlib import Path
from typing import Any

def run_namespace(config: dict[str, Any], *, scripted: bool = False) -> tuple[str, str]:
    """Return stable filesystem names for run artifacts.

    ``llm.model`` is the concrete provider model sent over the API. A stable
    ``llm.model_name`` can keep dated provider upgrades in one experiment
    namespace; the concrete identifier is still saved in run config.json.
    """

    method = sanitize_path_part(str(config.get("method") or config.get("agent_type") or "react"))
    if scripted:
        model = "scripted"
    else:
        llm_config = dict(config.get("llm") or {})
        model = sanitize_path_part(
            str(llm_config.get("model_name") or llm_config.get("model") or "scripted")
        )
    return method, model

def nested_artifact_dir(base_dir: str | Path, config: dict[str, Any], *, scripted: bool = False) -> Path:
    method, model = run_namespace(config, scripted=scripted)
    return Path(base_dir) / method / model

def sanitize_path_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"
