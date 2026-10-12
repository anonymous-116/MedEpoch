from __future__ import annotations

from pathlib import Path

from amf.datasets.base import BaseDataset
from amf.datasets.healthbench import HealthBenchDataset
from amf.datasets.multiple_choice import (
    MULTIPLE_CHOICE_PROFILES,
    MultipleChoiceDataset,
)

SUPPORTED_DATASET_TYPES = (
    "healthbench",
    "livemedbench",
    *MULTIPLE_CHOICE_PROFILES.keys(),
)

def load_dataset(dataset_type: str, path: str | Path) -> BaseDataset:
    normalized = dataset_type.lower().replace("-", "_")
    if normalized in {"healthbench", "health_bench"}:
        return HealthBenchDataset(path)
    if normalized in {"livemedbench", "live_med_bench"}:
        dataset = HealthBenchDataset(path)
        dataset.dataset_name = "livemedbench"
        return dataset
    if normalized in MULTIPLE_CHOICE_PROFILES:
        return MultipleChoiceDataset(path, dataset_name=normalized)
    raise ValueError(f"Unknown dataset type: {dataset_type}")
