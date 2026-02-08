"""Configuration loading for multi-model deployments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: str
    transcribe_options: Dict[str, Any]
    translate_options: Dict[str, Any]


def _normalize_options(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def load_config(path: str) -> List[ModelConfig]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Config must define a non-empty 'models' list.")

    configs: List[ModelConfig] = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be a mapping.")
        name = item.get("name")
        path_value = item.get("path")
        if not name or not path_value:
            raise ValueError("Each model entry must include 'name' and 'path'.")
        configs.append(
            ModelConfig(
                name=str(name),
                path=str(path_value),
                transcribe_options=_normalize_options(item.get("transcribe_options")),
                translate_options=_normalize_options(item.get("translate_options")),
            )
        )

    return configs
