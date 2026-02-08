"""Configuration loading for multi-model deployments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: str
    batch_size: int
    model_options: Dict[str, Any]
    transcribe_options: Dict[str, Any]
    translate_options: Dict[str, Any]


def _normalize_options(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def _normalize_batch_size(value: Any, *, default: int = 1) -> int:
    if value is None:
        return default
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size must be an integer") from exc
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return batch_size


def load_config(path: str) -> List[ModelConfig]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    default_batch_size = _normalize_batch_size(data.get("batch_size"), default=1)
    default_model_options = _normalize_options(data.get("model_options"))

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
        batch_size = _normalize_batch_size(
            item.get("batch_size"), default=default_batch_size
        )
        model_options = dict(default_model_options)
        model_options.update(_normalize_options(item.get("model_options")))
        configs.append(
            ModelConfig(
                name=str(name),
                path=str(path_value),
                batch_size=batch_size,
                model_options=model_options,
                transcribe_options=_normalize_options(item.get("transcribe_options")),
                translate_options=_normalize_options(item.get("translate_options")),
            )
        )

    return configs
