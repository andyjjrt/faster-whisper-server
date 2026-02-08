"""Model management for the faster-whisper server."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from faster_whisper import WhisperModel
from faster_whisper.utils import available_models, download_model

from faster_whisper_server.config import ModelConfig, load_config

logger = logging.getLogger("faster-whisper-server")


_DEFAULT_KEY = "_default"

_default_model_path: Optional[str] = None
_models: Dict[str, WhisperModel] = {}
_model_configs: Dict[str, ModelConfig] = {}
_config_mode = False


def _is_known_model_name(model_name: str) -> bool:
    return "/" in model_name or model_name in available_models()


def _log_download_if_needed(model_name: str) -> None:
    if os.path.exists(model_name):
        logger.info("Loading model from local path: %s", model_name)
        return
    if not _is_known_model_name(model_name):
        return
    try:
        download_model(model_name, local_files_only=True)
        return
    except Exception:
        logger.info(
            "Model '%s' not cached; ",
            "downloading if needed (first run may take a while).",
            model_name,
        )


def configure_model(model_path: str) -> None:
    """Configure a single-model server."""
    global _default_model_path, _models, _model_configs, _config_mode
    _config_mode = False
    _model_configs = {}
    _models = {}
    _default_model_path = model_path


def configure_models_from_config(path: str) -> None:
    """Configure a multi-model server from a YAML config file."""
    global _default_model_path, _models, _model_configs, _config_mode
    configs = load_config(path)
    _model_configs = {config.name: config for config in configs}
    _models = {}
    _default_model_path = None
    _config_mode = True


def _load_model(key: str, model_path: str) -> WhisperModel:
    _log_download_if_needed(model_path)
    model = WhisperModel(model_path)
    logger.info("Model ready: %s", model_path)
    _models[key] = model
    return model


def get_model_for_request(
    request_model: Optional[str],
    task: str,
) -> Tuple[WhisperModel, Dict[str, Any]]:
    if _config_mode:
        if not request_model:
            raise ValueError("model is required when using a config file")
        config = _model_configs.get(request_model)
        if config is None:
            raise ValueError(f"unknown model: {request_model}")
        model = _models.get(request_model)
        if model is None:
            model = _load_model(request_model, config.path)
        options = (
            config.transcribe_options
            if task == "transcribe"
            else config.translate_options
        )
        return model, dict(options)

    if _default_model_path is None:
        raise ValueError("default model is not configured")
    model = _models.get(_DEFAULT_KEY)
    if model is None:
        model = _load_model(_DEFAULT_KEY, _default_model_path)
    return model, {}


def initialize_from_env() -> None:
    config_path = os.getenv("FWS_CONFIG_PATH")
    model_path = os.getenv("FWS_MODEL_NAME")
    if config_path and not _config_mode and not _model_configs:
        configure_models_from_config(config_path)
    elif model_path and _default_model_path is None and not _config_mode:
        configure_model(model_path)
