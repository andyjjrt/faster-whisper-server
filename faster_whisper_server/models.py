"""Model management for the faster-whisper server.

This module also contains Pydantic response models used by the API.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Literal, Optional, Tuple

from faster_whisper import WhisperModel
from faster_whisper.transcribe import BatchedInferencePipeline
from faster_whisper.utils import available_models, download_model
from pydantic import BaseModel

from faster_whisper_server.config import ModelConfig, load_config

logger = logging.getLogger("faster-whisper-server")


_DEFAULT_KEY = "_default"

_default_model_path: Optional[str] = None
_default_batch_size: int = 1
_models: Dict[str, WhisperModel] = {}
_batched_pipelines: Dict[str, BatchedInferencePipeline] = {}
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
    global _default_model_path, _models, _batched_pipelines
    global _model_configs, _config_mode
    _config_mode = False
    _model_configs = {}
    _models = {}
    _batched_pipelines = {}
    _default_model_path = model_path


def configure_models_from_config(path: str) -> None:
    """Configure a multi-model server from a YAML config file."""
    global _default_model_path, _models, _batched_pipelines
    global _model_configs, _config_mode
    configs = load_config(path)
    _model_configs = {config.name: config for config in configs}
    _models = {}
    _batched_pipelines = {}
    _default_model_path = None
    _config_mode = True


def configure_batch_size(batch_size: int) -> None:
    """Configure the default batch size for single-model mode."""
    global _default_batch_size
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    _default_batch_size = batch_size


def _load_model(key: str, model_path: str) -> WhisperModel:
    _log_download_if_needed(model_path)
    model_options: Dict[str, Any] = {}
    if _config_mode and key in _model_configs:
        model_options = dict(_model_configs[key].model_options)
    try:
        model = WhisperModel(model_path, **model_options)
    except TypeError as exc:
        raise ValueError(f"Invalid WhisperModel options for '{key}': {exc}") from exc
    logger.info("Model ready: %s", model_path)
    _models[key] = model
    _batched_pipelines.pop(key, None)
    return model


def get_transcriber_for_request(
    request_model: Optional[str],
    task: str,
) -> Tuple[BatchedInferencePipeline, Dict[str, Any], int]:
    """Return a transcriber, its base options, and the effective batch size.

    Batch size is configured via CLI/env (single-model mode) or YAML config
    (multi-model mode). When the effective batch size is > 1, a cached
    ``BatchedInferencePipeline`` is returned.
    """
    model, options = get_model_for_request(request_model, task)

    if _config_mode:
        if not request_model:
            raise ValueError("model is required when using a config file")
        config = _model_configs.get(request_model)
        if config is None:
            raise ValueError(f"unknown model: {request_model}")
        batch_size = config.batch_size
        key = request_model
    else:
        batch_size = _default_batch_size
        key = _DEFAULT_KEY

    if batch_size <= 1:
        return model, options, 1

    pipeline = _batched_pipelines.get(key)
    if pipeline is None:
        pipeline = BatchedInferencePipeline(model=model)
        _batched_pipelines[key] = pipeline
    return pipeline, options, batch_size


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
    batch_size_env = os.getenv("FWS_BATCH_SIZE")
    if config_path and not _config_mode and not _model_configs:
        configure_models_from_config(config_path)
    elif model_path and _default_model_path is None and not _config_mode:
        configure_model(model_path)
    if batch_size_env and not _config_mode:
        configure_batch_size(int(batch_size_env))


class WhisperSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    tokens: Optional[list[int]] = None
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


class AudioJsonResponse(BaseModel):
    text: str


class AudioVerboseJsonResponse(BaseModel):
    task: Literal["transcribe", "translate"]
    language: str
    duration: float
    text: str
    segments: list[WhisperSegment]
