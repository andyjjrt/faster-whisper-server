"""FastAPI application and endpoints."""

from __future__ import annotations

import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .models import (
    AudioJsonResponse,
    AudioVerboseJsonResponse,
    WhisperSegment,
    get_transcriber_for_request,
    initialize_from_env,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    initialize_from_env()
    yield


def _get_version() -> str:
    """Return the package version from pyproject.toml.

    The function reads the ``pyproject.toml`` file located in the project
    root and extracts the ``project.version`` field.  It is used to keep the
    FastAPI application version in sync with the project metadata.
    """
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


app = FastAPI(title="faster-whisper-server", version=_get_version(), lifespan=_lifespan)


def _merge_options(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def _select_transcriber(model_name: str, task: str):
    try:
        return get_transcriber_for_request(model_name, task)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        ) from exc


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> Union[AudioJsonResponse, AudioVerboseJsonResponse]:
    """OpenAI compatible transcription endpoint."""
    transcriber, base_options, batch_size = _select_transcriber(
        model_name, "transcribe"
    )
    if response_format not in {"json", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format}",
                    "type": "invalid_request_error",
                }
            },
        )

    await file.seek(0)
    audio_input = file.file

    options = _merge_options(
        base_options,
        {
            "task": "transcribe",
            "language": language,
            "initial_prompt": prompt,
            "temperature": temperature,
        },
    )
    if batch_size > 1:
        segments, info = transcriber.transcribe(
            audio_input, batch_size=batch_size, **options
        )
    else:
        segments, info = transcriber.transcribe(audio_input, **options)
    segments_list = list(segments)
    transcription = "".join(segment.text for segment in segments_list)

    if response_format == "verbose_json":
        return AudioVerboseJsonResponse(
            task="transcribe",
            language=info.language,
            duration=info.duration,
            text=transcription,
            segments=[
                WhisperSegment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    tokens=segment.tokens,
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob,
                )
                for index, segment in enumerate(segments_list)
            ],
        )

    return AudioJsonResponse(text=transcription)


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> Union[AudioJsonResponse, AudioVerboseJsonResponse]:
    """OpenAI compatible translation endpoint."""
    transcriber, base_options, batch_size = _select_transcriber(model_name, "translate")
    if response_format not in {"json", "verbose_json"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format}",
                    "type": "invalid_request_error",
                }
            },
        )

    await file.seek(0)
    audio_input = file.file

    options = _merge_options(
        base_options,
        {
            "task": "translate",
            "initial_prompt": prompt,
            "temperature": temperature,
        },
    )
    if batch_size > 1:
        segments, info = transcriber.transcribe(
            audio_input, batch_size=batch_size, **options
        )
    else:
        segments, info = transcriber.transcribe(audio_input, **options)
    segments_list = list(segments)
    transcription = "".join(segment.text for segment in segments_list)

    if response_format == "verbose_json":
        return AudioVerboseJsonResponse(
            task="translate",
            language=info.language,
            duration=info.duration,
            text=transcription,
            segments=[
                WhisperSegment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    tokens=segment.tokens,
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob,
                )
                for index, segment in enumerate(segments_list)
            ],
        )

    return AudioJsonResponse(text=transcription)
