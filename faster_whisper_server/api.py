"""FastAPI application and endpoints."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .formats import file_suffix, segments_to_srt, segments_to_vtt
from .models import get_model_for_request, initialize_from_env

@asynccontextmanager
async def _lifespan(_: FastAPI):
    initialize_from_env()
    yield


app = FastAPI(title="faster-whisper-server", lifespan=_lifespan)


def _merge_options(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def _select_model(model_name: str, task: str):
    try:
        return get_model_for_request(model_name, task)
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
def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> JSONResponse:
    """OpenAI compatible transcription endpoint."""
    model, base_options = _select_model(model_name, "transcribe")
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix(file.filename)) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        options = _merge_options(
            base_options,
            {
                "task": "transcribe",
                "language": language,
                "initial_prompt": prompt,
                "temperature": temperature,
            },
        )
        segments, info = model.transcribe(tmp_path, **options)
        segments_list = list(segments)
        transcription = "".join(segment.text for segment in segments_list)

        if response_format == "text":
            return PlainTextResponse(transcription)
        if response_format == "srt":
            return PlainTextResponse(segments_to_srt(segments_list))
        if response_format == "vtt":
            return PlainTextResponse(segments_to_vtt(segments_list))
        if response_format == "verbose_json":
            return JSONResponse(
                content={
                    "task": "transcribe",
                    "language": info.language,
                    "duration": info.duration,
                    "text": transcription,
                    "segments": [
                        {
                            "id": index,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                            "tokens": segment.tokens,
                            "temperature": segment.temperature,
                            "avg_logprob": segment.avg_logprob,
                            "compression_ratio": segment.compression_ratio,
                            "no_speech_prob": segment.no_speech_prob,
                        }
                        for index, segment in enumerate(segments_list)
                    ],
                }
            )
        if response_format == "json":
            return JSONResponse(content={"text": transcription})

        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format}",
                    "type": "invalid_request_error",
                }
            },
        )
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),
) -> JSONResponse:
    """OpenAI compatible translation endpoint."""
    model, base_options = _select_model(model_name, "translate")
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix(file.filename)) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        options = _merge_options(
            base_options,
            {
                "task": "translate",
                "initial_prompt": prompt,
                "temperature": temperature,
            },
        )
        segments, info = model.transcribe(tmp_path, **options)
        segments_list = list(segments)
        transcription = "".join(segment.text for segment in segments_list)

        if response_format == "text":
            return PlainTextResponse(transcription)
        if response_format == "srt":
            return PlainTextResponse(segments_to_srt(segments_list))
        if response_format == "vtt":
            return PlainTextResponse(segments_to_vtt(segments_list))
        if response_format == "verbose_json":
            return JSONResponse(
                content={
                    "task": "translate",
                    "language": info.language,
                    "duration": info.duration,
                    "text": transcription,
                    "segments": [
                        {
                            "id": index,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                            "tokens": segment.tokens,
                            "temperature": segment.temperature,
                            "avg_logprob": segment.avg_logprob,
                            "compression_ratio": segment.compression_ratio,
                            "no_speech_prob": segment.no_speech_prob,
                        }
                        for index, segment in enumerate(segments_list)
                    ],
                }
            )
        if response_format == "json":
            return JSONResponse(content={"text": transcription})

        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Unsupported response_format: {response_format}",
                    "type": "invalid_request_error",
                }
            },
        )
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
