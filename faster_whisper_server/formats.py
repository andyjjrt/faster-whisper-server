"""Formatting helpers for transcription outputs."""

from __future__ import annotations

import os
from typing import Iterable, Optional


def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def segments_to_srt(segments: Iterable) -> str:
    lines = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(segment.start)
        end = format_srt_timestamp(segment.end)
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def segments_to_vtt(segments: Iterable) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = format_srt_timestamp(segment.start).replace(",", ".")
        end = format_srt_timestamp(segment.end).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def file_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".wav"
    return os.path.splitext(filename)[1] or ".wav"
