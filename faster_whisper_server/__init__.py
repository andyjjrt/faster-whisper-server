"""faster-whisper server package."""

from .api import app
from .cli import main

__all__ = ["app", "main"]
