"""Command-line entry point for the server."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .api import app
from .models import configure_batch_size, configure_model, configure_models_from_config


def _configure_logging(level_name: str) -> None:
    server_logger = logging.getLogger("faster-whisper-server")
    level = getattr(logging, level_name.upper(), logging.WARNING)
    server_logger.setLevel(level)
    server_logger.propagate = False
    if not server_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        server_logger.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the faster-whisper API server.")
    parser.add_argument("model", nargs="?", help="Model name or path.")
    parser.add_argument(
        "--config",
        "-f",
        help="Path to YAML config for multi-model hosting.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)."
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for dev."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes."
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        help="Server logger level (default: warning).",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=5,
        help="Keep-alive timeout in seconds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1).",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.config and args.model:
        parser.error("Do not pass a model when --config is provided.")
    if not args.config and not args.model:
        parser.error("A model is required unless --config is provided.")
    if args.config and args.batch_size != 1:
        parser.error(
            "--batch-size is not supported with --config; "
            "set batch_size in the YAML config instead."
        )

    if args.config:
        os.environ["FWS_CONFIG_PATH"] = args.config
        configure_models_from_config(args.config)
    else:
        os.environ["FWS_BATCH_SIZE"] = str(args.batch_size)
        configure_batch_size(args.batch_size)
        os.environ["FWS_MODEL_NAME"] = args.model
        configure_model(args.model)

    app_target = (
        "faster_whisper_server.api:app" if args.reload or args.workers > 1 else app
    )
    uvicorn.run(
        app_target,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        timeout_keep_alive=args.timeout_keep_alive,
    )
