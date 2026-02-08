# Faster-Whisper-Server

## Overview

FastAPI server for OpenAI-compatible audio transcription and translation using
faster-whisper. Supports single-model or multi-model hosting.

## Install

```bash
uv pip install -e .
```

## Run (single model)

```bash
faster-whisper-server small --reload
```

## Run (multi-model config)

```bash
faster-whisper-server --config /path/to/config.yaml --reload
```

Example config:

```yaml
models:
	- name: whisper-1
		path: small
		transcribe_options:
			beam_size: 5
			vad_filter: true
	- name: large-fast
		path: /models/large-v3
		translate_options:
			temperature: 0.2
```

## API usage

Transcription:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
	-F "file=@/path/to/audio.wav" \
	-F "model=whisper-1" \
	-F "response_format=json"
```

Translation:

```bash
curl -X POST "http://localhost:8000/v1/audio/translations" \
	-F "file=@/path/to/audio.wav" \
	-F "model=whisper-1" \
	-F "response_format=json"
```

Health check:

```bash
curl "http://localhost:8000/health"
```

## Notes

- When running with a config file, the request `model` must match a config
	`name` entry.
- `--log-level` controls only the `faster-whisper-server` logger and defaults
	to `warning`.
