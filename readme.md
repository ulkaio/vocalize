# vLLM Setup

## Requirements

- **Python 3.10+** (vLLM uses modern type hint syntax like `X | None`)
- **ffmpeg** (for audio processing)

## Installation

### 1. Install Python 3.12 and ffmpeg (if needed)

```bash
brew install python@3.12 ffmpeg
```

### 2. Install dependencies (recommended)

```bash
make install
```

This uses `uv` if available, and falls back to `venv + pip` otherwise, installing from `requirements.txt`.

## Installed Versions

- vLLM 0.11.0
- MLX Whisper 0.4.3
- MLX Audio 0.2.10
- Platform: Apple Silicon (Metal GPU)

---

# Speech-to-Text Transcription

## Overview

The `transcribe.py` script uses Whisper (GPU-accelerated on Apple Silicon via MLX).

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Show help
python transcribe.py --help

# Whisper transcription
python transcribe.py audio.wav                      # base model
python transcribe.py audio.wav -m small             # small model
python transcribe.py audio.wav -m large             # large model
```

## Whisper Models

| Model | Size | Memory | Speed | Accuracy |
|-------|------|--------|-------|----------|
| tiny | ~75 MB | Low | ⚡⚡⚡⚡ | ⭐⭐ |
| base | ~140 MB | Low | ⚡⚡⚡ | ⭐⭐⭐ |
| small | ~466 MB | Medium | ⚡⚡ | ⭐⭐⭐⭐ |
| medium | ~1.5 GB | Medium | ⚡ | ⭐⭐⭐⭐⭐ |
| large | ~2.9 GB | High | Slow | ⭐⭐⭐⭐⭐ |

**Default**: `base` (good balance of speed and accuracy)

## GPU Status

✅ **Running on Apple Silicon GPU (Metal)**

Whisper uses MLX which is optimized for Apple Silicon and runs on the GPU.

## MLX Models on HuggingFace

- `mlx-community/whisper-tiny-mlx`
- `mlx-community/whisper-base-mlx`
- `mlx-community/whisper-small-mlx`
- `mlx-community/whisper-medium-mlx`
- `mlx-community/whisper-large-v3-mlx`

---

# Text-to-Speech (TTS)

## Overview

The `tts.py` script generates natural-sounding speech from text using Kokoro TTS.

## Usage

```bash
# Basic usage
python tts.py "Hello, welcome to Kokoro TTS!"

# Save to specific file
python tts.py "Hello world" -o greeting.wav

# Use different voice
python tts.py "Good morning!" -v am_michael

# Adjust speed
python tts.py "This is slower speech" --speed 0.8

# List all voices
python tts.py --list-voices
```

## Available Voices

| Voice | Description |
|-------|-------------|
| `af_bella` | American Female - Bella (default) |
| `af_sarah` | American Female - Sarah |
| `af_nicole` | American Female - Nicole |
| `af_sky` | American Female - Sky |
| `am_adam` | American Male - Adam |
| `am_michael` | American Male - Michael |
| `bf_emma` | British Female - Emma |
| `bf_isabella` | British Female - Isabella |
| `bm_george` | British Male - George |
| `bm_lewis` | British Male - Lewis |

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-v, --voice` | Voice to use | `af_bella` |
| `-o, --output` | Output audio file | `output.wav` |
| `--speed` | Speed (0.5=slow, 2.0=fast) | 1.0 |
| `--list-voices` | List all voices | - |

---

# Text & Vision Generation

## Overview

The `generate.py` script provides text and vision-language generation using MLX on Apple Silicon.

| Model | Type | Size | Best For |
|-------|------|------|----------|
| **Qwen3-4B** | Text only | ~4GB | General text generation ✅ |
| **Gemma-3-4B** | Vision + Text | ~4GB | Image understanding |

## Usage

```bash
# Text generation
python generate.py "What is Python?"
python generate.py "Explain quantum computing" --temp 0.5

# Interactive chat
python generate.py -i

# Vision-language (describe images)
python generate.py "Describe this image" --image photo.jpg
python generate.py "What's in this picture?" --image screenshot.png
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model: `qwen` or `gemma` | `qwen` |
| `-t, --max-tokens` | Maximum tokens to generate | 512 |
| `--temp` | Temperature (0.0-1.0) | 0.7 |
| `-i, --interactive` | Interactive chat mode | - |
| `-q, --quiet` | Don't stream tokens | - |
| `--image` | Path to image (uses Gemma) | - |

---

# HTTP API Server

## Overview

The `serve.py` script provides a unified HTTP API for all MLX models, keeping them loaded in memory for fast inference. Perfect for accessing models from other machines on your local network.

## Starting the Server

```bash
# Start with LLM only (default)
python serve.py

# Enable all features (LLM + Vision + Audio)
python serve.py --vision --audio

# Audio only (STT + TTS)
python serve.py --audio

# Custom port and Whisper model
python serve.py --audio --whisper-model small --port 8080
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/vision/completions` | POST | Vision completion |
| `/v1/audio/transcriptions` | POST | Speech-to-Text (Whisper) |
| `/v1/audio/speech` | POST | Text-to-Speech (Kokoro) |
| `/v1/audio/voices` | GET | List available TTS voices |
| `/docs` | GET | Interactive API documentation |

## Client Examples

### From Another Machine on Network

Replace `macbook` with your Mac's hostname or IP address:

```bash
# Speech-to-Text (transcribe audio file)
curl -X POST http://macbook:8000/v1/audio/transcriptions \
  -F "file=@recording.wav"

# Text-to-Speech (generate audio)
curl -X POST http://macbook:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from the network!", "voice": "af_bella"}' \
  --output speech.wav

# Chat with LLM
curl -X POST http://macbook:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Python (using requests)

```python
import requests

SERVER = "http://macbook:8000"  # or use IP address

# Speech-to-Text
with open("recording.wav", "rb") as f:
    response = requests.post(
        f"{SERVER}/v1/audio/transcriptions",
        files={"file": f}
    )
    text = response.json()["text"]
    print(f"Transcription: {text}")

# Chat completion
response = requests.post(
    f"{SERVER}/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 256,
    }
)
reply = response.json()["choices"][0]["message"]["content"]
print(f"LLM Reply: {reply}")

# Text-to-Speech
response = requests.post(
    f"{SERVER}/v1/audio/speech",
    json={"input": reply, "voice": "af_bella"}
)
with open("response.wav", "wb") as f:
    f.write(response.content)
```

### Python (using OpenAI client)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://macbook:8000/v1",
    api_key="not-needed",
)

# Chat completion
completion = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(completion.choices[0].message.content)

# Speech-to-Text
with open("recording.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper",
        file=f,
    )
    print(transcript.text)
```

### curl Examples

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# Vision completion
curl -X POST http://localhost:8000/v1/vision/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Describe this image", "image_url": "/path/to/image.png"}'

# Speech-to-Text
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=base"

# Text-to-Speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "am_michael", "speed": 1.0}' \
  --output speech.wav

# List TTS voices
curl http://localhost:8000/v1/audio/voices
```

## Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Text model: `qwen` or `gemma` | `qwen` |
| `--vision` | Load vision model (Gemma-3) | - |
| `--audio` | Enable STT (Whisper) + TTS (Kokoro) | - |
| `--whisper-model` | Whisper size: tiny/base/small/medium/large | `base` |
| `-p, --port` | Server port | 8000 |
| `--host` | Host to bind | 0.0.0.0 |
| `--reload` | Auto-reload for dev | - |

## Running in Background (survives SSH disconnect)

Use the Makefile for easy background management:

```bash
# Start server in background
make start           # LLM + Audio (STT/TTS)
make start-full      # LLM + Vision + Audio

# Manage server
make stop            # Stop the server
make restart         # Restart
make status          # Check if running
make logs            # View recent logs
make tail            # Follow logs in real-time

# Test endpoints
make test-health     # Health check
make test-chat       # Test LLM
make test-tts        # Test text-to-speech
make test-stt        # Test speech-to-text
```

All commands available: `make help`

## Network Access

The server binds to `0.0.0.0` by default, making it accessible from other machines on your local network. Find your Mac's IP with:

```bash
ipconfig getifaddr en0  # Wi-Fi
# or
hostname
```

Then access from other machines using `http://<ip>:8000` or `http://<hostname>.local:8000`.
