"""HTTP API server for MLX models on Apple Silicon.

Unified API for:
- LLM text generation (Qwen3)
- Vision-language (Gemma-3)
- Speech-to-Text transcription (Whisper)
- Text-to-Speech synthesis (Kokoro)

Usage:
    # Start server with all features
    python serve.py --audio
    
    # Or specific features
    python serve.py --vision --audio
    
Client usage:
    # Text generation
    curl -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
    
    # Speech-to-Text
    curl -X POST http://localhost:8000/v1/audio/transcriptions \
        -F "file=@recording.wav"
    
    # Text-to-Speech
    curl -X POST http://localhost:8000/v1/audio/speech \
        -H "Content-Type: application/json" \
        -d '{"input": "Hello world", "voice": "af_bella"}' \
        --output speech.wav
"""

import argparse
import base64
import io
import os
import re
import struct
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from mlx_lm import generate as lm_generate
from mlx_lm import load as lm_load
from mlx_lm.sample_utils import make_sampler

import numpy as np


# Available models
MODELS = {
    "qwen": "mlx-community/Qwen3-4B-8bit",
    "gemma": "mlx-community/gemma-3-4b-it-8bit",
}

DEFAULT_MODEL = "gemma"


# Environment variable based config (persists across uvicorn reloads)
def get_config() -> dict[str, Any]:
    """Get server configuration from environment variables.

    Returns:
        Configuration dictionary.
    """
    return {
        "model_name": os.environ.get("MLX_MODEL", DEFAULT_MODEL),
        "load_vision": os.environ.get("MLX_VISION", "").lower() == "true",
        "load_audio": os.environ.get("MLX_AUDIO", "").lower() == "true",
        "whisper_model": os.environ.get("MLX_WHISPER_MODEL", "base"),
    }


# Whisper models
WHISPER_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}

# Kokoro TTS paths
KOKORO_MODEL_DIR = Path(__file__).parent / "models" / "kokoro"
KOKORO_MODEL_FILE = KOKORO_MODEL_DIR / "kokoro-v0_19.onnx"
KOKORO_VOICES_FILE = KOKORO_MODEL_DIR / "voices.bin"

# Available TTS voices
TTS_VOICES = [
    "af_bella", "af_sarah", "af_nicole", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
]


# Global model storage (loaded at startup, stays in memory)
class ModelState:
    """Container for loaded models and their components."""

    def __init__(self) -> None:
        """Initialize empty model state."""
        self.text_model: Any = None
        self.text_tokenizer: Any = None
        self.text_model_name: str = ""
        
        self.vision_model: Any = None
        self.vision_processor: Any = None
        self.vision_config: Any = None
        self.vision_model_name: str = ""
        
        # Audio models
        self.whisper_model: str = ""  # Just store the model path, mlx_whisper loads on demand
        self.tts_model: Any = None  # Kokoro instance


state = ModelState()


# Request/Response models (OpenAI-compatible format)
class Message(BaseModel):
    """Chat message with role and content."""

    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions."""

    messages: list[Message] = Field(..., description="List of messages in conversation")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(default=False, description="Stream response (not yet supported)")


class VisionCompletionRequest(BaseModel):
    """Request body for vision completions."""

    prompt: str = Field(..., description="Text prompt for the image")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image")
    image_url: Optional[str] = Field(default=None, description="URL to image (local path)")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")


class Choice(BaseModel):
    """A single completion choice."""

    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response for chat completions (OpenAI-compatible)."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    id: str
    object: str = "model"
    owned_by: str = "mlx-community"


class ModelsResponse(BaseModel):
    """Response listing available models."""

    object: str = "list"
    data: list[ModelInfo]


# Audio request/response models (OpenAI-compatible)
class TranscriptionResponse(BaseModel):
    """Response for audio transcription."""

    text: str = Field(..., description="Transcribed text")


class TTSRequest(BaseModel):
    """Request body for text-to-speech."""

    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="af_bella", description="Voice ID to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    response_format: str = Field(default="wav", description="Audio format (wav)")
    stream: bool = Field(default=False, description="Stream audio as it is generated")
    stream_chunk_chars: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Approximate character count per streamed TTS chunk",
    )


def load_text_model(model_name: str) -> None:
    """Load text generation model into memory.

    Args:
        model_name: Model key from MODELS dict or HuggingFace path.
    """
    model_path = MODELS.get(model_name, model_name)
    print(f"Loading text model: {model_path}")
    print("Using Apple Silicon GPU (Metal) 🚀")
    
    state.text_model, state.text_tokenizer = lm_load(model_path)
    state.text_model_name = model_path
    print(f"✓ Text model loaded: {model_path}")


def load_vision_model(model_name: str = "gemma") -> None:
    """Load vision-language model into memory.

    Args:
        model_name: Model key from MODELS dict or HuggingFace path.
    """
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config

    model_path = MODELS.get(model_name, model_name)
    print(f"Loading vision model: {model_path}")
    
    state.vision_model, state.vision_processor = vlm_load(model_path)
    state.vision_config = load_config(model_path)
    state.vision_model_name = model_path
    print(f"✓ Vision model loaded: {model_path}")


def load_whisper_model(model_name: str = "base") -> None:
    """Configure Whisper model for transcription.

    Args:
        model_name: Whisper model size (tiny, base, small, medium, large).
    """
    model_path = WHISPER_MODELS.get(model_name, model_name)
    print(f"Configuring Whisper model: {model_path}")
    state.whisper_model = model_path
    print(f"✓ Whisper configured: {model_path}")


def load_tts_model() -> None:
    """Load Kokoro TTS model into memory."""
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("⚠ kokoro-onnx not installed, TTS disabled")
        print("  Install with: uv pip install kokoro-onnx soundfile")
        return
    
    if not KOKORO_MODEL_FILE.exists() or not KOKORO_VOICES_FILE.exists():
        print("⚠ Kokoro model files not found, TTS disabled")
        print("  Run: python tts.py --list-voices  (to trigger download)")
        return
    
    print(f"Loading Kokoro TTS model...")
    state.tts_model = Kokoro(str(KOKORO_MODEL_FILE), str(KOKORO_VOICES_FILE))
    print(f"✓ Kokoro TTS loaded")


def _split_text_for_streaming(text: str, target_chars: int) -> list[str]:
    """Split text into sentence-like chunks for streaming TTS."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        sentence = sentence.strip()
        sentence_len = len(sentence)
        if current and current_len + sentence_len + 1 > target_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + (1 if current_len else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Create a WAV header with unknown data size for streaming."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    unknown_size = 0xFFFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        unknown_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        unknown_size,
    )


def _float_to_int16_bytes(samples: np.ndarray) -> bytes:
    """Convert float audio to 16-bit PCM bytes."""
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    return pcm.tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after models are loaded.
    """
    # Startup: Load models
    print("=" * 50)
    print("Starting MLX Model Server")
    print("=" * 50)
    
    # Get config from environment variables (persists across uvicorn reloads)
    config = get_config()
    model_name = config["model_name"]
    load_vision = config["load_vision"]
    load_audio = config["load_audio"]
    whisper_model = config["whisper_model"]
    
    load_text_model(model_name)
    
    if load_vision:
        load_vision_model("gemma")
    
    if load_audio:
        load_whisper_model(whisper_model)
        load_tts_model()
    
    print("=" * 50)
    print("Server ready! Models loaded in memory.")
    print("=" * 50)
    
    yield
    
    # Shutdown: Cleanup (optional, models will be freed anyway)
    print("Shutting down server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="MLX Model Server",
        description="HTTP API for MLX models on Apple Silicon",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Enable CORS for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status dictionary indicating server health.
    """
    return {"status": "healthy"}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List loaded models (OpenAI-compatible).

    Returns:
        List of available models.
    """
    models = []
    
    if state.text_model is not None:
        models.append(ModelInfo(id=state.text_model_name))
    
    if state.vision_model is not None:
        models.append(ModelInfo(id=state.vision_model_name))
    
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate chat completions (OpenAI-compatible).

    Args:
        request: Chat completion request with messages.

    Returns:
        Generated completion response.

    Raises:
        HTTPException: If text model is not loaded.
    """
    if state.text_model is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")
    
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet supported")
    
    # Convert messages to the format expected by the tokenizer
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Apply chat template
    if state.text_tokenizer.chat_template is not None:
        formatted_prompt = state.text_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        # Fallback: just use the last user message
        formatted_prompt = messages[-1]["content"] if messages else ""
    
    # Create sampler and generate
    sampler = make_sampler(temp=request.temperature)
    
    response_text = lm_generate(
        state.text_model,
        state.text_tokenizer,
        prompt=formatted_prompt,
        max_tokens=request.max_tokens,
        sampler=sampler,
        verbose=False,
    )
    
    return ChatCompletionResponse(
        model=state.text_model_name,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=response_text),
            )
        ],
    )


@app.post("/v1/vision/completions", response_model=ChatCompletionResponse)
async def vision_completions(request: VisionCompletionRequest) -> ChatCompletionResponse:
    """Generate completions from images using vision model.

    Args:
        request: Vision completion request with prompt and image.

    Returns:
        Generated completion response.

    Raises:
        HTTPException: If vision model not loaded or no image provided.
    """
    if state.vision_model is None:
        raise HTTPException(
            status_code=503,
            detail="Vision model not loaded. Start server with --vision flag",
        )
    
    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=400,
            detail="Either image_base64 or image_url must be provided",
        )
    
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    
    # Handle image input
    if request.image_base64:
        # Decode base64 and save to temp file
        image_data = base64.b64decode(request.image_base64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_data)
            image_path = f.name
    else:
        image_path = request.image_url  # Treat as local path
    
    try:
        # Apply chat template
        formatted_prompt = apply_chat_template(
            state.vision_processor,
            state.vision_config,
            request.prompt,
            num_images=1,
        )
        
        # Generate response
        response_text = vlm_generate(
            state.vision_model,
            state.vision_processor,
            formatted_prompt,
            image=image_path,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            verbose=False,
        )
        
        return ChatCompletionResponse(
            model=state.vision_model_name,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                )
            ],
        )
    finally:
        # Cleanup temp file if we created one
        if request.image_base64:
            Path(image_path).unlink(missing_ok=True)


@app.post("/v1/completions")
async def completions(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> ChatCompletionResponse:
    """Simple completion endpoint (non-chat format).

    Args:
        prompt: Text prompt for completion.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated completion response.
    """
    # Convert to chat format and use chat endpoint
    request = ChatCompletionRequest(
        messages=[Message(role="user", content=prompt)],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return await chat_completions(request)


# =============================================================================
# Audio Endpoints (STT & TTS)
# =============================================================================


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="base", description="Whisper model size"),
) -> TranscriptionResponse:
    """Transcribe audio to text using Whisper (OpenAI-compatible).

    Args:
        file: Audio file (WAV, MP3, etc.).
        model: Whisper model size (tiny, base, small, medium, large).

    Returns:
        Transcription response with text.

    Raises:
        HTTPException: If Whisper is not configured.
    """
    if not state.whisper_model:
        raise HTTPException(
            status_code=503,
            detail="Whisper not loaded. Start server with --audio flag",
        )
    
    import mlx_whisper
    
    # Save uploaded file to temp location
    suffix = Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Use configured model or override from request
        model_path = WHISPER_MODELS.get(model, state.whisper_model)
        
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=model_path,
        )
        
        return TranscriptionResponse(text=result["text"].strip())
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest) -> Response:
    """Convert text to speech using Kokoro TTS (OpenAI-compatible).

    Args:
        request: TTS request with text and voice settings.

    Returns:
        WAV audio file response.

    Raises:
        HTTPException: If TTS model not loaded or invalid voice.
    """
    if state.tts_model is None:
        raise HTTPException(
            status_code=503,
            detail="TTS not loaded. Start server with --audio flag and ensure model files exist",
        )
    
    if request.voice not in TTS_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Available: {', '.join(TTS_VOICES)}",
        )
    
    import soundfile as sf

    if request.stream:
        stream_chunks = _split_text_for_streaming(request.input, request.stream_chunk_chars)
        if not stream_chunks:
            raise HTTPException(status_code=400, detail="Input text is empty")

        def generate_stream() -> Iterator[bytes]:
            first_samples, sample_rate = state.tts_model.create(
                stream_chunks[0],
                voice=request.voice,
                speed=request.speed,
            )
            yield _wav_header(sample_rate)
            pcm_bytes = _float_to_int16_bytes(first_samples)
            for i in range(0, len(pcm_bytes), 8192):
                yield pcm_bytes[i:i + 8192]

            for chunk in stream_chunks[1:]:
                samples, _ = state.tts_model.create(
                    chunk,
                    voice=request.voice,
                    speed=request.speed,
                )
                pcm_bytes = _float_to_int16_bytes(samples)
                for i in range(0, len(pcm_bytes), 8192):
                    yield pcm_bytes[i:i + 8192]

        return StreamingResponse(
            generate_stream(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    # Non-streaming: generate full audio then return WAV
    samples, sample_rate = state.tts_model.create(
        request.input,
        voice=request.voice,
        speed=request.speed,
    )

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    buffer.seek(0)

    return Response(
        content=buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


@app.get("/v1/audio/voices")
async def list_voices() -> dict[str, list[str]]:
    """List available TTS voices.

    Returns:
        Dictionary with list of available voice IDs.
    """
    return {"voices": TTS_VOICES}


def main() -> None:
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(
        description="HTTP API server for MLX models on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default model (Qwen3)
  %(prog)s
  
  # Load all features (vision + audio)
  %(prog)s --vision --audio
  
  # Audio only (STT + TTS)
  %(prog)s --audio
  
  # Custom port
  %(prog)s --port 8080
  
Client examples:
  # Chat completion
  curl -X POST http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
  
  # Speech-to-Text
  curl -X POST http://localhost:8000/v1/audio/transcriptions \\
    -F "file=@recording.wav"
  
  # Text-to-Speech
  curl -X POST http://localhost:8000/v1/audio/speech \\
    -H "Content-Type: application/json" \\
    -d '{"input": "Hello world", "voice": "af_bella"}' \\
    --output speech.wav
        """,
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        choices=["qwen", "gemma"],
        help="Text model to load (default: qwen)",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Also load vision model (Gemma-3)",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Enable audio endpoints (STT with Whisper, TTS with Kokoro)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to run server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    # Set environment variables (persist across uvicorn module reloads)
    os.environ["MLX_MODEL"] = args.model
    os.environ["MLX_VISION"] = "true" if args.vision else ""
    os.environ["MLX_AUDIO"] = "true" if args.audio else ""
    os.environ["MLX_WHISPER_MODEL"] = args.whisper_model
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"API docs available at http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(
        "serve:app",  # Use string to reference module-level app with routes
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
