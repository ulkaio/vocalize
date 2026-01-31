"""Speech-to-text transcription using MLX Whisper (GPU-accelerated on Apple Silicon)."""

import argparse
import sys
from pathlib import Path

import mlx_whisper


# Whisper model mapping (memory efficient)
WHISPER_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",      # ~75 MB
    "base": "mlx-community/whisper-base-mlx",      # ~140 MB
    "small": "mlx-community/whisper-small-mlx",    # ~466 MB
    "medium": "mlx-community/whisper-medium-mlx",  # ~1.5 GB
    "large": "mlx-community/whisper-large-v3-mlx", # ~2.9 GB
}


def transcribe_with_whisper(audio_path: str, model_name: str) -> dict:
    """Transcribe audio using MLX Whisper.

    Args:
        audio_path: Path to the audio file.
        model_name: MLX Whisper model name or HuggingFace repo.

    Returns:
        Dictionary with transcription results including 'text' key.
    """
    resolved_model = WHISPER_MODELS.get(model_name, model_name)
    print(f"Loading Whisper model: {resolved_model}")
    print("Using Apple Silicon GPU (Metal) 🚀")

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=resolved_model,
    )
    return result


def transcribe_audio(
    audio_path: str,
    model: str = "base",
) -> dict:
    """Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.).
        model: Model size/name for Whisper.

    Returns:
        Dictionary with transcription results including 'text' key.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    return transcribe_with_whisper(str(audio_file), model)


def main() -> None:
    """Main entry point for the transcription script."""
    parser = argparse.ArgumentParser(
        description="Speech-to-text transcription using MLX Whisper (Apple Silicon GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav                # Whisper base model (default)
  %(prog)s audio.wav -m small       # Whisper small model
  %(prog)s audio.wav -m large       # Whisper large model

Whisper models:
  tiny   (~75 MB)   - Fastest, lower accuracy
  base   (~140 MB)  - Good balance (default)
  small  (~466 MB)  - Better accuracy
  medium (~1.5 GB)  - High accuracy
  large  (~2.9 GB)  - Best accuracy
        """,
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to audio file (default: recordings/recording_20260124_165433.wav)",
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        help="Model name/size (default: 'base')",
    )

    args = parser.parse_args()

    # Set default audio path
    if args.audio is None:
        audio_path = Path(__file__).parent / "recordings" / "recording_20260124_165433.wav"
    else:
        audio_path = Path(args.audio)

    print(f"Audio file: {audio_path}")
    print("-" * 50)

    try:
        result = transcribe_audio(str(audio_path), args.model)
        print("\nTranscription:")
        print(result["text"])
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
