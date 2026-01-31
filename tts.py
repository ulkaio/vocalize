"""Text-to-Speech using Kokoro on Apple Silicon.

Kokoro is a lightweight, fast TTS model with natural-sounding voices.

Usage:
    # Generate speech from text
    python tts.py "Hello, welcome to Kokoro TTS!"
    
    # Specify voice and output file
    python tts.py "Hello world" -v af_bella -o output.wav
    
    # List available voices
    python tts.py --list-voices
"""

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


# Model files
MODEL_DIR = Path(__file__).parent / "models" / "kokoro"
MODEL_FILE = "kokoro-v0_19.onnx"
VOICES_FILE = "voices.bin"

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"

# Available voices (Kokoro voice packs)
# Format: language_name (e.g., af = American Female, am = American Male)
VOICES = {
    # American English
    "af_bella": "American Female - Bella (default)",
    "af_sarah": "American Female - Sarah",
    "af_nicole": "American Female - Nicole",
    "af_sky": "American Female - Sky",
    "am_adam": "American Male - Adam",
    "am_michael": "American Male - Michael",
    # British English
    "bf_emma": "British Female - Emma",
    "bf_isabella": "British Female - Isabella",
    "bm_george": "British Male - George",
    "bm_lewis": "British Male - Lewis",
}

DEFAULT_VOICE = "af_bella"
SAMPLE_RATE = 24000  # Kokoro outputs 24kHz audio


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indicator.

    Args:
        url: URL to download from.
        dest: Destination file path.
    """
    print(f"Downloading {dest.name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(count: int, block_size: int, total_size: int) -> None:
        percent = min(100, count * block_size * 100 // total_size)
        print(f"\r  Progress: {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print(f"\n  ✓ Downloaded {dest.name}")


def ensure_model_files() -> tuple[Path, Path]:
    """Ensure model files are downloaded.

    Returns:
        Tuple of (model_path, voices_path).
    """
    model_path = MODEL_DIR / MODEL_FILE
    voices_path = MODEL_DIR / VOICES_FILE
    
    if not model_path.exists():
        print("Model file not found. Downloading (~85MB)...")
        download_file(MODEL_URL, model_path)
    
    if not voices_path.exists():
        print("Voices file not found. Downloading (~300MB)...")
        download_file(VOICES_URL, voices_path)
    
    return model_path, voices_path


def list_voices() -> None:
    """Print all available voices."""
    print("\nAvailable Kokoro Voices:")
    print("-" * 50)
    for voice_id, description in VOICES.items():
        default_marker = " (default)" if voice_id == DEFAULT_VOICE else ""
        print(f"  {voice_id:15} - {description}{default_marker}")
    print()


def generate_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    output_path: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Generate speech from text using Kokoro TTS.

    Args:
        text: The text to convert to speech.
        voice: Voice ID to use (e.g., 'af_bella', 'am_adam').
        output_path: Optional path to save the audio file.
        speed: Speech speed multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast).

    Returns:
        Tuple of (audio_samples, sample_rate).

    Raises:
        ValueError: If voice is not found.
        ImportError: If kokoro-onnx is not installed.
    """
    try:
        from kokoro_onnx import Kokoro
    except ImportError as e:
        print("Error: kokoro-onnx not installed.")
        print("Install with: uv pip install kokoro-onnx soundfile")
        raise ImportError("kokoro-onnx is required for TTS") from e

    if voice not in VOICES:
        available = ", ".join(VOICES.keys())
        raise ValueError(f"Voice '{voice}' not found. Available: {available}")

    # Ensure model files are downloaded
    model_path, voices_path = ensure_model_files()

    print(f"Loading Kokoro TTS model...")
    print(f"Voice: {voice} ({VOICES[voice]})")
    print("Using Apple Silicon GPU (Metal) 🚀")
    print("-" * 50)

    # Initialize Kokoro with model and voices file
    kokoro = Kokoro(str(model_path), str(voices_path))

    # Generate audio
    print(f"Generating speech for: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=speed,
    )

    # Save to file if path provided
    if output_path:
        output_file = Path(output_path)
        sf.write(output_file, samples, sample_rate)
        print(f"✓ Audio saved to: {output_file}")
        print(f"  Duration: {len(samples) / sample_rate:.2f}s")
        print(f"  Sample rate: {sample_rate} Hz")

    return samples, sample_rate


def generate_speech_simple(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
) -> bytes:
    """Generate speech and return as WAV bytes (for HTTP API).

    Args:
        text: The text to convert to speech.
        voice: Voice ID to use.
        speed: Speech speed multiplier.

    Returns:
        WAV audio data as bytes.
    """
    import io

    samples, sample_rate = generate_speech(text, voice=voice, speed=speed)

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    buffer.seek(0)

    return buffer.read()


def main() -> None:
    """Main entry point for TTS script."""
    parser = argparse.ArgumentParser(
        description="Text-to-Speech using Kokoro on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s "Hello, welcome to Kokoro TTS!"
  
  # Save to specific file
  %(prog)s "Hello world" -o greeting.wav
  
  # Use different voice
  %(prog)s "Good morning!" -v am_michael
  
  # Adjust speed (0.5 = slow, 2.0 = fast)
  %(prog)s "This is slower speech" --speed 0.8
  
  # List all voices
  %(prog)s --list-voices

Voices:
  af_*  - American Female (bella, sarah, nicole, sky)
  am_*  - American Male (adam, michael)
  bf_*  - British Female (emma, isabella)
  bm_*  - British Male (george, lewis)
        """,
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to convert to speech",
    )
    parser.add_argument(
        "-v", "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice to use (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "-o", "--output",
        default="output.wav",
        help="Output audio file path (default: output.wav)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed: 0.5=slow, 1.0=normal, 2.0=fast (default: 1.0)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voices",
    )

    args = parser.parse_args()

    # List voices and exit
    if args.list_voices:
        list_voices()
        return

    # Check for text input
    if args.text is None:
        parser.print_help()
        print("\nError: Please provide text to convert to speech")
        sys.exit(1)

    try:
        generate_speech(
            text=args.text,
            voice=args.voice,
            output_path=args.output,
            speed=args.speed,
        )
        print("\n✓ TTS generation complete!")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError:
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

