"""Streaming TTS CLI for Kokoro (MLX/ONNX).

Streams audio in chunks to stdout or writes a WAV file incrementally.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np


MODEL_DIR = Path(__file__).parent / "models" / "kokoro"
MODEL_FILE = MODEL_DIR / "kokoro-v1.0.fp16-gpu.onnx"
VOICES_FILE = MODEL_DIR / "voices-v1.0.bin"


def _split_text_for_streaming(text: str, target_chars: int) -> list[str]:
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


def _split_text_with_first_chunk(
    text: str,
    first_chars: int,
    rest_chars: int,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    first_chunk: list[str] = []
    first_len = 0
    idx = 0
    while idx < len(sentences):
        sentence = sentences[idx]
        sentence_len = len(sentence)
        if first_chunk and first_len + sentence_len + 1 > first_chars:
            break
        first_chunk.append(sentence)
        first_len += sentence_len + (1 if first_len else 0)
        idx += 1
        if first_len >= first_chars:
            break
    chunks = [" ".join(first_chunk)]
    remaining = " ".join(sentences[idx:])
    if remaining:
        chunks.extend(_split_text_for_streaming(remaining, rest_chars))
    return chunks


def _wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
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


def _patch_wav_header(file_obj, data_bytes: int) -> None:
    riff_size = 36 + data_bytes
    file_obj.seek(4)
    file_obj.write(struct.pack("<I", riff_size))
    file_obj.seek(40)
    file_obj.write(struct.pack("<I", data_bytes))


def _float_to_int16_bytes(samples: np.ndarray) -> bytes:
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _generate_stream(
    kokoro,
    text: str,
    voice: str,
    speed: float,
    stream_chars: int,
    first_chunk_chars: int | None,
) -> Iterator[tuple[bytes, int]]:
    if first_chunk_chars is not None and first_chunk_chars > 0:
        chunks = _split_text_with_first_chunk(text, first_chunk_chars, stream_chars)
    else:
        chunks = _split_text_for_streaming(text, stream_chars)
    if not chunks:
        return iter(())

    first_samples, sample_rate = kokoro.create(
        chunks[0],
        voice=voice,
        speed=speed,
    )
    yield _float_to_int16_bytes(first_samples), sample_rate

    for chunk in chunks[1:]:
        samples, _ = kokoro.create(
            chunk,
            voice=voice,
            speed=speed,
        )
        yield _float_to_int16_bytes(samples), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming TTS for Kokoro (writes WAV incrementally).",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to convert to speech (or read from stdin if omitted)",
    )
    parser.add_argument(
        "-v",
        "--voice",
        default="af_heart",
        help="Kokoro voice ID (default: af_bella)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.25,
        help="Speech speed (0.5 - 2.0, default: 1.25)",
    )
    parser.add_argument(
        "--stream-chars",
        type=int,
        default=220,
        help="Approximate chars per streamed chunk (default: 220)",
    )
    parser.add_argument(
        "--first-chunk-chars",
        type=int,
        default=50,
        help="Smaller first chunk for faster start (default: 50)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output WAV file path. If omitted, streams to stdout.",
    )
    parser.add_argument(
        "--model-path",
        help="Path to Kokoro ONNX model file (defaults to models/kokoro/kokoro-v0_19.onnx)",
    )
    parser.add_argument(
        "--voices-path",
        help="Path to Kokoro voices bin file (defaults to models/kokoro/voices.bin)",
    )
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

    try:
        from kokoro_onnx import Kokoro
    except ImportError as exc:
        print("Error: kokoro-onnx not installed.", file=sys.stderr)
        print("Install with: uv pip install kokoro-onnx soundfile", file=sys.stderr)
        raise SystemExit(1) from exc

    model_path = Path(args.model_path) if args.model_path else MODEL_FILE
    voices_path = Path(args.voices_path) if args.voices_path else VOICES_FILE

    if not model_path.exists() or not voices_path.exists():
        print("Error: Kokoro model files not found.", file=sys.stderr)
        print("Run: python tts.py --list-voices (to trigger download)", file=sys.stderr)
        sys.exit(1)

    load_start = time.perf_counter()
    kokoro = Kokoro(str(model_path), str(voices_path))
    load_seconds = time.perf_counter() - load_start
    print(f"Model load time: {load_seconds:.2f}s", file=sys.stderr)

    if args.output:
        out_path = Path(args.output)
        with out_path.open("wb") as f:
            total_bytes = 0
            sample_rate_written = False
            for pcm_bytes, sample_rate in _generate_stream(
                kokoro,
                text,
                args.voice,
                args.speed,
                args.stream_chars,
                args.first_chunk_chars,
            ):
                if not sample_rate_written:
                    f.write(_wav_header(sample_rate))
                    sample_rate_written = True
                f.write(pcm_bytes)
                total_bytes += len(pcm_bytes)
            _patch_wav_header(f, total_bytes)
        return

    # Stream to stdout
    out = sys.stdout.buffer
    sample_rate_written = False
    for pcm_bytes, sample_rate in _generate_stream(
        kokoro,
        text,
        args.voice,
        args.speed,
        args.stream_chars,
        args.first_chunk_chars,
    ):
        if not sample_rate_written:
            out.write(_wav_header(sample_rate))
            out.flush()
            sample_rate_written = True
        out.write(pcm_bytes)
        out.flush()


if __name__ == "__main__":
    main()
