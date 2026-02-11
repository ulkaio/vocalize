"""Stream an NVIDIA NIM chat completion and report detailed token statistics.

Connects to the NVIDIA Integrate API, streams the response token-by-token,
and prints comprehensive timing and throughput statistics including TTFT,
inter-token latency, tokens per second, and more.
"""

import json
import sys
import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INVOKE_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"
STREAM: bool = True

HEADERS: dict[str, str] = {
    "Authorization": (
        "Bearer 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"
    ),
    "Accept": "text/event-stream" if STREAM else "application/json",
}

PAYLOAD: dict[str, Any] = {
    "model": "moonshotai/kimi-k2.5",
    "messages": [
        {
            "role": "user",
            "content": "Write an advanced load balancing algorithm in Python",
        }
    ],
    "max_tokens": 16384,
    "temperature": 1.00,
    "top_p": 1.00,
    "stream": STREAM,
    "stream_options": {"include_usage": True},  # request token usage in stream
    "chat_template_kwargs": {"thinking": True},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE data line into a JSON dict.

    Args:
        line: A raw line from the SSE stream (already decoded to str).

    Returns:
        Parsed JSON dict, or ``None`` if the line is not a data payload
        (e.g. comments, keep-alives, or the ``[DONE]`` sentinel).
    """
    if not line.startswith("data:"):
        return None

    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return None

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def format_duration(seconds: float) -> str:
    """Return a human-friendly duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string, e.g. ``"1.234 s"`` or ``"123.4 ms"``.
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


# ---------------------------------------------------------------------------
# Main streaming logic
# ---------------------------------------------------------------------------


def stream_response() -> None:
    """Stream a chat completion from NVIDIA NIM and print live statistics."""
    print("=" * 70)
    print(f"Model   : {PAYLOAD['model']}")
    print(f"Prompt  : {PAYLOAD['messages'][0]['content'][:80]}...")
    print("=" * 70)
    print()

    # -- timing bookkeeping --------------------------------------------------
    request_start: float = time.perf_counter()
    first_token_time: float | None = None
    last_token_time: float = request_start
    inter_token_times: list[float] = []

    # -- token bookkeeping ---------------------------------------------------
    generated_tokens: int = 0
    finish_reason: str | None = None
    model_name: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None

    # -- usage reported by API (if available) --------------------------------
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # -- accumulated content -------------------------------------------------
    full_content: str = ""

    # -----------------------------------------------------------------------
    # Send the request
    # -----------------------------------------------------------------------
    try:
        response = requests.post(
            INVOKE_URL,
            headers=HEADERS,
            json=PAYLOAD,
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    time_to_first_byte: float = time.perf_counter() - request_start

    print("--- Response stream begin ---\n")

    # -----------------------------------------------------------------------
    # Iterate over the SSE stream
    # -----------------------------------------------------------------------
    for raw_line in response.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode("utf-8")
        chunk = parse_sse_line(line)
        if chunk is None:
            continue

        now = time.perf_counter()

        # Capture metadata from the first chunk
        if model_name is None:
            model_name = chunk.get("model")
            response_id = chunk.get("id")
            system_fingerprint = chunk.get("system_fingerprint")

        # ----- Handle usage object (final chunk with stream_options) -------
        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

        # ----- Process choices / delta content ------------------------------
        choices = chunk.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")

            # Check finish reason
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

            token_text = content or reasoning
            if token_text:
                # Record timing
                if first_token_time is None:
                    first_token_time = now
                else:
                    inter_token_times.append(now - last_token_time)
                last_token_time = now
                generated_tokens += 1
                full_content += token_text

                # Print token as it arrives (no newline, flush immediately)
                print(token_text, end="", flush=True)

    # End of stream
    request_end: float = time.perf_counter()
    total_duration: float = request_end - request_start

    # -----------------------------------------------------------------------
    # Print statistics
    # -----------------------------------------------------------------------
    print("\n\n--- Response stream end ---\n")
    print("=" * 70)
    print("                      STREAMING STATISTICS")
    print("=" * 70)

    # -- Request metadata ----------------------------------------------------
    print(f"\n{'Response ID':<30}: {response_id or 'N/A'}")
    print(f"{'Model':<30}: {model_name or 'N/A'}")
    print(f"{'System Fingerprint':<30}: {system_fingerprint or 'N/A'}")
    print(f"{'Finish Reason':<30}: {finish_reason or 'N/A'}")
    print(f"{'HTTP Status':<30}: {response.status_code}")

    # -- Timing statistics ---------------------------------------------------
    print(f"\n{'--- Timing ---'}")
    print(f"{'Total Wall Time':<30}: {format_duration(total_duration)}")
    print(f"{'Time to First Byte (TTFB)':<30}: {format_duration(time_to_first_byte)}")

    if first_token_time is not None:
        ttft = first_token_time - request_start
        print(f"{'Time to First Token (TTFT)':<30}: {format_duration(ttft)}")
    else:
        ttft = None
        print(f"{'Time to First Token (TTFT)':<30}: N/A (no tokens received)")

    if first_token_time is not None:
        generation_duration = last_token_time - first_token_time
        print(f"{'Generation Duration':<30}: {format_duration(generation_duration)}")
    else:
        generation_duration = 0.0

    # -- Inter-token latency -------------------------------------------------
    if inter_token_times:
        avg_itl = sum(inter_token_times) / len(inter_token_times)
        min_itl = min(inter_token_times)
        max_itl = max(inter_token_times)
        # Median
        sorted_itl = sorted(inter_token_times)
        mid = len(sorted_itl) // 2
        median_itl = (
            sorted_itl[mid]
            if len(sorted_itl) % 2 == 1
            else (sorted_itl[mid - 1] + sorted_itl[mid]) / 2
        )
        # P95 / P99
        p95_idx = int(len(sorted_itl) * 0.95)
        p99_idx = int(len(sorted_itl) * 0.99)
        p95_itl = sorted_itl[min(p95_idx, len(sorted_itl) - 1)]
        p99_itl = sorted_itl[min(p99_idx, len(sorted_itl) - 1)]

        print(f"\n{'--- Inter-Token Latency ---'}")
        print(f"{'  Average':<30}: {format_duration(avg_itl)}")
        print(f"{'  Median':<30}: {format_duration(median_itl)}")
        print(f"{'  Min':<30}: {format_duration(min_itl)}")
        print(f"{'  Max':<30}: {format_duration(max_itl)}")
        print(f"{'  P95':<30}: {format_duration(p95_itl)}")
        print(f"{'  P99':<30}: {format_duration(p99_itl)}")

    # -- Throughput ----------------------------------------------------------
    print(f"\n{'--- Throughput ---'}")
    print(f"{'Tokens Generated (counted)':<30}: {generated_tokens}")

    if generation_duration > 0 and generated_tokens > 1:
        tps = (generated_tokens - 1) / generation_duration  # exclude first token wait
        print(f"{'Tokens / Second (decode)':<30}: {tps:.2f}")

    if total_duration > 0 and generated_tokens > 0:
        effective_tps = generated_tokens / total_duration
        print(f"{'Tokens / Second (end-to-end)':<30}: {effective_tps:.2f}")

    # -- API-reported usage (if available) -----------------------------------
    if prompt_tokens is not None or completion_tokens is not None:
        print(f"\n{'--- API-Reported Usage ---'}")
        print(f"{'Prompt Tokens':<30}: {prompt_tokens or 'N/A'}")
        print(f"{'Completion Tokens':<30}: {completion_tokens or 'N/A'}")
        print(f"{'Total Tokens':<30}: {total_tokens or 'N/A'}")

    # -- Content stats -------------------------------------------------------
    print(f"\n{'--- Content ---'}")
    print(f"{'Characters Generated':<30}: {len(full_content)}")
    print(f"{'Approx Words':<30}: {len(full_content.split())}")

    print("\n" + "=" * 70)


def non_stream_response() -> None:
    """Send a non-streaming chat completion request and print the result."""
    try:
        response = requests.post(
            INVOKE_URL,
            headers=HEADERS,
            json=PAYLOAD,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        print(json.dumps(data, indent=2))
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if STREAM:
        stream_response()
    else:
        non_stream_response()
