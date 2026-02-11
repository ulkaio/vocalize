"""Coding agent built with the OpenAI Agents SDK and NVIDIA-hosted Kimi K2.5.

Uses ``OpenAIChatCompletionsModel`` to route inference through the NVIDIA NIM
endpoint (``https://integrate.api.nvidia.com/v1``) while sending traces to the
OpenAI dashboard via a separate API key.

Environment variables
---------------------
NVIDIA_API_KEY : str
    Bearer token for the NVIDIA Integrate API.
OPENAI_API_KEY : str
    OpenAI platform key used **only** for tracing.

Usage
-----
    # One-shot query
    uv run python openai_agentsdk/coding_agent.py "List all Python files"

    # Interactive REPL
    uv run python openai_agentsdk/coding_agent.py -i
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_export_api_key,
    trace,
)
from agents.items import (
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.stream_events import RunItemStreamEvent, StreamEvent
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger("openai_agentsdk.coding_agent")


def _configure_logging(verbose: bool = False) -> None:
    """Set up the module logger.

    Args:
        verbose: When ``True``, emit DEBUG-level messages with a detailed
            format.  Otherwise only warnings and above are shown.
    """
    handler = logging.StreamHandler(sys.stderr)
    if verbose:
        handler.setFormatter(
            logging.Formatter(
                "\033[90m[%(levelname)s %(funcName)s]\033[0m %(message)s"
            )
        )
        logger.setLevel(logging.DEBUG)
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.setLevel(logging.WARNING)
    logger.handlers = [handler]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
# MODEL_NAME: str = "qwen/qwen3-next-80b-a3b-instruct"
# MODEL_NAME: str = "z-ai/glm4.7"
MODEL_NAME: str = "moonshotai/kimi-k2.5"
WORKSPACE_ROOT: Path = Path.cwd().resolve()
TRACE_INCLUDE_SENSITIVE_DATA: bool = True

SYSTEM_PROMPT: str = (
    "You are a coding agent with access to workspace tools. "
    "Use tools to inspect code before making claims. "
    "Prefer workspace inspection tools (list_files, search_in_files, "
    "read_file) over fabricated examples. "
    "When asked to implement changes, propose minimal edits and verify "
    "with commands when possible. "
    "Do not fabricate tool results. "
    "When finished, provide concise actionable output."
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_workspace_path(path: str) -> Path:
    """Resolve *path* relative to ``WORKSPACE_ROOT`` and validate it.

    Args:
        path: A relative or absolute filesystem path.

    Returns:
        The resolved ``Path`` object.

    Raises:
        ValueError: If the resolved path escapes the workspace root.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    if WORKSPACE_ROOT not in resolved.parents and resolved != WORKSPACE_ROOT:
        raise ValueError(
            f"Path '{path}' is outside workspace root '{WORKSPACE_ROOT}'."
        )
    return resolved


# ---------------------------------------------------------------------------
# Tools — decorated with @function_tool for the Agents SDK
# ---------------------------------------------------------------------------


@function_tool
def list_files(path: str = ".", max_entries: int = 200) -> str:
    """List files and directories under a workspace path.

    Args:
        path: Path relative to the workspace root.
        max_entries: Maximum number of entries to return.

    Returns:
        A newline-separated listing of entries, or an error message.
    """
    try:
        root = _resolve_workspace_path(path)
        if not root.exists():
            return f"Path not found: {path}"
        if root.is_file():
            rel = root.relative_to(WORKSPACE_ROOT)
            return str(rel)
        entries: list[str] = []
        for entry in sorted(root.rglob("*")):
            if len(entries) >= max_entries:
                break
            if entry.name.startswith(".git"):
                continue
            rel = entry.relative_to(WORKSPACE_ROOT)
            suffix = "/" if entry.is_dir() else ""
            entries.append(f"{rel}{suffix}")
        if not entries:
            return f"No entries found under: {path}"
        out = "\n".join(entries)
        if len(entries) >= max_entries:
            out += f"\n... truncated to {max_entries} entries."
        return out
    except Exception as exc:
        return f"list_files error: {exc}"


@function_tool
def search_in_files(
    pattern: str, path: str = ".", max_results: int = 200
) -> str:
    """Search for a text pattern in workspace files using ripgrep.

    Args:
        pattern: Regex or text pattern to search for.
        path: Path relative to the workspace root.
        max_results: Maximum number of matched lines to return.

    Returns:
        Matching lines with file paths and line numbers, or an error
        message.
    """
    try:
        root = _resolve_workspace_path(path)
        cmd: list[str] = [
            "rg",
            "-n",
            "--no-heading",
            "--line-number",
            "--color",
            "never",
            "--max-count",
            str(max_results),
            pattern,
            str(root),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if proc.returncode not in (0, 1):
            return f"search_in_files command failed: {proc.stderr.strip()}"
        output = proc.stdout.strip()
        return output if output else f"No matches for pattern: {pattern}"
    except FileNotFoundError:
        return "search_in_files error: 'rg' (ripgrep) not found."
    except Exception as exc:
        return f"search_in_files error: {exc}"


@function_tool
def read_file(
    path: str, start_line: int = 1, end_line: int = 200
) -> str:
    """Read a range of lines from a UTF-8 text file in the workspace.

    Args:
        path: File path relative to the workspace root.
        start_line: 1-based start line number.
        end_line: 1-based end line number.

    Returns:
        The numbered lines, or an error message.
    """
    try:
        file_path = _resolve_workspace_path(path)
        if not file_path.exists():
            return f"File not found: {path}"
        if not file_path.is_file():
            return f"Not a file: {path}"
        start = max(1, start_line)
        end = max(start, end_line)
        lines = file_path.read_text(encoding="utf-8").splitlines()
        selected = lines[start - 1 : end]
        if not selected:
            return f"No content in requested range {start}-{end}."
        numbered = [f"{i + start}: {line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)
    except UnicodeDecodeError:
        return f"File is not UTF-8 text: {path}"
    except Exception as exc:
        return f"read_file error: {exc}"


@function_tool
def write_file(
    path: str, content: str, mode: str = "overwrite"
) -> str:
    """Write or append text content to a file in the workspace.

    Args:
        path: File path relative to the workspace root.
        content: The text content to write.
        mode: ``"overwrite"`` to replace the file, ``"append"`` to add to it.

    Returns:
        A confirmation message, or an error message.
    """
    try:
        file_path = _resolve_workspace_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if mode not in {"overwrite", "append"}:
            return "Invalid mode. Use 'overwrite' or 'append'."
        if mode == "overwrite":
            file_path.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        with file_path.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return f"Appended {len(content)} bytes to {path}"
    except Exception as exc:
        return f"write_file error: {exc}"


@function_tool
def run_shell_command(command: str, timeout_sec: int = 60) -> str:
    """Run a shell command inside the workspace directory.

    Potentially destructive commands (``rm -rf /``, ``mkfs``, etc.) are
    blocked by a simple keyword filter.

    Args:
        command: The shell command string to execute.
        timeout_sec: Maximum seconds before the command is killed.

    Returns:
        The exit code, stdout, and stderr of the command.
    """
    blocked_tokens: list[str] = [
        "rm -rf /",
        "mkfs",
        "dd if=",
        "shutdown",
        "reboot",
        "poweroff",
        "git reset --hard",
        "git checkout --",
    ]
    lower = command.lower()
    if any(token in lower for token in blocked_tokens):
        return f"Blocked potentially destructive command: {command}"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=max(1, timeout_sec),
            check=False,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        parts: list[str] = [f"exit_code={proc.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_sec} seconds."
    except Exception as exc:
        return f"run_shell_command error: {exc}"


# ---------------------------------------------------------------------------
# Client & model setup
# ---------------------------------------------------------------------------


def _build_client() -> AsyncOpenAI:
    """Create an ``AsyncOpenAI`` client pointed at the NVIDIA NIM endpoint.

    Returns:
        A configured ``AsyncOpenAI`` instance.

    Raises:
        SystemExit: If the ``NVIDIA_API_KEY`` environment variable is not set.
    """
    api_key: str = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print(
            "Error: NVIDIA_API_KEY environment variable is required.",
            file=sys.stderr,
        )
        sys.exit(1)
    return AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def _setup_tracing() -> None:
    """Configure OpenAI tracing if an ``OPENAI_API_KEY`` is available.

    When the key is present the traces are exported to the OpenAI dashboard
    at ``platform.openai.com/traces``.  If the key is absent tracing is
    silently skipped (no error).
    """
    openai_key: str = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        set_tracing_export_api_key(openai_key)
        _patch_tracing_export_usage_keys()
        logger.warning(
            "OpenAI tracing enabled. Trace payload will include agent input/output."
        )
        logger.debug("Tracing enabled — exporting to OpenAI dashboard.")
    else:
        logger.debug(
            "OPENAI_API_KEY not set; tracing will not be exported."
        )


def _patch_tracing_export_usage_keys() -> None:
    """Apply a local compatibility patch for trace usage fields.

    Some tracing ingest endpoints reject newer usage fields such as
    ``span_data.usage.total_tokens`` and token detail objects.
    We remove unsupported keys from the exporter sanitization allowlist so
    payloads remain accepted while preserving basic input/output token counts.
    """
    try:
        from agents.tracing.processors import default_exporter

        exporter = default_exporter()
        allowed = getattr(exporter, "_OPENAI_TRACING_ALLOWED_USAGE_KEYS", None)
        unsupported_usage_keys: frozenset[str] = frozenset(
            {"total_tokens", "input_tokens_details", "output_tokens_details"}
        )
        if isinstance(allowed, frozenset):
            patched = frozenset(
                key for key in allowed if key not in unsupported_usage_keys
            )
            setattr(exporter, "_OPENAI_TRACING_ALLOWED_USAGE_KEYS", patched)
            removed = sorted(set(allowed) - set(patched))
            if removed:
                logger.debug(
                    "Applied tracing usage compatibility patch (removed: %s).",
                    ", ".join(removed),
                )
    except Exception as exc:
        logger.debug("Tracing compatibility patch skipped: %s", exc)


def _build_agent(client: AsyncOpenAI, model_name: str = MODEL_NAME) -> Agent:
    """Construct the coding ``Agent`` with all tools wired up.

    Args:
        client: The ``AsyncOpenAI`` client for NVIDIA NIM.
        model_name: The model identifier to use for inference.

    Returns:
        A fully configured ``Agent`` instance.
    """
    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    )
    return Agent(
        name="CodingAgent",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=[
            list_files,
            search_in_files,
            read_file,
            write_file,
            run_shell_command,
        ],
        model_settings=ModelSettings(
            temperature=1.0,
            max_tokens=16384,
        ),
    )


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


async def run_once(agent: Agent, query: str) -> str:
    """Execute a single agent query and return the final output.

    Args:
        agent: The ``Agent`` instance to run.
        query: The user's natural-language question or request.

    Returns:
        The agent's final text answer.
    """
    logger.debug(
        "Starting agent run (trace_include_sensitive_data=%s).",
        TRACE_INCLUDE_SENSITIVE_DATA,
    )
    with trace("Coding Agent Run"):
        result = await Runner.run(
            agent,
            input=query,
            run_config=RunConfig(
                trace_include_sensitive_data=TRACE_INCLUDE_SENSITIVE_DATA,
            ),
        )
    return result.final_output


def _truncate_text(value: str, limit: int = 400) -> str:
    """Return a shortened single-line representation of text.

    Args:
        value: The input text to normalize.
        limit: Maximum number of characters to retain.

    Returns:
        The normalized and truncated text.
    """
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]} ...[truncated]"


def _safe_json_preview(raw_value: Any, limit: int = 400) -> str:
    """Serialize *raw_value* as JSON when possible for debug printing.

    Args:
        raw_value: Arbitrary object to format for console output.
        limit: Maximum rendered length in characters.

    Returns:
        A compact string preview of the value.
    """
    try:
        text = json.dumps(raw_value, ensure_ascii=True)
    except (TypeError, ValueError):
        text = str(raw_value)
    return _truncate_text(text, limit=limit)


def _extract_message_text(item: MessageOutputItem) -> str:
    """Extract assistant output text from a message run item.

    Args:
        item: A message output item emitted by the SDK.

    Returns:
        Concatenated message text, if any.
    """
    content: Any = getattr(item.raw_item, "content", [])
    text_parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def _format_stream_event(event: StreamEvent) -> str | None:
    """Format a stream event as a concise debug line.

    Args:
        event: A stream event produced by ``RunResultStreaming.stream_events``.

    Returns:
        A printable debug string, or ``None`` when the event should be skipped.
    """
    if not isinstance(event, RunItemStreamEvent):
        return None

    if event.name == "tool_called" and isinstance(event.item, ToolCallItem):
        raw_item = event.item.raw_item
        tool_name: str = getattr(raw_item, "name", "<unknown_tool>")
        arguments: Any = getattr(raw_item, "arguments", "")
        return (
            f"[tool_called] {tool_name} args="
            f"{_safe_json_preview(arguments)}"
        )

    if event.name == "tool_output" and isinstance(event.item, ToolCallOutputItem):
        output = _safe_json_preview(event.item.output)
        return f"[tool_output] {output}"

    if (
        event.name == "message_output_created"
        and isinstance(event.item, MessageOutputItem)
    ):
        text = _extract_message_text(event.item)
        if not text:
            return "[assistant_output] <empty>"
        return f"[assistant_output] {_truncate_text(text)}"

    if event.name == "reasoning_item_created" and isinstance(
        event.item, ReasoningItem
    ):
        return "[reasoning] model reasoning item created"

    return None


async def run_once_with_event_debug(agent: Agent, query: str) -> str:
    """Execute one run and print incremental agent I/O events.

    Args:
        agent: The ``Agent`` instance to run.
        query: The user query string.

    Returns:
        The final output text from the completed run.
    """
    logger.debug(
        "Starting streamed agent run (trace_include_sensitive_data=%s).",
        TRACE_INCLUDE_SENSITIVE_DATA,
    )
    with trace("Coding Agent Run"):
        streamed = Runner.run_streamed(
            agent,
            input=query,
            run_config=RunConfig(
                trace_include_sensitive_data=TRACE_INCLUDE_SENSITIVE_DATA,
            ),
        )
        async for event in streamed.stream_events():
            line = _format_stream_event(event)
            if line is not None:
                print(line)
        return streamed.final_output


async def run_interactive(
    client: AsyncOpenAI, model_name: str, show_events: bool = False
) -> None:
    """Run the agent in an interactive REPL.

    The user types queries and receives answers.  Type ``quit``, ``exit``,
    or ``q`` to stop.  Type ``/new`` to start a fresh session.

    Args:
        client: The configured ``AsyncOpenAI`` client.
        model_name: Initial model identifier for the interactive session.
        show_events: When ``True``, stream and print intermediate event logs.
    """
    session_id: int = 1
    current_model_name: str = model_name
    agent: Agent = _build_agent(client, model_name=current_model_name)
    print("=" * 64)
    print(f"  Coding Agent ({current_model_name}) — interactive mode")
    print(f"  Inference : NVIDIA NIM ({NVIDIA_BASE_URL})")
    print(f"  Workspace : {WORKSPACE_ROOT}")
    print(f"  Session   : {session_id}")
    print("  Commands  : /new, /model <name>, quit")
    print("=" * 64 + "\n")

    while True:
        try:
            user_input: str = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_input:
            continue
        if user_input == "/new":
            session_id += 1
            print(f"\nStarted new session: {session_id}\n")
            continue
        if user_input.startswith("/model"):
            raw_model_name: str = user_input.removeprefix("/model").strip()
            if not raw_model_name:
                print(
                    "\nUsage: /model <model_name>\n"
                    f"Current model: {current_model_name}\n"
                )
                continue

            current_model_name = raw_model_name
            agent = _build_agent(client, model_name=current_model_name)
            print(f"\nSwitched model to: {current_model_name}\n")
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return

        if show_events:
            print()
            answer = await run_once_with_event_debug(agent, user_input)
        else:
            answer = await run_once(agent, user_input)
        print(f"\nAgent:\n{answer}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the coding agent."""
    parser = argparse.ArgumentParser(
        description=(
            "Coding agent using the OpenAI Agents SDK with "
            "NVIDIA-hosted Kimi K2.5 inference and OpenAI tracing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "List all Python files in the workspace"
  %(prog)s "Find where model loading happens"
  %(prog)s --model moonshotai/kimi-k2.5 "Find where model loading happens"
  %(prog)s --show-events "Find where model loading happens"
  %(prog)s -i                    # Interactive mode
  %(prog)s -i --model z-ai/glm4.7
  %(prog)s -i --show-events      # Interactive + event stream
  %(prog)s -i -v                 # Interactive + verbose logging
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="A coding question or task for the agent.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive REPL mode.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=(
            "Model identifier for NVIDIA NIM inference. "
            f"Default: {MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help=(
            "Print intermediate agent events (tool calls/outputs and "
            "assistant message chunks)."
        ),
    )

    args: argparse.Namespace = parser.parse_args()
    _configure_logging(args.verbose)

    # ---- setup ----
    _setup_tracing()
    client: AsyncOpenAI = _build_client()

    # ---- dispatch ----
    if args.interactive:
        asyncio.run(
            run_interactive(
                client,
                model_name=args.model,
                show_events=args.show_events,
            )
        )
        return

    if args.query is None:
        parser.print_help()
        print(
            "\nError: provide a query or use -i for interactive mode",
            file=sys.stderr,
        )
        sys.exit(1)

    agent: Agent = _build_agent(client, model_name=args.model)
    if args.show_events:
        answer = asyncio.run(run_once_with_event_debug(agent, args.query))
    else:
        answer = asyncio.run(run_once(agent, args.query))
    print(f"\nFinal Answer:\n{answer}")


if __name__ == "__main__":
    main()
