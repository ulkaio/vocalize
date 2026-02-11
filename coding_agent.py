"""Coding Agent powered by Qwen3-4B on Apple Silicon (MLX).

A tool-calling coding agent that can inspect files, search code, edit files,
and run shell commands in the current workspace.

Usage:
    uv run python coding_agent.py "Find where model loading happens."
    uv run python coding_agent.py -i
"""

import argparse
import ast
import json
import logging
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate as lm_generate
from mlx_lm import load as lm_load
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger("coding_agent")


def _configure_logging(verbose: bool = False) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if verbose:
        handler.setFormatter(
            logging.Formatter("\033[90m[%(levelname)s %(funcName)s]\033[0m %(message)s")
        )
        logger.setLevel(logging.DEBUG)
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.setLevel(logging.WARNING)
    logger.handlers = [handler]


MODEL_PRESETS: dict[str, str] = {
    "qwen": "mlx-community/Qwen3-4B-8bit",
    "gemma": "mlx-community/gemma-3-4b-it-8bit",
}
DEFAULT_MODEL_NAME: str = "qwen"
MODEL_PATH: str = MODEL_PRESETS[DEFAULT_MODEL_NAME]
WORKSPACE_ROOT: Path = Path.cwd().resolve()


def _resolve_workspace_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    if WORKSPACE_ROOT not in resolved.parents and resolved != WORKSPACE_ROOT:
        raise ValueError(f"Path '{path}' is outside workspace root '{WORKSPACE_ROOT}'.")
    return resolved


def _resolve_model_path(model_name: str, model_path_override: str | None) -> str:
    """Resolve final model path from preset + optional explicit override."""
    if model_path_override:
        return model_path_override
    return MODEL_PRESETS[model_name]


def list_files(path: str = ".", max_entries: int = 200) -> str:
    """List files under a workspace path."""
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


def search_in_files(pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Search text in files using ripgrep."""
    try:
        root = _resolve_workspace_path(path)
        cmd = [
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
        return "search_in_files error: 'rg' not found."
    except Exception as exc:
        return f"search_in_files error: {exc}"


def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """Read a line range from a text file."""
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


def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write or append text content to a file in workspace."""
    try:
        file_path = _resolve_workspace_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if mode not in {"overwrite", "append"}:
            return "Invalid mode. Use 'overwrite' or 'append'."
        if mode == "overwrite":
            file_path.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        with file_path.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} bytes to {path}"
    except Exception as exc:
        return f"write_file error: {exc}"


def run_shell_command(command: str, timeout_sec: int = 60) -> str:
    """Run a shell command in workspace with basic safety filters."""
    blocked_tokens = [
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
        parts = [f"exit_code={proc.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_sec} seconds."
    except Exception as exc:
        return f"run_shell_command error: {exc}"


def analyze_code_snippet(code: str, max_lines: int = 120) -> str:
    """Heuristic analysis for inline code snippets when model emits tool_code inspect calls."""
    try:
        lines = code.splitlines()[: max(1, max_lines)]
        patterns = [
            "from_pretrained(",
            "load_model(",
            "lm_load(",
            "torch.load(",
            "AutoModel",
            "AutoTokenizer",
            "pipeline(",
        ]
        matches: list[str] = []
        for i, line in enumerate(lines, start=1):
            if any(token in line for token in patterns):
                matches.append(f"{i}: {line.rstrip()}")
        if matches:
            return "Potential model-loading lines:\n" + "\n".join(matches)
        return "No obvious model-loading call found in provided inline code snippet."
    except Exception as exc:
        return f"analyze_code_snippet error: {exc}"


TOOL_FUNCTIONS: dict[str, Any] = {
    "list_files": list_files,
    "search_in_files": search_in_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_shell_command": run_shell_command,
    "analyze_code_snippet": analyze_code_snippet,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories under a workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root.", "default": "."},
                    "max_entries": {"type": "integer", "description": "Maximum entries to return.", "default": 200},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Search text pattern in files using ripgrep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex/text pattern to search for."},
                    "path": {"type": "string", "description": "Path relative to workspace root.", "default": "."},
                    "max_results": {"type": "integer", "description": "Maximum matched lines to return.", "default": 200},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read specific lines from a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                    "start_line": {"type": "integer", "description": "1-based start line.", "default": 1},
                    "end_line": {"type": "integer", "description": "1-based end line.", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file (overwrite or append).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                    "content": {"type": "string", "description": "Text content to write."},
                    "mode": {"type": "string", "description": "overwrite or append", "default": "overwrite"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command in the workspace and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout_sec": {"type": "integer", "description": "Command timeout in seconds.", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code_snippet",
            "description": "Analyze inline code text and highlight probable model-loading lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Inline source code to inspect."},
                    "max_lines": {"type": "integer", "description": "Max lines to analyze.", "default": 120},
                },
                "required": ["code"],
            },
        },
    },
]

SYSTEM_PROMPT: str = (
    "You are a coding agent with access to workspace tools. "
    "Use tools to inspect code before making claims. "
    "Prefer workspace inspection tools (list_files, search_in_files, read_file) over fabricated examples. "
    "If using tool_code, call only available tools and aliases that map to them. "
    "When asked to implement changes, propose minimal edits and verify with commands when possible. "
    "Do not fabricate tool results. "
    "When finished, provide concise actionable output."
)


class CodingAgent:
    TOOL_CALL_PATTERN: re.Pattern[str] = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL,
    )
    TOOL_CODE_BLOCK_PATTERN: re.Pattern[str] = re.compile(
        r"```tool_code\s*(.*?)\s*```",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        max_tokens: int = 700,
        temperature: float = 0.2,
        max_iterations: int = 8,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose

        _configure_logging(verbose)
        self._model: Any = None
        self._tokenizer: Any = None
        self._sampler: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        print(f"Loading model: {self.model_path}")
        print("Using Apple Silicon GPU (Metal) for local coding agent")
        self._model, self._tokenizer = lm_load(self.model_path)
        self._sampler = make_sampler(temp=self.temperature)
        print("Model loaded.\n")

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        logger.debug("Formatting %d messages", len(messages))
        return self._tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
        )

    def _generate(self, prompt: str) -> str:
        return lm_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )

    def _normalize_tool_call(self, raw_call: Any) -> dict[str, Any] | None:
        """Normalize various tool-call shapes to {'name': str, 'arguments': dict}."""
        if not isinstance(raw_call, dict):
            return None

        # Common structures:
        # {"name": "...", "arguments": {...}}
        # {"function": {"name": "...", "arguments": {...}}}
        # {"tool_name": "...", "parameters": {...}}
        name = raw_call.get("name") or raw_call.get("tool") or raw_call.get("tool_name")
        arguments = raw_call.get("arguments")

        function_obj = raw_call.get("function")
        if isinstance(function_obj, dict):
            name = name or function_obj.get("name")
            arguments = arguments if arguments is not None else function_obj.get("arguments")

        if arguments is None:
            arguments = raw_call.get("args")
        if arguments is None:
            arguments = raw_call.get("parameters")

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                arguments = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        if not isinstance(name, str) or not name.strip():
            return None
        return {"name": name.strip(), "arguments": arguments}

    def _extract_calls_from_json_value(self, value: Any) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        if isinstance(value, dict):
            direct = self._normalize_tool_call(value)
            if direct is not None:
                calls.append(direct)

            for key in ("tool_calls", "calls", "functions"):
                item = value.get(key)
                if isinstance(item, list):
                    for entry in item:
                        normalized = self._normalize_tool_call(entry)
                        if normalized is not None:
                            calls.append(normalized)

        elif isinstance(value, list):
            for entry in value:
                normalized = self._normalize_tool_call(entry)
                if normalized is not None:
                    calls.append(normalized)

        return calls

    def _extract_json_values(self, text: str) -> list[Any]:
        """Extract JSON values (objects/arrays) from free-form text."""
        decoder = json.JSONDecoder()
        values: list[Any] = []
        idx = 0
        length = len(text)

        while idx < length:
            ch = text[idx]
            if ch not in "{[":
                idx += 1
                continue
            try:
                value, end = decoder.raw_decode(text, idx)
                values.append(value)
                idx = end
            except json.JSONDecodeError:
                idx += 1
        return values

    def _dotted_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._dotted_name(node.value)
            if base is None:
                return None
            return f"{base}.{node.attr}"
        return None

    def _literal_from_ast(self, node: ast.AST) -> Any:
        return ast.literal_eval(node)

    def _tool_from_call_node(self, call: ast.Call) -> dict[str, Any] | None:
        fn_name = self._dotted_name(call.func)
        if fn_name is None:
            return None

        positional: list[Any] = []
        for arg in call.args:
            try:
                positional.append(self._literal_from_ast(arg))
            except Exception:
                return None

        keyword: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                keyword[kw.arg] = self._literal_from_ast(kw.value)
            except Exception:
                return None

        # Native tool names.
        if fn_name in TOOL_FUNCTIONS:
            arguments = dict(keyword)
            if positional:
                if fn_name == "list_files" and "path" not in arguments:
                    arguments["path"] = positional[0]
                elif fn_name == "search_in_files" and "pattern" not in arguments:
                    arguments["pattern"] = positional[0]
                elif fn_name == "read_file" and "path" not in arguments:
                    arguments["path"] = positional[0]
                elif fn_name == "write_file":
                    if len(positional) >= 1 and "path" not in arguments:
                        arguments["path"] = positional[0]
                    if len(positional) >= 2 and "content" not in arguments:
                        arguments["content"] = positional[1]
                elif fn_name == "run_shell_command" and "command" not in arguments:
                    arguments["command"] = positional[0]
            return {"name": fn_name, "arguments": arguments}

        # Gemma-style tool_code aliases.
        if fn_name in {"code_editor.inspect", "code_inspector.inspect"}:
            inline_code = keyword.get("code")
            if isinstance(inline_code, str) and inline_code.strip():
                return {"name": "analyze_code_snippet", "arguments": {"code": inline_code}}
            path = keyword.get("path")
            if path is None and positional:
                path = positional[0]
            return {"name": "list_files", "arguments": {"path": path or "."}}
        if fn_name in {
            "code_editor.search",
            "code_editor.grep",
            "code_inspector.search",
            "code_inspector.grep",
        }:
            pattern = keyword.get("pattern") or keyword.get("query")
            if pattern is None and positional:
                pattern = positional[0]
            if not isinstance(pattern, str) or not pattern:
                return None
            args: dict[str, Any] = {"pattern": pattern}
            if "path" in keyword:
                args["path"] = keyword["path"]
            elif len(positional) > 1:
                args["path"] = positional[1]
            return {"name": "search_in_files", "arguments": args}
        if fn_name in {"code_editor.open", "code_editor.read", "code_inspector.open", "code_inspector.read"}:
            path = keyword.get("path")
            if path is None and positional:
                path = positional[0]
            if not isinstance(path, str) or not path:
                return None
            args = {"path": path}
            if "start_line" in keyword:
                args["start_line"] = keyword["start_line"]
            if "end_line" in keyword:
                args["end_line"] = keyword["end_line"]
            return {"name": "read_file", "arguments": args}
        if fn_name in {
            "code_editor.write",
            "code_editor.append",
            "code_inspector.write",
            "code_inspector.append",
        }:
            path = keyword.get("path")
            content = keyword.get("content")
            if path is None and positional:
                path = positional[0]
            if content is None and len(positional) > 1:
                content = positional[1]
            if not isinstance(path, str) or not isinstance(content, str):
                return None
            mode = "append" if fn_name.endswith(".append") else "overwrite"
            return {"name": "write_file", "arguments": {"path": path, "content": content, "mode": mode}}
        if fn_name in {"terminal.run", "bash.run", "shell.run"}:
            command = keyword.get("command")
            if command is None and positional:
                command = positional[0]
            if not isinstance(command, str) or not command:
                return None
            args = {"command": command}
            if "timeout_sec" in keyword:
                args["timeout_sec"] = keyword["timeout_sec"]
            return {"name": "run_shell_command", "arguments": args}

        return None

    def _extract_calls_from_tool_code(self, text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for block in self.TOOL_CODE_BLOCK_PATTERN.findall(text):
            try:
                module = ast.parse(block)
            except SyntaxError:
                continue
            for stmt in module.body:
                expr = stmt.value if isinstance(stmt, ast.Expr) else None
                if not isinstance(expr, ast.Call):
                    continue
                call_node = expr
                # Common pattern: print(tool_call(...))
                if self._dotted_name(call_node.func) == "print" and call_node.args:
                    inner = call_node.args[0]
                    if isinstance(inner, ast.Call):
                        call_node = inner
                converted = self._tool_from_call_node(call_node)
                if converted is not None:
                    calls.append(converted)
        return calls

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add_call(candidate: Any) -> None:
            normalized = self._normalize_tool_call(candidate)
            if normalized is None:
                return
            signature = (normalized["name"], json.dumps(normalized["arguments"], sort_keys=True))
            if signature in seen:
                return
            seen.add(signature)
            calls.append(normalized)

        # 1) Qwen-style XML tool call blocks.
        for match in self.TOOL_CALL_PATTERN.finditer(text):
            raw_json = match.group(1)
            try:
                call = json.loads(raw_json)
                add_call(call)
            except json.JSONDecodeError:
                continue

        # 2) Gemma-style tool_code blocks with python function calls.
        for call in self._extract_calls_from_tool_code(text):
            add_call(call)

        # 3) JSON fenced code blocks.
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        for block in fenced_blocks:
            try:
                value = json.loads(block)
            except json.JSONDecodeError:
                continue
            for call in self._extract_calls_from_json_value(value):
                add_call(call)

        # 4) Free-form JSON objects/arrays embedded in text.
        for value in self._extract_json_values(text):
            for call in self._extract_calls_from_json_value(value):
                add_call(call)

        logger.debug("Parsed %d tool call(s)", len(calls))
        return calls

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            return f"Unknown tool: {name}"
        try:
            return str(func(**arguments))
        except Exception as exc:
            return f"Tool '{name}' error: {exc}"

    def _supports_tool_role_messages(self) -> bool:
        """Whether current model template accepts explicit 'tool' role messages."""
        model = self.model_path.lower()
        return "qwen" in model

    def run(self, user_query: str) -> str:
        self._ensure_loaded()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        start = time.perf_counter()
        for iteration in range(1, self.max_iterations + 1):
            prompt = self._format_prompt(messages)
            response = self._generate(prompt)

            print(f"\n--- Iteration {iteration} ---")
            print(f"Model output:\n{response}")

            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                logger.debug("Completed in %.2fs", time.perf_counter() - start)
                return answer

            messages.append({"role": "assistant", "content": response})
            tool_results: list[tuple[str, dict[str, Any], str]] = []
            for call in tool_calls:
                name = call["name"]
                args = call.get("arguments", {})
                print(f"  Calling tool: {name}({args})")
                result = self._execute_tool(name, args)
                preview = textwrap.shorten(result, width=280, placeholder=" ...")
                print(f"  Result: {preview}")
                tool_results.append((name, args, result))

            if self._supports_tool_role_messages():
                for _, _, result in tool_results:
                    messages.append({"role": "tool", "content": result})
            else:
                # Gemma templates require strict user/assistant alternation.
                # Feed all tool outputs back as one synthetic user message.
                tool_feedback_parts: list[str] = [
                    "Tool results are available below. Use them to continue."
                ]
                for idx, (name, args, result) in enumerate(tool_results, start=1):
                    tool_feedback_parts.append(
                        f"[{idx}] {name}({json.dumps(args, ensure_ascii=True)})\n{result}"
                    )
                messages.append({"role": "user", "content": "\n\n".join(tool_feedback_parts)})

        return "Reached max iterations before producing a final answer."

    def interactive(self) -> None:
        self._ensure_loaded()
        session_id = 1
        print("=" * 64)
        print(f"  Coding Agent ({self.model_path}) interactive mode")
        print(f"  Workspace: {WORKSPACE_ROOT}")
        print(f"  Session: {session_id}")
        print("  Commands: /new, quit")
        print("  Type 'quit' to exit")
        print("=" * 64 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return
            if not user_input:
                continue
            if user_input == "/new":
                session_id += 1
                print(f"\nStarted new session: {session_id}\n")
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                return
            answer = self.run(user_input)
            print(f"\nAgent:\n{answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tool-calling coding agent using Qwen or Gemma via MLX."
    )
    parser.add_argument("query", nargs="?", default=None, help="Coding task or question.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode.")
    parser.add_argument("-t", "--max-tokens", type=int, default=700, help="Max generation tokens per step.")
    parser.add_argument("--temp", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-iter", type=int, default=8, help="Max tool-call iterations.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        default=DEFAULT_MODEL_NAME,
        help="Model preset to use (default: qwen).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model path; overrides --model preset.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose debug logging.")

    args = parser.parse_args()
    resolved_model_path = _resolve_model_path(args.model, args.model_path)
    agent = CodingAgent(
        model_path=resolved_model_path,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        max_iterations=args.max_iter,
        verbose=args.verbose,
    )

    if args.interactive:
        agent.interactive()
        return
    if args.query is None:
        parser.print_help()
        print("\nError: provide a query or use -i for interactive mode")
        sys.exit(1)

    answer = agent.run(args.query)
    print(f"\nFinal Answer:\n{answer}")


if __name__ == "__main__":
    main()
def compile_code(source_code: str) -> str:
    try:
        compiled = compile(source_code, '<string>', 'exec')
        return str(compiled)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Compilation error: {e}"
