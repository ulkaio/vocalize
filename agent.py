"""LLM Agent powered by Qwen3-4B on Apple Silicon (MLX).

A tool-calling agent that reasons iteratively: it generates a response,
decides whether to invoke a tool, observes the result, and continues
until it produces a final answer.

## Agent Flow (Mermaid)

```mermaid
flowchart TD
    A([User Query]) --> B[Build Messages<br/>system + tools + history]
    B --> C[Generate Response<br/>Qwen3-4B via MLX]
    C --> D{Tool call in<br/>response?}
    D -- Yes --> E[Parse Tool Call<br/>name + arguments]
    E --> F[Execute Tool]
    F --> G[Append tool result<br/>to conversation]
    G --> C
    D -- No --> H[Extract Final Answer]
    H --> I([Return Answer<br/>to User])

    style A fill:#4CAF50,color:#fff
    style I fill:#2196F3,color:#fff
    style D fill:#FF9800,color:#fff
    style F fill:#9C27B0,color:#fff
```

Usage:
    # One-shot question
    uv run python agent.py "What is 47 * 89 + 12?"

    # Interactive mode
    uv run python agent.py -i

    # With more tokens / higher temperature
    uv run python agent.py "Plan a trip to Tokyo" -t 1024 --temp 0.8
"""

import argparse
import json
import logging
import math
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Optional

from mlx_lm import generate as lm_generate
from mlx_lm import load as lm_load
from mlx_lm.sample_utils import make_sampler

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("agent")


def _configure_logging(verbose: bool = False) -> None:
    """Configure the agent logger.

    Args:
        verbose: When ``True``, set level to DEBUG and use a detailed format.
                 Otherwise only warnings and above are shown.
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
# Model configuration
# ---------------------------------------------------------------------------

MODEL_PATH: str = "mlx-community/Qwen3-4B-8bit"

# ---------------------------------------------------------------------------
# Tools – each tool is a plain function + a JSON-schema descriptor
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Supports basic arithmetic, powers, roots, and common math functions.

    Args:
        expression: A math expression string, e.g. ``"47 * 89 + 12"``.

    Returns:
        The evaluated result as a string, or an error message.
    """
    allowed_names: dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


def get_current_time(timezone_name: str = "UTC") -> str:
    """Return the current date and time.

    Args:
        timezone_name: Currently only ``"UTC"`` is supported.

    Returns:
        An ISO-8601 formatted datetime string.
    """
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def search_knowledge(query: str) -> str:
    """Search a small built-in knowledge base and return matching facts.

    This is a stub that simulates retrieval from a knowledge store.

    Args:
        query: A natural-language search query.

    Returns:
        Matching facts as a newline-separated string, or a not-found message.
    """
    knowledge_base: list[dict[str, str]] = [
        {
            "topic": "python",
            "fact": (
                "Python was created by Guido van Rossum and first released "
                "in 1991. It emphasizes code readability and supports "
                "multiple programming paradigms."
            ),
        },
        {
            "topic": "mlx",
            "fact": (
                "MLX is a machine-learning framework by Apple designed for "
                "efficient training and inference on Apple Silicon. It "
                "provides a NumPy-like API with automatic differentiation."
            ),
        },
        {
            "topic": "apple silicon",
            "fact": (
                "Apple Silicon is Apple's ARM-based chip family (M1-M4). "
                "It features a unified memory architecture with a powerful "
                "GPU accessible through Metal and MLX."
            ),
        },
        {
            "topic": "qwen",
            "fact": (
                "Qwen is a family of large language models developed by "
                "Alibaba Cloud. Qwen3-4B is a 4-billion-parameter model "
                "that supports tool calling and multi-turn conversation."
            ),
        },
        {
            "topic": "rust",
            "fact": (
                "Rust is a systems programming language focused on safety, "
                "concurrency, and performance. It uses an ownership model "
                "to guarantee memory safety without a garbage collector."
            ),
        },
    ]
    query_lower = query.lower()
    matches = [
        entry["fact"]
        for entry in knowledge_base
        if entry["topic"] in query_lower or query_lower in entry["topic"]
    ]
    if matches:
        return "\n".join(matches)
    return f"No results found for '{query}'. Try a different search term."


# ---------------------------------------------------------------------------
# Tool registry – maps name → callable
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_knowledge": search_knowledge,
}

# JSON-schema descriptions passed to the model via the chat template.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression. Supports arithmetic, "
                "powers, sqrt, log, trig functions, pi, and e."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2**10 + sqrt(144)'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "Timezone name (currently only 'UTC').",
                        "default": "UTC",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search a knowledge base for facts about a topic. "
                "Good for questions about Python, MLX, Apple Silicon, Qwen, or Rust."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'python' or 'apple silicon'",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a helpful assistant with access to tools. "
    "When a user asks a question that can benefit from a tool, call the "
    "appropriate tool. You may call multiple tools in sequence. "
    "Once you have enough information, provide a clear, concise final answer. "
    "Do NOT fabricate tool results — always call the tool first."
)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """A tool-calling LLM agent backed by Qwen3-4B via MLX.

    The agent maintains a conversation history, detects ``<tool_call>``
    blocks in the model's output, executes the requested tools, and feeds
    results back until the model produces a final text answer.

    Attributes:
        model_path: HuggingFace model identifier.
        max_tokens: Maximum tokens per generation step.
        temperature: Sampling temperature.
        max_iterations: Safety cap on tool-call rounds.
    """

    TOOL_CALL_PATTERN: re.Pattern[str] = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL,
    )

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_iterations: int = 6,
        verbose: bool = False,
    ) -> None:
        """Initialise the agent.

        Args:
            model_path: HuggingFace model path for MLX.
            max_tokens: Maximum tokens per generation step.
            temperature: Sampling temperature (0.0–1.0).
            max_iterations: Maximum number of tool-call iterations.
            verbose: When ``True``, emit detailed DEBUG logs for every step.
        """
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
        """Lazily load the model, tokenizer, and sampler on first use."""
        if self._model is not None:
            return
        print(f"Loading model: {self.model_path}")
        print("Using Apple Silicon GPU (Metal) 🚀")
        logger.debug("lm_load(%s) starting …", self.model_path)
        load_start = time.perf_counter()
        self._model, self._tokenizer = lm_load(self.model_path)
        self._sampler = make_sampler(temp=self.temperature)
        load_elapsed = time.perf_counter() - load_start
        logger.debug("Model loaded in %.1fs", load_elapsed)
        logger.debug("Sampler temperature: %.2f", self.temperature)
        print("Model loaded.\n")

    # ----- generation helpers ------------------------------------------------

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Apply the chat template to a list of messages, including tool schemas.

        Args:
            messages: Conversation messages (system / user / assistant / tool).

        Returns:
            A single formatted prompt string ready for the model.
        """
        logger.debug(
            "Messages (%d) being sent to chat template:", len(messages)
        )
        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            preview = textwrap.shorten(content, width=200, placeholder=" …")
            logger.debug("  [%d] role=%-9s | %s", idx, role, preview)

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
        )

        logger.debug(
            "Formatted prompt length: %d chars", len(prompt)
        )
        logger.debug(
            "Formatted prompt:\n%s", prompt
        )
        # if len(prompt) > 500:
        #     logger.debug(
        #         "Formatted prompt:\n…%s", prompt
        #     )
        return prompt

    def _generate(self, prompt: str) -> str:
        """Run a single generation step.

        Args:
            prompt: The fully formatted prompt string.

        Returns:
            The raw model output string.
        """
        logger.debug("Calling lm_generate (max_tokens=%d) …", self.max_tokens)
        gen_start = time.perf_counter()
        response = lm_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
        gen_elapsed = time.perf_counter() - gen_start
        logger.debug("Generation completed in %.2fs", gen_elapsed)
        logger.debug(
            "Raw model output (%d chars):\n%s", len(response), response
        )
        return response

    # ----- tool-call parsing & execution -------------------------------------

    def _parse_tool_calls(
        self, text: str
    ) -> list[dict[str, Any]]:
        """Extract tool calls from the model output.

        Qwen3 wraps tool calls in ``<tool_call>...</tool_call>`` XML tags
        containing a JSON object with ``name`` and ``arguments`` keys.

        Args:
            text: Raw model output.

        Returns:
            A list of parsed tool-call dicts, each with ``name`` and ``arguments``.
        """
        logger.debug("Scanning model output for <tool_call> blocks …")
        raw_matches = list(self.TOOL_CALL_PATTERN.finditer(text))
        logger.debug("Regex matches found: %d", len(raw_matches))

        calls: list[dict[str, Any]] = []
        for i, match in enumerate(raw_matches):
            raw_json = match.group(1)
            logger.debug("  Match %d raw JSON: %s", i, raw_json.strip())
            try:
                call = json.loads(raw_json)
                if "name" in call:
                    calls.append(call)
                    logger.debug(
                        "  Match %d parsed → tool=%s  args=%s",
                        i,
                        call["name"],
                        json.dumps(call.get("arguments", {})),
                    )
                else:
                    logger.debug("  Match %d skipped (no 'name' key)", i)
            except json.JSONDecodeError as exc:
                logger.debug("  Match %d JSON parse error: %s", i, exc)
                continue

        if not calls:
            logger.debug("No tool calls detected — treating as final answer.")
        else:
            logger.debug("Total valid tool calls: %d", len(calls))
        return calls

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments.

        Args:
            name: The tool function name.
            arguments: Keyword arguments to pass to the tool.

        Returns:
            The tool's string result, or an error message.
        """
        logger.debug("Executing tool '%s' with args: %s", name, arguments)
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            logger.debug("Tool '%s' not found in registry!", name)
            return f"Unknown tool: {name}"
        try:
            exec_start = time.perf_counter()
            result = func(**arguments)
            exec_elapsed = time.perf_counter() - exec_start
            logger.debug(
                "Tool '%s' returned in %.4fs: %s", name, exec_elapsed, result
            )
            return result
        except Exception as exc:
            logger.debug("Tool '%s' raised exception: %s", name, exc)
            return f"Tool '{name}' error: {exc}"

    # ----- main agent loop ---------------------------------------------------

    def run(self, user_query: str) -> str:
        """Run the agent loop for a single user query.

        The loop generates a response, checks for tool calls, executes them,
        appends results to the conversation, and repeats until the model
        produces a final answer (no tool calls) or the iteration cap is hit.

        Args:
            user_query: The user's natural-language question or request.

        Returns:
            The agent's final text answer.
        """
        self._ensure_loaded()

        logger.debug("=" * 60)
        logger.debug("AGENT RUN START")
        logger.debug("  User query: %s", user_query)
        logger.debug("  Max iterations: %d", self.max_iterations)
        logger.debug("  Max tokens/step: %d", self.max_tokens)
        logger.debug("  Temperature: %.2f", self.temperature)
        logger.debug("=" * 60)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        logger.debug(
            "Initial messages: system (%d chars) + user (%d chars)",
            len(SYSTEM_PROMPT),
            len(user_query),
        )

        run_start = time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            logger.debug("-" * 50)
            logger.debug("ITERATION %d / %d", iteration, self.max_iterations)
            logger.debug("-" * 50)

            prompt = self._format_prompt(messages)
            response = self._generate(prompt)

            print(f"\n--- Iteration {iteration} ---")
            print(f"Model output:\n{response}")

            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No tool call → final answer
                # Strip any leftover thinking tags from Qwen3
                answer = re.sub(
                    r"<think>.*?</think>", "", response, flags=re.DOTALL
                ).strip()
                total_elapsed = time.perf_counter() - run_start
                logger.debug("=" * 60)
                logger.debug("AGENT RUN COMPLETE")
                logger.debug("  Iterations used: %d", iteration)
                logger.debug("  Total wall time: %.2fs", total_elapsed)
                logger.debug(
                    "  Final answer (%d chars): %s",
                    len(answer),
                    textwrap.shorten(answer, width=300, placeholder=" …"),
                )
                logger.debug("=" * 60)
                return answer

            # Append the assistant's raw response (with tool calls)
            messages.append({"role": "assistant", "content": response})
            logger.debug(
                "Appended assistant message (%d chars) to conversation",
                len(response),
            )

            # Execute each tool call and feed results back
            for call_idx, call in enumerate(tool_calls):
                name = call["name"]
                args = call.get("arguments", {})
                print(f"  🔧 Calling tool: {name}({args})")
                result = self._execute_tool(name, args)
                print(f"  📎 Result: {result}")
                messages.append({"role": "tool", "content": result})
                logger.debug(
                    "Appended tool result [%d] for '%s' (%d chars) to conversation",
                    call_idx,
                    name,
                    len(result),
                )

            logger.debug(
                "Conversation now has %d messages", len(messages)
            )

        total_elapsed = time.perf_counter() - run_start
        logger.debug("Max iterations reached after %.2fs", total_elapsed)
        return "I reached the maximum number of reasoning steps. Here is what I have so far."

    def interactive(self) -> None:
        """Run the agent in an interactive REPL loop.

        The user can type queries and receive answers. Conversation state
        is **not** carried across queries (each query is independent).
        Type ``quit``, ``exit``, or ``q`` to stop.
        """
        self._ensure_loaded()

        print("=" * 55)
        print("  Agent (Qwen3-4B) — interactive mode")
        print("  Tools: calculator, get_current_time, search_knowledge")
        print("  Type 'quit' to exit")
        print("=" * 55 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            start = time.perf_counter()
            answer = self.run(user_input)
            elapsed = time.perf_counter() - start

            print(f"\nAgent: {answer}")
            print(f"  ({elapsed:.1f}s)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the agent CLI."""
    parser = argparse.ArgumentParser(
        description="LLM Agent with tool calling — Qwen3-4B on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is 47 * 89 + 12?"
  %(prog)s "What time is it?"
  %(prog)s "Tell me about Rust programming language"
  %(prog)s -i                         # Interactive mode
  %(prog)s "What is sqrt(144) + 2^10?" -t 256
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The question or task for the agent",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "-t",
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per generation step (default: 512)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=6,
        help="Max tool-call iterations (default: 6)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging (shows prompts, parsing, tool I/O)",
    )

    args = parser.parse_args()

    agent = Agent(
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

    try:
        answer = agent.run(args.query)
        print(f"\nFinal Answer:\n{answer}")
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
