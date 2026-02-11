# OpenAI Agents SDK — Coding Agent

A tool-calling coding agent built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), powered by **Kimi K2.5** inference on [NVIDIA NIM](https://build.nvidia.com/) and traced via the [OpenAI Traces dashboard](https://platform.openai.com/traces).

## Architecture

| Component        | Provider                                          |
| ---------------- | ------------------------------------------------- |
| Agent framework  | OpenAI Agents SDK (`openai-agents`)               |
| LLM inference    | NVIDIA NIM — `moonshotai/kimi-k2.5`               |
| Tracing          | OpenAI platform (`platform.openai.com/traces`)    |

Inference requests go to `https://integrate.api.nvidia.com/v1` using the `OpenAIChatCompletionsModel` wrapper, while trace telemetry is exported to OpenAI using a separate API key.

## Setup

### 1. Install dependencies

```bash
pip install openai-agents
# or, if using the project requirements:
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
# Required — NVIDIA NIM bearer token
export NVIDIA_API_KEY="nvapi-..."

# Optional — enables tracing on the OpenAI dashboard
export OPENAI_API_KEY="sk-..."
```

## Usage

### One-shot query

```bash
python openai_agentsdk/coding_agent.py "List all Python files in the workspace"
python openai_agentsdk/coding_agent.py "Find where model loading happens"
```

### Interactive REPL

```bash
python openai_agentsdk/coding_agent.py -i
```

Inside the REPL, type `/new` to start a fresh session or `quit` to exit.

### Verbose logging

```bash
python openai_agentsdk/coding_agent.py -i -v
```

## Available Tools

| Tool                | Description                                         |
| ------------------- | --------------------------------------------------- |
| `list_files`        | List files and directories under a workspace path   |
| `search_in_files`   | Search text patterns using ripgrep (`rg`)           |
| `read_file`         | Read a range of lines from a UTF-8 text file        |
| `write_file`        | Write or append text content to a file              |
| `run_shell_command`  | Run a shell command with basic safety filters       |

## Tracing

When `OPENAI_API_KEY` is set, every agent run automatically produces a trace viewable at <https://platform.openai.com/traces>. If the key is absent tracing is silently skipped — inference still works via the NVIDIA endpoint.
