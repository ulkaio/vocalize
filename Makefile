# MLX Model Server Makefile
# Run server in background, survives SSH disconnect

SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PID_FILE := .server.pid
LOG_FILE := server.log
PORT := 8000

.PHONY: help install start stop restart status logs tail clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	uv venv --python python3.12
	uv pip install -U mlx-lm mlx-vlm mlx-whisper
	uv pip install fastapi uvicorn pydantic
	uv pip install kokoro-onnx soundfile
	uv pip install "numpy<2.4"
	@echo "✓ Dependencies installed"

start:  ## Start server in background (TTS only)
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Server already running (PID: $$(cat $(PID_FILE)))"; \
	else \
		echo "Starting server on port $(PORT)..."; \
		nohup $(PYTHON) serve.py --tts --port $(PORT) > $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		sleep 2; \
		if kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
			echo "✓ Server started (PID: $$(cat $(PID_FILE)))"; \
			echo "  Logs: tail -f $(LOG_FILE)"; \
			echo "  URL:  http://localhost:$(PORT)"; \
		else \
			echo "✗ Failed to start. Check $(LOG_FILE)"; \
			rm -f $(PID_FILE); \
		fi \
	fi

start-full:  ## Start with all features (Text + Vision + STT + TTS)
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Server already running (PID: $$(cat $(PID_FILE)))"; \
	else \
		echo "Starting server with all features on port $(PORT)..."; \
		nohup $(PYTHON) serve.py --model gemma --text --vision --stt --tts --port $(PORT) > $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		sleep 2; \
		if kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
			echo "✓ Server started (PID: $$(cat $(PID_FILE)))"; \
			echo "  Logs: tail -f $(LOG_FILE)"; \
			echo "  URL:  http://localhost:$(PORT)"; \
		else \
			echo "✗ Failed to start. Check $(LOG_FILE)"; \
			rm -f $(PID_FILE); \
		fi \
	fi

stop:  ## Stop the server
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			echo "Stopping server (PID: $$PID)..."; \
			kill $$PID; \
			rm -f $(PID_FILE); \
			echo "✓ Server stopped"; \
		else \
			echo "Server not running (stale PID file)"; \
			rm -f $(PID_FILE); \
		fi \
	else \
		echo "No PID file found. Server may not be running."; \
	fi

restart:  ## Restart the server
	@$(MAKE) stop
	@sleep 1
	@$(MAKE) start

status:  ## Check if server is running
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "✓ Server running (PID: $$(cat $(PID_FILE)))"; \
		echo "  URL: http://localhost:$(PORT)"; \
		curl -s http://localhost:$(PORT)/health 2>/dev/null && echo "" || echo "  (not responding yet)"; \
	else \
		echo "✗ Server not running"; \
	fi

logs:  ## Show recent logs
	@if [ -f $(LOG_FILE) ]; then \
		tail -50 $(LOG_FILE); \
	else \
		echo "No log file found"; \
	fi

tail:  ## Follow logs in real-time
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "No log file found"; \
	fi

test-health:  ## Test health endpoint
	curl -s http://localhost:$(PORT)/health | python -m json.tool

test-chat:  ## Test chat completion
	curl -s -X POST http://localhost:$(PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"messages": [{"role": "user", "content": "tell me something interesting"}], "max_tokens": 50}' \
		| python -m json.tool

test-tts:  ## Test text-to-speech
	curl -s -X POST http://localhost:$(PORT)/v1/audio/speech \
		-H "Content-Type: application/json" \
		-d '{"input": "Hello, this is a test", "voice": "af_bella"}' \
		--output test_output.wav
	@echo "✓ Audio saved to test_output.wav"

test-stt:  ## Test speech-to-text (requires test_output.wav)
	@if [ -f test_output.wav ]; then \
		curl -s -X POST http://localhost:$(PORT)/v1/audio/transcriptions \
			-F "file=@test_output.wav" | python -m json.tool; \
	else \
		echo "No test_output.wav found. Run 'make test-tts' first."; \
	fi

clean:  ## Remove logs and temp files
	rm -f $(LOG_FILE) $(PID_FILE) test_output.wav output.wav
	@echo "✓ Cleaned up"
