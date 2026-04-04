#!/bin/bash
# =============================================================================
# Ensures Ollama is running and Gemma 4 models are loaded.
# =============================================================================
# Ollama serves an OpenAI-compatible API at http://localhost:11434/v1
# No vLLM, no torch — Ollama handles GPU inference natively.
#
# Models:
#   gemma4:26b  — Primary model for council reviews (MoE, 4B active)
#   gemma4:31b  — Lighter alternative
#
# Gemma 4 thinks by default (reasoning traces in separate field).
# =============================================================================

set -e

echo "=== Starting Ollama for Gemma 4 Council ==="

# Check if Ollama server is already running
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Ollama server is already running."
else
    echo "Starting Ollama server..."
    OLLAMA_DEBUG=0 ollama serve > ollama.log 2>&1 &
    echo "Waiting for Ollama to be ready..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            echo "Ollama is ready."
            break
        fi
        sleep 1
    done
fi

echo ""
echo "Available Gemma 4 models:"
ollama list | grep gemma4 || echo "  No gemma4 models found. Run: bash setup_env.sh"

echo ""
echo "OpenAI-compatible API endpoint: http://localhost:11434/v1"
echo ""
echo "Quick test:"
echo "  curl http://localhost:11434/v1/chat/completions -d '{\"model\":\"gemma4:26b\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
echo ""
echo "Ready. Run: python baseline_eval.py  or  python dynamic_council.py"
