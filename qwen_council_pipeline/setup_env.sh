#!/bin/bash
# =============================================================================
# Environment Setup for Gemma 4 Council Pipeline via Ollama
# =============================================================================
# Ollama handles model serving natively — no vLLM, no torch, no triton.
# It provides an OpenAI-compatible API at http://localhost:11434/v1
# =============================================================================

set -e

echo "=== Gemma 4 Council Pipeline Setup ==="
echo "Architecture: $(uname -m)"
echo ""

# Step 1: Check Ollama is installed and running
echo "[1/3] Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "ERROR: Ollama not found. Install it from https://ollama.com"
    exit 1
fi
echo "  Ollama version: $(ollama --version)"

# Check if Ollama server is reachable
if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "  Ollama server: running"
else
    echo "  Ollama server: not reachable. Starting..."
    ollama serve &
    sleep 3
fi

# Step 2: Pull the Gemma 4 models
echo ""
echo "[2/3] Pulling Gemma 4 models (this may take a while on first run)..."
echo "  Pulling gemma4:26b (MoE, ~16GB)..."
ollama pull gemma4:26b
echo "  Pulling gemma4:31b (dense, ~7GB)..."
ollama pull gemma4:31b

# Step 3: Install Python dependencies (lightweight — no torch needed)
echo ""
echo "[3/3] Installing Python dependencies..."
pip install openai openreview-py marker-pdf

echo ""
echo "=== Setup Complete ==="
echo "Models available:"
ollama list | grep gemma4
echo ""
echo "Run:  bash start_srv.sh   (starts Ollama if not running)"
echo "Then: python baseline_eval.py"
echo "Or:   python dynamic_council.py"
