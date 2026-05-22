#!/bin/bash
set -e

# Trap SIGTERM for clean shutdown
cleanup() {
    echo "Shutting down..."
    kill $LLAMA_PID $WEBUI_PID $JUPYTER_PID 2>/dev/null
    wait $LLAMA_PID 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT

echo "=== Starting RunPod Agentic Coding Template ==="

echo "[1/3] Starting llama.cpp server..."
llama-server \
    --model /workspace/models/qwen3.6-27b-uncensored-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8910 \
    --n-gpu-layers 99 \
    --ctx-size 32768 \
    --threads 8 \
    --api-key local &
LLAMA_PID=$!

echo "Waiting for llama-server to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8910/health > /dev/null 2>&1; then
        echo "llama-server is ready!"
        break
    fi
    sleep 2
done

echo "[2/3] Starting Open WebUI..."
OPENAI_API_BASE=http://localhost:8910/v1 \
OPENAI_API_KEY=local \
WEBUI_AUTH=false \
python3 -m open_webui.serve --host 0.0.0.0 --port 3000 &
WEBUI_PID=$!

echo "[3/3] Starting JupyterLab..."
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True &
JUPYTER_PID=$!

cat > /workspace/aider-start.sh << 'AIDEREOF'
#!/bin/bash
export OPENAI_API_BASE=http://localhost:8910/v1
export OPENAI_API_KEY=local
echo "Starting Aider with Qwen3.6 27B..."
echo "Use /add <file> to include files, then describe what to code."
echo "---"
aider --model openai/qwen3.6-27b
AIDEREOF
chmod +x /workspace/aider-start.sh

cat > /workspace/README.md << 'READMEEOF'
# RunPod Agentic Coding Template

## Interfaces
- **JupyterLab**: http://[pod-id]-8888.proxy.runpod.net
- **Open WebUI (Chat)**: http://[pod-id]-3000.proxy.runpod.net
- **llama.cpp API**: http://[pod-id]-8910.proxy.runpod.net/v1

## Agentic Coding
Open a terminal in JupyterLab and run: `bash /workspace/aider-start.sh`

## Model
HauhauCS Qwen3.6-27B-Uncensored-Balanced Q4_K_M
READMEEOF

echo ""
echo "=== All services started ==="
echo "JupyterLab: http://localhost:8888"
echo "Open WebUI: http://localhost:3000"
echo "API: http://localhost:8910"
echo ""
echo "Agentic coding: bash /workspace/aider-start.sh"
echo ""

# Keep container alive on llama-server
wait $LLAMA_PID
