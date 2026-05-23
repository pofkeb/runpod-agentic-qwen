#!/bin/bash
set -e

###############################################################################
# Config — override any of these as RunPod environment variables.
###############################################################################
# Which model llama-server serves:
#   qwen-35b       -> Qwen3.6-35B-A3B uncensored (HauhauCS, Aggressive)  [default]
#   qwen-coder     -> Huihui Qwen3-Coder-30B-A3B abliterated (A/B candidate)
#   qwen-35b-base  -> Qwen3.6-35B-A3B NON-uncensored baseline (for A/B vs Aggressive)
MODEL_CHOICE="${MODEL_CHOICE:-qwen-35b}"

# Quant for the 35B model:
#   Q8_K_P (44GB) -> needs the 96GB RTX PRO 6000 for full 262K context  [default]
#   Q6_K_P (31GB) / Q5_K_P (28GB) -> use these on a 48GB card (A6000/L40S)
QWEN_QUANT="${QWEN_QUANT:-Q8_K_P}"
# Quant for the coder model (mradermacher imatrix naming, e.g. i1-Q6_K, i1-Q4_K_M)
CODER_QUANT="${CODER_QUANT:-i1-Q6_K}"
# Quant for the non-uncensored baseline (unsloth GGUF naming, e.g. Q6_K, Q8_0, Q4_K_M)
BASE_QUANT="${BASE_QUANT:-Q6_K}"

CTX_SIZE="${CTX_SIZE:-262144}"       # native max; drop to 131072 if you want lighter
GPU_LAYERS="${GPU_LAYERS:-99}"       # 99 = offload everything to GPU
THREADS="${THREADS:-16}"

# API_KEY is optional. If set (e.g. API_KEY=mysecret in RunPod env vars),
# llama-server and all services will require that key.
# If left unset or empty, no auth is required — anyone with the URL can use it.
API_KEY="${API_KEY:-}"

# Thinking ON by default (best for agentic loops). Set ENABLE_THINKING=false to disable.
ENABLE_THINKING="${ENABLE_THINKING:-true}"

MODELS_DIR=/workspace/models
mkdir -p "$MODELS_DIR"

cleanup() {
    echo "Shutting down..."
    kill $LLAMA_PID $WEBUI_PID $OPENHANDS_PID $JUPYTER_PID 2>/dev/null || true
    wait $LLAMA_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

###############################################################################
# Resolve model repo + download pattern
###############################################################################
case "$MODEL_CHOICE" in
  qwen-35b)
    REPO="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
    PATTERN="*${QWEN_QUANT}.gguf"
    SERVED_NAME="qwen3.6-35b-a3b"
    TARGET_DIR="$MODELS_DIR/qwen35"
    ;;
  qwen-coder)
    REPO="mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF"
    PATTERN="*${CODER_QUANT}*.gguf"
    SERVED_NAME="qwen3-coder-30b"
    TARGET_DIR="$MODELS_DIR/qwencoder"
    ;;
  qwen-35b-base)
    REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
    PATTERN="*${BASE_QUANT}*.gguf"
    SERVED_NAME="qwen3.6-35b-a3b-base"
    TARGET_DIR="$MODELS_DIR/qwen35base"
    ;;
  *)
    echo "ERROR: unknown MODEL_CHOICE='$MODEL_CHOICE' (use qwen-35b, qwen-coder, or qwen-35b-base)"
    exit 1 ;;
esac

###############################################################################
# Download (resumable, skipped if a .gguf is already on the volume)
###############################################################################
mkdir -p "$TARGET_DIR"
if ! ls "$TARGET_DIR"/*.gguf >/dev/null 2>&1; then
    echo "=== Downloading $REPO  (pattern: $PATTERN) ==="
    hf download "$REPO" --include "$PATTERN" --local-dir "$TARGET_DIR"
else
    echo "=== Model already on volume in $TARGET_DIR — skipping download ==="
fi

# Resolve the GGUF to load (if split into shards, point at shard 00001 — llama.cpp
# auto-loads the rest).
MODEL_FILE="$(ls "$TARGET_DIR"/*-00001-of-*.gguf 2>/dev/null | head -1 || true)"
[ -z "$MODEL_FILE" ] && MODEL_FILE="$(ls "$TARGET_DIR"/*.gguf 2>/dev/null | head -1 || true)"
if [ -z "$MODEL_FILE" ]; then
    echo "ERROR: no .gguf found in $TARGET_DIR."
    echo "       Check your QWEN_QUANT / CODER_QUANT pattern matches a real file."
    exit 1
fi

# Thinking toggle
THINK_ARGS=()
if [ "$ENABLE_THINKING" = "false" ]; then
    THINK_ARGS=(--chat-template-kwargs '{"enable_thinking": false}')
fi

# API key args — only added if API_KEY is non-empty
APIKEY_ARGS=()
if [ -n "$API_KEY" ]; then
    APIKEY_ARGS=(--api-key "$API_KEY")
    AUTH_NOTE="key: $API_KEY"
else
    AUTH_NOTE="no auth (open)"
fi

echo "======================================================================"
echo " Model    : $REPO"
echo " File     : $MODEL_FILE"
echo " Served as: $SERVED_NAME"
echo " Context  : $CTX_SIZE | Thinking: $ENABLE_THINKING"
echo " Auth     : $AUTH_NOTE"
echo "======================================================================"

###############################################################################
# [1/4] llama.cpp server
###############################################################################
echo "[1/4] Starting llama.cpp server on :8910 ..."
llama-server \
    --model "$MODEL_FILE" \
    --alias "$SERVED_NAME" \
    --host 0.0.0.0 --port 8910 \
    --n-gpu-layers "$GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
    --flash-attn on \
    --jinja \
    "${APIKEY_ARGS[@]}" \
    "${THINK_ARGS[@]}" &
LLAMA_PID=$!

echo "Waiting for llama-server to be ready (model load can take a minute)..."
for i in $(seq 1 180); do
    if curl -s "http://localhost:8910/health" >/dev/null 2>&1; then
        echo "llama-server is ready!"
        break
    fi
    sleep 2
done

###############################################################################
# [2/4] Open WebUI  (chat interface)
###############################################################################
echo "[2/4] Starting Open WebUI on :3000 ..."
OPENAI_API_BASE_URL="http://localhost:8910/v1" \
OPENAI_API_KEY="${API_KEY:-none}" \
OPENAI_API_KEYS="${API_KEY:-none}" \
WEBUI_AUTH=false \
open-webui serve --host 0.0.0.0 --port 3000 &
WEBUI_PID=$!

###############################################################################
# [3/4] Open Hands  (agentic coding UI — like Claude Code in the browser)
###############################################################################
echo "[3/4] Starting Open Hands on :3001 ..."
WORKSPACE_BASE=/workspace/project \
RUNTIME=local \
LLM_MODEL="openai/${SERVED_NAME}" \
LLM_BASE_URL="http://localhost:8910/v1" \
LLM_API_KEY="${API_KEY:-none}" \
/opt/openhands/bin/python -m openhands.server.listen \
    --host 0.0.0.0 --port 3001 &
OPENHANDS_PID=$!

###############################################################################
# [4/4] JupyterLab
###############################################################################
echo "[4/4] Starting JupyterLab on :8888 ..."
jupyter lab \
    --ip=0.0.0.0 --port=8888 \
    --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True &
JUPYTER_PID=$!

cat > /workspace/README.md << READMEEOF
# RunPod Agentic Coding Template

## Currently serving: $SERVED_NAME
File: $MODEL_FILE

## Auth: $AUTH_NOTE
Set API_KEY in RunPod env vars to enable auth, leave unset for open access.

## Interfaces
- Open Hands (agentic coding) : http://[pod-id]-3001.proxy.runpod.net  <-- start here
- Open WebUI  (chat)          : http://[pod-id]-3000.proxy.runpod.net
- llama-ui    (raw API/chat)  : http://[pod-id]-8910.proxy.runpod.net
- JupyterLab                  : http://[pod-id]-8888.proxy.runpod.net

## Open Hands usage
Open the :3001 URL — it will already be pointed at your local model.
Your project files live in /workspace/project — mount a network volume
there to persist work across pod restarts.

## Switch / resize the model (set as pod env vars, then restart)
- MODEL_CHOICE=qwen-35b      # default, HauhauCS 35B-A3B uncensored
- MODEL_CHOICE=qwen-coder    # Huihui Qwen3-Coder-30B abliterated
- MODEL_CHOICE=qwen-35b-base # non-uncensored baseline
- QWEN_QUANT=Q8_K_P          # 44GB, needs 96GB card | Q6_K_P=31GB for 48GB cards
- ENABLE_THINKING=false      # faster, no chain-of-thought

Recommended sampler for coding: temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.
READMEEOF

echo ""
echo "=== All services started ==="
echo "Open Hands : http://localhost:3001  <-- agentic coding"
echo "Open WebUI : http://localhost:3000  (chat)"
echo "llama-ui   : http://localhost:8910"
echo "JupyterLab : http://localhost:8888"
echo "Auth       : $AUTH_NOTE"
echo ""

wait $LLAMA_PID
