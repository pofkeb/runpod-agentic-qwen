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
API_KEY="${API_KEY:-local}"
# Thinking ON by default (best for agentic loops). Set ENABLE_THINKING=false to disable.
ENABLE_THINKING="${ENABLE_THINKING:-true}"

MODELS_DIR=/workspace/models
mkdir -p "$MODELS_DIR"

cleanup() {
    echo "Shutting down..."
    kill $LLAMA_PID $WEBUI_PID $JUPYTER_PID 2>/dev/null || true
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

# Thinking toggle (Qwen3.6 dropped the /think /no_think switches — use template kwarg)
THINK_ARGS=()
if [ "$ENABLE_THINKING" = "false" ]; then
    THINK_ARGS=(--chat-template-kwargs '{"enable_thinking": false}')
fi

echo "======================================================================"
echo " Model    : $REPO"
echo " File     : $MODEL_FILE"
echo " Served as: $SERVED_NAME   (use this as the model name in API/aider)"
echo " Context  : $CTX_SIZE | Thinking: $ENABLE_THINKING"
echo "======================================================================"

###############################################################################
# [1/3] llama.cpp server
###############################################################################
echo "[1/3] Starting llama.cpp server on :8910 ..."
llama-server \
    --model "$MODEL_FILE" \
    --alias "$SERVED_NAME" \
    --host 0.0.0.0 --port 8910 \
    --n-gpu-layers "$GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
    --flash-attn on \
    --api-key "$API_KEY" \
    --jinja \
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
# [2/3] Open WebUI
###############################################################################
echo "[2/3] Starting Open WebUI on :3000 ..."
OPENAI_API_BASE_URL="http://localhost:8910/v1" \
OPENAI_API_KEY="$API_KEY" \
WEBUI_AUTH=false \
python3 -m open_webui.serve --host 0.0.0.0 --port 3000 &
WEBUI_PID=$!

###############################################################################
# [3/3] JupyterLab
###############################################################################
echo "[3/3] Starting JupyterLab on :8888 ..."
jupyter lab \
    --ip=0.0.0.0 --port=8888 \
    --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True &
JUPYTER_PID=$!

###############################################################################
# Helper: aider launcher (recommended coding sampler baked in)
###############################################################################
cat > /workspace/aider-start.sh << AIDEREOF
#!/bin/bash
export OPENAI_API_BASE=http://localhost:8910/v1
export OPENAI_API_KEY=$API_KEY
echo "Starting Aider against '$SERVED_NAME' ..."
echo "Use /add <file> to include files, then describe the change."
echo "---"
# temp 0.6 / top-p 0.95 are the Qwen-recommended precise-coding settings.
aider --model openai/$SERVED_NAME \
      --temperature 0.6 \
      --no-show-model-warnings
AIDEREOF
chmod +x /workspace/aider-start.sh

cat > /workspace/README.md << READMEEOF
# RunPod Agentic Coding Template

## Currently serving: $SERVED_NAME
File: $MODEL_FILE

## Interfaces
- JupyterLab : http://[pod-id]-8888.proxy.runpod.net
- Open WebUI : http://[pod-id]-3000.proxy.runpod.net
- API (OpenAI): http://[pod-id]-8910.proxy.runpod.net/v1   (api-key: $API_KEY)

## Agentic coding
    bash /workspace/aider-start.sh

## Switch / resize the model (set as pod env vars, then restart)
- MODEL_CHOICE=qwen-35b      # default, HauhauCS 35B-A3B uncensored
- MODEL_CHOICE=qwen-coder    # Huihui Qwen3-Coder-30B abliterated (A/B test)
- MODEL_CHOICE=qwen-35b-base # non-uncensored baseline (A/B vs Aggressive for drift)
- QWEN_QUANT=Q8_K_P          # 44GB, needs 96GB card | Q6_K_P=31GB for 48GB cards
- ENABLE_THINKING=false      # faster, no chain-of-thought

Recommended sampler for coding: temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.
READMEEOF

echo ""
echo "=== All services started ==="
echo "JupyterLab : http://localhost:8888"
echo "Open WebUI : http://localhost:3000"
echo "API        : http://localhost:8910/v1   (key: $API_KEY)"
echo "Agentic    : bash /workspace/aider-start.sh"
echo ""

wait $LLAMA_PID
