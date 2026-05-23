# syntax=docker/dockerfile:1
###############################################################################
# Stage 1 — Build llama.cpp with CUDA
#
# CUDA_ARCH must match the GPU you actually run on:
#   120 = Blackwell  (RTX PRO 6000, RTX 5090)        <-- default (what you wanted)
#    89 = Ada        (L40S, RTX 6000 Ada, RTX 4090)
#    86 = Ampere     (RTX A6000, A40)   |  80 = A100
# A wrong value here builds no kernels for your card => silent CPU fallback
# (i.e. "it loads but is unusably slow"). This is the #1 gotcha.
###############################################################################
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ARG CUDA_ARCH=120
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential curl wget ca-certificates \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build from master on purpose: the Qwen3.6 hybrid architecture
# (qwen35moe = linear attention + MoE) only landed in llama.cpp recently,
# so pinning an old release would fail to load the model.
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

# NOTE: -DGGML_CUDA=ON  (NOT the old, now-ignored -DLLAMA_CUDA=ON).
#
# The two CMAKE_*_PATH / linker flags below fix the
#   "undefined reference to cuMemGetAllocationGranularity"
# link error: those cuMem* symbols live in the real CUDA *driver*
# (libcuda.so.1), which only exists at RUNTIME (injected by the NVIDIA
# container runtime). At build time only a stub is present, so we point
# the linker at the stub dir and allow the symbols to resolve later.
RUN cd llama.cpp && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    && cmake --build build --config Release -j"$(nproc)" \
        --target llama-server llama-cli

###############################################################################
# Stage 2 — Runtime
###############################################################################
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_XET_HIGH_PERFORMANCE=1
# llama.cpp + each app's venv bin on PATH
ENV PATH="/opt/llama:/opt/webui/bin:/opt/openhands/bin:/opt/tools/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/llama:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        git curl wget ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole bin dir so both the executables AND libllama/libggml*.so come along.
COPY --from=builder /app/llama.cpp/build/bin/ /opt/llama/
RUN ldconfig /opt/llama

WORKDIR /workspace

# Each app gets its own isolated venv to avoid dependency conflicts.
RUN python3 -m venv /opt/webui && /opt/webui/bin/pip install --no-cache-dir open-webui

# Open Hands — browser-based agentic coding UI (like Claude Code but in the browser).
# RUNTIME=local tells it to run code directly in the container instead of
# spinning up a nested Docker sandbox (which isn't available on RunPod).
RUN python3 -m venv /opt/openhands && \
    /opt/openhands/bin/pip install --no-cache-dir openhands-ai

RUN python3 -m venv /opt/tools && /opt/tools/bin/pip install --no-cache-dir \
        jupyterlab "huggingface_hub[cli,hf_transfer]"

RUN mkdir -p /workspace/models /workspace/project

COPY start.sh /start.sh
RUN chmod +x /start.sh

# 8910 = llama.cpp API | 3000 = Open WebUI | 3001 = Open Hands | 8888 = JupyterLab
EXPOSE 8888 8910 3000 3001
ENTRYPOINT ["/start.sh"]
