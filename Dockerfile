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
RUN cd llama.cpp && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j"$(nproc)" \
        --target llama-server llama-cli

###############################################################################
# Stage 2 — Runtime
###############################################################################
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# llama.cpp binaries + their shared libs live here
ENV PATH="/opt/llama:${PATH}"
ENV LD_LIBRARY_PATH="/opt/llama:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git curl wget ca-certificates \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy the whole bin dir so both the executables AND libllama/libggml*.so come along.
COPY --from=builder /app/llama.cpp/build/bin/ /opt/llama/
RUN ldconfig /opt/llama

WORKDIR /workspace

RUN pip3 install --no-cache-dir \
        "huggingface_hub[cli,hf_transfer]" \
        jupyterlab open-webui aider-chat

RUN mkdir -p /workspace/models

COPY start.sh /start.sh
RUN chmod +x /start.sh

# 8910 = llama.cpp API | 3000 = Open WebUI | 8888 = JupyterLab
EXPOSE 8888 8910 3000
ENTRYPOINT ["/start.sh"]
