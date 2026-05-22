# Stage 1: Build llama.cpp with CUDA
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggerganov/llama.cpp.git

RUN cd llama.cpp && mkdir build && cd build && \
    cmake .. \
    -DLLAMA_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DLLAMA_CURL=ON && \
    cmake --build . --config Release -j$(nproc) --target llama-server

# Stage 2: Runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl wget ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

COPY --from=builder /app/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /workspace

RUN pip3 install --no-cache-dir jupyterlab open-webui aider-chat

RUN mkdir -p /workspace/models

RUN wget -O /workspace/models/qwen3.6-27b-balanced-q4_k_p.gguf \
    "https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced/resolve/main/Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q4_K_P.gguf"

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8888 8910 3000

ENTRYPOINT ["/start.sh"]
