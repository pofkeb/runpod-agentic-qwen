FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    git checkout $(git describe --tags --abbrev=0)

RUN cd llama.cpp && mkdir build && cd build && \
    cmake .. \
    -DLLAMA_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DLLAMA_CURL=ON && \
    cmake --build . --config Release -j$(nproc) --target llama-server

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

RUN wget -O /workspace/models/qwen3.6-27b-uncensored-q4_k_m.gguf \
    "https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced-GGUF/resolve/main/qwen3.6-27b-uncensored-hauhaucs-balanced-q4_k_m.gguf"

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8888 8910 3000

ENTRYPOINT ["/start.sh"]