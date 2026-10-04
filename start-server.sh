#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/bin-linux-vulkan:${LD_LIBRARY_PATH}"

echo "================================================================"
echo "  Local LLM Server (Native Linux • Intel Arc Vulkan • High TPS)"
echo "  Engine: llama.cpp b11385 (Native Linux x86_64)"
echo "  Speculative Acceleration: N-Gram In-Memory Self-Speculation"
echo "  Models Directory: ${SCRIPT_DIR}/Models"
echo "  Web UI: http://127.0.0.1:8080"
echo "  API Endpoint: http://127.0.0.1:8080/v1"
echo "================================================================"
echo ""

"${SCRIPT_DIR}/bin-linux-vulkan/llama-server" \
    --models-dir "${SCRIPT_DIR}/Models" \
    --models-max 1 \
    -ngl 99 \
    -t 6 \
    -c 16384 \
    -np 1 \
    -fa on \
    --spec-type ngram-mod \
    --host 127.0.0.1 \
    --port 8080
