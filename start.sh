#!/bin/bash
# RHAIIS CPU Quickstart — One-click model serving
# Pulls the Red Hat AI Inference Server, serves your chosen model, runs your first call.

set -e

IMAGE="${IMAGE:-registry.redhat.io/rhaii-early-access/vllm-cpu-rhel9:3.5.0-ea.2-1782965184}"
CACHE_DIR="${CACHE_DIR:-$HOME/rhaii-cache}"
CONTAINER="inference-server"
PORT=8000

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}▸${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}!${NC} $1"; }
fail()  { echo -e "${RED}✗${NC} $1"; exit 1; }
header(){ echo -e "\n${BOLD}$1${NC}"; }

# --- Model selection ---
if [ -z "$MODEL" ]; then
    header "Which model do you want to serve?"
    echo
    echo -e "  ${BOLD}1)${NC} Granite 2B  — lightweight, runs on 16 GB RAM (good for getting started)"
    echo -e "  ${BOLD}2)${NC} Qwen 7B    — higher quality, runs on 32 GB RAM (matches the RHAIIS 3.5 blog)"
    echo
    read -rp "  Enter 1 or 2 [default: 1]: " MODEL_CHOICE
    MODEL_CHOICE="${MODEL_CHOICE:-1}"

    case "$MODEL_CHOICE" in
        2)
            MODEL="Qwen/Qwen2.5-7B-Instruct"
            ;;
        *)
            MODEL="ibm-granite/granite-3.3-2b-instruct"
            ;;
    esac
fi

# --- Auto-detect model size and set parameters ---
MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
if echo "$MODEL_LOWER" | grep -qE '14b|15b|16b'; then
    KVCACHE_SPACE="${KVCACHE_SPACE:-10}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    SHM_SIZE="${SHM_SIZE:-8g}"
    MIN_RAM_GB=56
elif echo "$MODEL_LOWER" | grep -qE '7b|8b|9b|10b|11b|12b|13b'; then
    KVCACHE_SPACE="${KVCACHE_SPACE:-10}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    SHM_SIZE="${SHM_SIZE:-8g}"
    MIN_RAM_GB=28
else
    KVCACHE_SPACE="${KVCACHE_SPACE:-4}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
    SHM_SIZE="${SHM_SIZE:-4g}"
    MIN_RAM_GB=14
fi

ok "Model: $MODEL"
info "KV cache: ${KVCACHE_SPACE} GB | Max context: ${MAX_MODEL_LEN} tokens | Shared memory: ${SHM_SIZE}"

# --- Pre-flight checks ---
header "Checking requirements..."

# Container runtime
if command -v podman &>/dev/null; then
    RUNTIME=podman
elif command -v docker &>/dev/null; then
    RUNTIME=docker
else
    fail "podman or docker is required. Install one and try again."
fi
ok "Container runtime: $RUNTIME"

# Memory
if [ -f /proc/meminfo ]; then
    MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
else
    MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
fi

if [ -z "$MEM_GB" ]; then
    warn "Could not detect memory. This model needs ${MIN_RAM_GB}+ GB RAM."
elif [ "$MEM_GB" -lt "$MIN_RAM_GB" ]; then
    fail "Found ${MEM_GB} GB RAM. ${MODEL} needs ${MIN_RAM_GB} GB minimum. Try a smaller model or set MODEL= to override."
else
    ok "Memory: ${MEM_GB} GB (minimum ${MIN_RAM_GB} GB for this model)"
fi

# Architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    warn "Architecture: $ARCH — this image is built for x86_64. It may not work."
else
    ok "Architecture: $ARCH"
fi

# OS check — warn on macOS
if [ "$(uname -s)" = "Darwin" ]; then
    warn "macOS detected. The RHAIIS container requires Linux (RHEL, Fedora, Ubuntu). It will not run natively on macOS."
fi

# curl + jq
command -v curl &>/dev/null || fail "curl is required."
command -v jq &>/dev/null || fail "jq is required. Install it: sudo dnf install jq (RHEL/Fedora) or sudo apt install jq (Ubuntu)."
ok "curl and jq available"

# --- Model cache directory ---
mkdir -p "$CACHE_DIR"
if [ -d "$CACHE_DIR/hub" ] && find "$CACHE_DIR/hub" -maxdepth 2 -name "*.safetensors" -o -name "*.bin" 2>/dev/null | head -1 | grep -q .; then
    ok "Model cache: $CACHE_DIR (cached weights found — startup will be faster)"
else
    ok "Model cache: $CACHE_DIR (first run will download model weights)"
fi

# --- Clean up any previous run ---
if $RUNTIME ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
    info "Removing previous $CONTAINER..."
    $RUNTIME stop "$CONTAINER" &>/dev/null || true
    $RUNTIME rm "$CONTAINER" &>/dev/null || true
fi

# --- Registry login ---
header "Checking registry access..."

if ! $RUNTIME pull --quiet "$IMAGE" &>/dev/null 2>&1; then
    warn "Need to log in to registry.redhat.io"
    echo "  Enter your Red Hat Customer Portal credentials (free at access.redhat.com):"
    $RUNTIME login registry.redhat.io || fail "Registry login failed."
fi
ok "Registry access confirmed"

# --- Hugging Face token ---
header "Checking Hugging Face access..."

if [ -z "$HF_TOKEN" ]; then
    echo -e "  The model weights are on Hugging Face. Enter your token"
    echo -e "  (free at ${BLUE}huggingface.co/settings/tokens${NC}):"
    read -rsp "  HF_TOKEN: " HF_TOKEN
    echo
    if [ -z "$HF_TOKEN" ]; then
        fail "HF_TOKEN is required to download the model."
    fi
fi
ok "Hugging Face token set"

# --- Build runtime-specific flags ---
PODMAN_FLAGS=""
if [ "$RUNTIME" = "podman" ]; then
    PODMAN_FLAGS="--security-opt=label=disable --userns=keep-id:uid=1001"
fi

# --- Start the server ---
header "Starting the Red Hat AI Inference Server..."

info "Pulling container image (first time may take a few minutes)..."
$RUNTIME pull "$IMAGE" 2>&1 | tail -1

info "Starting server with $MODEL..."
# shellcheck disable=SC2086
$RUNTIME run -d --name "$CONTAINER" \
  -p "${PORT}:8000" \
  --shm-size="$SHM_SIZE" \
  $PODMAN_FLAGS \
  -v "${CACHE_DIR}:/opt/app-root/src/.cache:Z" \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "VLLM_CPU_KVCACHE_SPACE=$KVCACHE_SPACE" \
  "$IMAGE" \
  --model "$MODEL" \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --max-model-len "$MAX_MODEL_LEN" &>/dev/null

ok "Container started"

# --- Wait for model to load ---
header "Loading model (this takes 2-10 minutes on first run)..."

SECONDS=0
while true; do
    if $RUNTIME logs "$CONTAINER" 2>&1 | grep -q "Uvicorn running"; then
        if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
            ok "Server ready in ${SECONDS}s"
            break
        fi
    fi

    if $RUNTIME logs "$CONTAINER" 2>&1 | grep -q "RuntimeError\|Error.*memory\|Failed"; then
        echo
        fail "Server failed to start. Check logs: $RUNTIME logs $CONTAINER"
    fi

    if ! $RUNTIME ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
        echo
        fail "Container exited unexpectedly. Check logs: $RUNTIME logs $CONTAINER"
    fi

    if [ $SECONDS -gt 900 ]; then
        fail "Timed out after 15 minutes. Check logs: $RUNTIME logs $CONTAINER"
    fi

    if [ $((SECONDS % 30)) -eq 0 ] && [ $SECONDS -gt 0 ]; then
        info "Still loading... (${SECONDS}s)"
    fi
    sleep 5
done

# --- Verify ---
header "Verifying the API..."

HEALTH=$(curl -sf "http://localhost:${PORT}/health" 2>/dev/null && echo "healthy" || echo "unhealthy")
if [ "$HEALTH" = "healthy" ]; then
    ok "Health endpoint: healthy"
else
    fail "Health endpoint not responding. Check logs: $RUNTIME logs $CONTAINER"
fi

MODELS=$(curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null)
if echo "$MODELS" | jq -e '.data[0].id' &>/dev/null; then
    MODEL_ID=$(echo "$MODELS" | jq -r '.data[0].id')
    ok "Model serving: $MODEL_ID"
else
    fail "API not responding. Check logs: $RUNTIME logs $CONTAINER"
fi

# --- First call ---
header "Making your first inference call..."
echo

RESPONSE=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What are three advantages of running AI models on-premises instead of using cloud APIs? Be concise.\"}],
    \"max_tokens\": 150
  }")

CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content')
TOKENS=$(echo "$RESPONSE" | jq -r '.usage.total_tokens')

echo -e "${BOLD}Question:${NC} What are three advantages of running AI on-premises?"
echo
echo -e "${BOLD}Answer:${NC}"
echo "$CONTENT"
echo
echo -e "${BOLD}Tokens used:${NC} $TOKENS (cost on this server: \$0.00)"

# --- Classification demo ---
header "Bonus: classifying text..."

POS=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Classify as positive, negative, or neutral. One word only.\n\nText: The new product exceeded all expectations.\"}],
    \"max_tokens\": 5
  }" | jq -r '.choices[0].message.content')

NEG=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Classify as positive, negative, or neutral. One word only.\n\nText: The deployment failed and data was lost.\"}],
    \"max_tokens\": 5
  }" | jq -r '.choices[0].message.content')

echo -e "  \"The new product exceeded all expectations.\" → ${GREEN}${POS}${NC}"
echo -e "  \"The deployment failed and data was lost.\"    → ${RED}${NEG}${NC}"

# --- Summary ---
header "Done!"
echo
echo -e "  The Red Hat AI Inference Server is running at ${BOLD}http://localhost:${PORT}${NC}"
echo -e "  Model: ${BOLD}$MODEL${NC}"
echo -e "  API: ${BOLD}OpenAI-compatible${NC} (/v1/chat/completions)"
echo -e "  Cache: ${BOLD}$CACHE_DIR${NC} (weights persist across restarts)"
echo
echo "  Try your own prompt:"
echo
echo "    curl -s http://localhost:${PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"YOUR QUESTION HERE\"}], \"max_tokens\": 150}' | jq '.choices[0].message.content'"
echo
echo "  Benchmark your hardware:"
echo
echo "    ./benchmark.sh"
echo
echo "  Clean up when done:"
echo
echo "    $RUNTIME stop $CONTAINER && $RUNTIME rm $CONTAINER"
echo
