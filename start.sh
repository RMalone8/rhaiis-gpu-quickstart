#!/bin/bash
# RHAIIS GPU Quickstart — One-click model serving
# Pulls the Red Hat AI Inference Server, serves your chosen model on an NVIDIA or AMD GPU (auto-detected), runs your first call.

set -e

CACHE_DIR="${CACHE_DIR:-$HOME/rhaii-cache}"
CONTAINER="inference-server"
PORT=8000
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"

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

# --- GPU vendor detection ---
if [ -e /dev/kfd ]; then
    ACCEL="amd"
elif compgen -G "/dev/nvidia[0-9]*" &>/dev/null; then
    ACCEL="nvidia"
elif command -v rocm-smi &>/dev/null && rocm-smi --showid &>/dev/null; then
    ACCEL="amd"
elif command -v nvidia-smi &>/dev/null && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    ACCEL="nvidia"
else
    fail "No supported GPU found. Expected /dev/kfd (AMD) or /dev/nvidia* (NVIDIA) device nodes — install the matching driver and try again."
fi

if [ "$ACCEL" = "nvidia" ]; then
    IMAGE="${IMAGE:-quay.io/aipcc/rhaiis/cuda-ubi9:3.5.0}"  # TODO: changed for testing! Switch back to registry.redhat.io image!!
else
    IMAGE="${IMAGE:-quay.io/aipcc/rhaiis/rocm-ubi9:3.4.4}"  # TODO: changed for testing! Switch back to registry.redhat.io image!!
fi

# --- Model selection ---
if [ -z "$MODEL" ]; then
    header "Which model do you want to serve?"
    echo
    echo -e "  ${BOLD}1)${NC} Granite 2B          — lightweight, runs on 8 GB VRAM (good for getting started)"
    echo -e "  ${BOLD}2)${NC} Qwen3.8-27B-FP8    — higher quality, runs on 40 GB+ VRAM (matches the RHAIIS 3.5 blog)"
    echo
    read -rp "  Enter 1 or 2 [default: 1]: " MODEL_CHOICE
    MODEL_CHOICE="${MODEL_CHOICE:-1}"

    case "$MODEL_CHOICE" in
        2)
            MODEL="Qwen/Qwen3.8-27B-FP8"
            ;;
        *)
            MODEL="ibm-granite/granite-3.3-2b-instruct"
            ;;
    esac
fi

# --- Auto-detect model size and set parameters ---
MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
if echo "$MODEL_LOWER" | grep -qE '(^|[^0-9])(27|30|32|34|70|72)b'; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    SHM_SIZE="${SHM_SIZE:-8g}"
    MIN_VRAM_GB=40
elif echo "$MODEL_LOWER" | grep -qE '(^|[^0-9])(14|15|16)b'; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    SHM_SIZE="${SHM_SIZE:-8g}"
    MIN_VRAM_GB=20
elif echo "$MODEL_LOWER" | grep -qE '(^|[^0-9])(7|8|9|10|11|12|13)b'; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    SHM_SIZE="${SHM_SIZE:-8g}"
    MIN_VRAM_GB=16
else
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
    SHM_SIZE="${SHM_SIZE:-4g}"
    MIN_VRAM_GB=8
fi

ok "Model: $MODEL"
info "GPU memory utilization: ${GPU_MEMORY_UTILIZATION} | Max context: ${MAX_MODEL_LEN} tokens | Shared memory: ${SHM_SIZE}"

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

# OS check — warn on macOS
if [ "$(uname -s)" = "Darwin" ]; then
    warn "macOS detected. The RHAIIS container requires Linux (RHEL, Fedora, Ubuntu) with an NVIDIA or AMD GPU. It will not run natively on macOS."
fi

ok "Accelerator: $([ "$ACCEL" = "nvidia" ] && echo NVIDIA || echo AMD)"

if [ "$ACCEL" = "nvidia" ]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
else
    if command -v rocm-smi &>/dev/null; then
        GPU_COUNT=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -c "VRAM Total Memory")
        VRAM_BYTES=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Memory" | grep -oE '[0-9]+' | sort -n | tail -1)
        [ -n "$VRAM_BYTES" ] && VRAM_MB=$((VRAM_BYTES / 1024 / 1024))
    else
        GPU_COUNT=$(amd-smi list --csv 2>/dev/null | tail -n +2 | grep -c .)
    fi
fi

if ! [[ "$VRAM_MB" =~ ^[0-9]+$ ]]; then
    VRAM_MB=""
fi

if [ -z "$VRAM_MB" ]; then
    warn "Could not detect GPU memory. This model needs ${MIN_VRAM_GB}+ GB VRAM."
else
    VRAM_GB=$((VRAM_MB / 1024))
    if [ "$VRAM_GB" -lt "$MIN_VRAM_GB" ]; then
        fail "Found ${VRAM_GB} GB VRAM on the largest GPU. ${MODEL} needs ${MIN_VRAM_GB} GB minimum. Try a smaller model or set MODEL= to override."
    else
        ok "GPU: ${GPU_COUNT} device(s) detected, ${VRAM_GB} GB VRAM on the largest (minimum ${MIN_VRAM_GB} GB for this model)"
    fi
fi

if [ "$ACCEL" = "nvidia" ]; then
    # NVIDIA Container Toolkit CDI spec — required for --device nvidia.com/gpu=all
    if [ -f /etc/cdi/nvidia.yaml ] || [ -f /var/run/cdi/nvidia.yaml ]; then
        ok "NVIDIA CDI spec found"
    else
        warn "No NVIDIA CDI spec found. If the container fails to see the GPU, generate one: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
    fi
else
    # AMD KFD/DRI device nodes — required for --device=/dev/kfd --device=/dev/dri
    if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
        ok "AMD KFD/DRI device nodes found"
    else
        warn "No /dev/kfd or /dev/dri found. Ensure the amdgpu kernel driver is loaded."
    fi

    if [ "$RUNTIME" = "podman" ]; then
        if command -v crun &>/dev/null; then
            ok "crun runtime available"
        else
            fail "crun is required for AMD GPU access under podman (--group-add keep-groups only works with the crun runtime, not runc). Install it: sudo dnf install crun (RHEL/Fedora) or sudo apt install crun (Ubuntu/Debian)."
        fi
    fi
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
    $RUNTIME login quay.io || fail "Registry login failed." # TODO: changed for testing! Switch back to registry.redhat.io!!
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
    if [ "$ACCEL" = "amd" ]; then
        PODMAN_FLAGS="--runtime crun --security-opt=label=disable"
    else
        PODMAN_FLAGS="--security-opt=label=disable --userns=keep-id:uid=1001"
    fi
fi

# --- Build accelerator-specific device flags ---
if [ "$ACCEL" = "nvidia" ]; then
    DEVICE_FLAGS="--device nvidia.com/gpu=all"
else
    DEVICE_FLAGS="--device=/dev/kfd --device=/dev/dri --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    if [ "$RUNTIME" = "podman" ]; then
        DEVICE_FLAGS="$DEVICE_FLAGS --group-add keep-groups"
    else
        DEVICE_FLAGS="$DEVICE_FLAGS --group-add video --group-add render"
    fi
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
  $DEVICE_FLAGS \
  $PODMAN_FLAGS \
  -v "${CACHE_DIR}:/opt/app-root/src/.cache:Z" \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e HF_HUB_OFFLINE=0 \
  "$IMAGE" \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --host 0.0.0.0 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" &>/dev/null

ok "Container started"

# --- Wait for model to load ---
header "Loading model (this takes 2-10 minutes on first run)..."

SECONDS=0
while true; do
    if curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null; then
        ok "Server ready in ${SECONDS}s"
        break
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

HEALTH=$(curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null && echo "healthy" || echo "unhealthy")
if [ "$HEALTH" = "healthy" ]; then
    ok "Health endpoint: healthy"
else
    fail "Health endpoint not responding. Check logs: $RUNTIME logs $CONTAINER"
fi

MODELS=$(curl -s "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)
if echo "$MODELS" | jq -e '.data[0].id' &>/dev/null; then
    MODEL_ID=$(echo "$MODELS" | jq -r '.data[0].id')
    ok "Model serving: $MODEL_ID"
else
    fail "API not responding. Check logs: $RUNTIME logs $CONTAINER"
fi

# --- First call ---
header "Making your first inference call..."
echo

RESPONSE=$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
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

POS=$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Classify as positive, negative, or neutral. One word only.\n\nText: The new product exceeded all expectations.\"}],
    \"max_tokens\": 5
  }" | jq -r '.choices[0].message.content')

NEG=$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
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
echo -e "  The Red Hat AI Inference Server is running at ${BOLD}http://127.0.0.1:${PORT}${NC}"
echo -e "  Model: ${BOLD}$MODEL${NC}"
echo -e "  API: ${BOLD}OpenAI-compatible${NC} (/v1/chat/completions)"
echo -e "  Cache: ${BOLD}$CACHE_DIR${NC} (weights persist across restarts)"
echo
echo "  Try your own prompt:"
echo
echo "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
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
