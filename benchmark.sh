#!/bin/bash
# RHAII GPU Quickstart — Benchmark your hardware
# Runs the official Red Hat GuideLLM container against your already-running inference server.
# Requires: start.sh already ran successfully

set -e

GUIDELLM_IMAGE="${GUIDELLM_IMAGE:-registry.redhat.io/rhai/guidellm-rhel9:3.5.0}"
CONTAINER="${CONTAINER:-guidellm-benchmark}"
PORT="${PORT:-8000}"
ENDPOINT="http://127.0.0.1:${PORT}"
PROMPT_TOKENS="${PROMPT_TOKENS:-256}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
PROFILE="${PROFILE:-sweep}"
MAX_SECONDS="${MAX_SECONDS:-30}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/.guidellm/results}"

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

# curl + jq
command -v curl &>/dev/null || fail "curl is required."
command -v jq &>/dev/null || fail "jq is required. Install it: sudo dnf install jq (RHEL/Fedora) or sudo apt install jq (Ubuntu)."
ok "curl and jq available"

# Inference server running
if curl -sf "${ENDPOINT}/health" &>/dev/null; then
    MODEL_ID=$(curl -s "${ENDPOINT}/v1/models" | jq -r '.data[0].id' 2>/dev/null || echo "unknown")
    ok "Inference server running at ${ENDPOINT} (model: ${MODEL_ID})"
else
    fail "Inference server not running at ${ENDPOINT}. Run ./start.sh first."
fi

# --- Registry login ---
header "Checking registry access..."

if ! $RUNTIME pull --quiet "$GUIDELLM_IMAGE" &>/dev/null 2>&1; then
    warn "Need to log in to registry.redhat.io"
    echo "  Enter your Red Hat Customer Portal credentials (free at access.redhat.com):"
    $RUNTIME login registry.redhat.io || fail "Registry login failed."
fi
ok "Registry access confirmed"

# --- Pull GuideLLM ---
header "Pulling GuideLLM..."

info "Pulling container image (first time may take a few minutes)..."
$RUNTIME pull "$GUIDELLM_IMAGE" 2>&1 | tail -1
ok "GuideLLM image ready"

# --- Build runtime-specific flags ---
PODMAN_FLAGS=""
if [ "$RUNTIME" = "podman" ]; then
    PODMAN_FLAGS="--security-opt=label=disable --userns=keep-id:uid=1001"
fi

# --- Run benchmark ---
header "Running benchmark against ${ENDPOINT}..."
echo
info "Profile: ${PROFILE} | Prompt tokens: ${PROMPT_TOKENS} | Output tokens: ${OUTPUT_TOKENS} | Max duration per strategy: ${MAX_SECONDS}s"
info "Results include throughput (tok/s), time to first token (TTFT), and time per output token (TPOT)."
echo

mkdir -p "$RESULTS_DIR"

if $RUNTIME ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
    $RUNTIME rm -f "$CONTAINER" &>/dev/null || true
fi

# shellcheck disable=SC2086
$RUNTIME run --rm -t \
  --name "$CONTAINER" \
  --network=host \
  $PODMAN_FLAGS \
  -v "${RESULTS_DIR}:/results:Z" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  "$GUIDELLM_IMAGE" \
  guidellm run \
  --backend "kind=openai_http,target=${ENDPOINT},model=${MODEL_ID}" \
  --data "kind=synthetic_text,prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
  --profile "kind=${PROFILE}" \
  --constraint "kind=max_duration,seconds=${MAX_SECONDS}" \
  --output "kind=json,path=/results/benchmark.json" \
  --output "kind=html,path=/results/benchmark.html"

# --- Summary ---
header "Done!"
echo
echo "  Results saved to:"
echo "    ${RESULTS_DIR}/benchmark.json"
echo "    ${RESULTS_DIR}/benchmark.html"
echo
echo "  Key metrics:"
echo "    TTFT  — Time to First Token (prompt evaluation delay)"
echo "    TPOT  — Time per Output Token (generation speed)"
echo "    Tok/s — Aggregate throughput across parallel streams"
echo
echo "  For deeper analysis and interactive dashboards, see the blog post:"
echo "    [From Zero to Benchmark — RHAII 3.5](URL)"
echo
