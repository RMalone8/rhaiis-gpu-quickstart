#!/bin/bash
# RHAIIS GPU Quickstart — Benchmark your hardware
# Runs GuideLLM against your already-running inference server.
# Requires: start.sh already ran successfully, Python 3.10+

set -e

GUIDELLM_DIR="${GUIDELLM_DIR:-$HOME/.guidellm}"
PORT="${PORT:-8000}"
ENDPOINT="http://127.0.0.1:${PORT}"
PROMPT_TOKENS="${PROMPT_TOKENS:-256}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
PROFILE="${PROFILE:-sweep}"
MAX_SECONDS="${MAX_SECONDS:-30}"
RESULTS_DIR="${RESULTS_DIR:-$GUIDELLM_DIR/results}"

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

# Python
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
        fail "Python 3.10+ required. Found $PY_VERSION."
    fi
    ok "Python: $PY_VERSION"
else
    fail "Python 3 is required. Install it: sudo dnf install python3 (RHEL/Fedora) or sudo apt install python3 (Ubuntu)."
fi

# pip
python3 -m pip --version &>/dev/null || fail "pip is required. Install it: sudo dnf install python3-pip (RHEL/Fedora) or sudo apt install python3-pip (Ubuntu)."
ok "pip available"

# Inference server running
if curl -sf "${ENDPOINT}/health" &>/dev/null; then
    MODEL_ID=$(curl -s "${ENDPOINT}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    ok "Inference server running at ${ENDPOINT} (model: ${MODEL_ID})"
else
    fail "Inference server not running at ${ENDPOINT}. Run ./start.sh first."
fi

# --- Set up GuideLLM ---
header "Setting up GuideLLM..."

VENV_DIR="$GUIDELLM_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if command -v guidellm &>/dev/null; then
    ok "GuideLLM already installed"
else
    info "Installing GuideLLM..."
    pip install --quiet "guidellm[recommended]" 2>&1 | tail -1
    ok "GuideLLM installed"
fi

# --- Run benchmark ---
header "Running benchmark against ${ENDPOINT}..."
echo
info "Profile: ${PROFILE} | Prompt tokens: ${PROMPT_TOKENS} | Output tokens: ${OUTPUT_TOKENS} | Max duration per strategy: ${MAX_SECONDS}s"
info "Results include throughput (tok/s), time to first token (TTFT), and time per output token (TPOT)."
echo

mkdir -p "$RESULTS_DIR"

guidellm run \
  --backend "kind=openai_http,target=${ENDPOINT},model=${MODEL_ID}" \
  --data "kind=synthetic_text,prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
  --profile "kind=${PROFILE}" \
  --constraint "kind=max_duration,seconds=${MAX_SECONDS}" \
  --output "kind=json,path=${RESULTS_DIR}/benchmark.json" \
  --output "kind=html,path=${RESULTS_DIR}/benchmark.html"

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
echo "    [From Zero to Benchmark — RHAIIS 3.5](URL)"
echo
