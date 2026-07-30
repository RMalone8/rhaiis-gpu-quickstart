#!/bin/bash
# RHAIIS CPU Quickstart — Benchmark your hardware
# Runs cpueval against your already-running inference server.
# Requires: start.sh already ran successfully, Python 3.9+, git

set -e

CPUEVAL_DIR="${CPUEVAL_DIR:-$HOME/.cpueval}"
PORT="${PORT:-8000}"
ENDPOINT="http://localhost:${PORT}"

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
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
        fail "Python 3.9+ required. Found $PY_VERSION."
    fi
    ok "Python: $PY_VERSION"
else
    fail "Python 3 is required. Install it: sudo dnf install python3 (RHEL/Fedora) or sudo apt install python3 (Ubuntu)."
fi

# git
command -v git &>/dev/null || fail "git is required."
ok "git available"

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

# --- Install cpueval ---
header "Setting up cpueval benchmarking tool..."

if [ -d "$CPUEVAL_DIR/vllm-cpu-perf-eval" ]; then
    ok "cpueval already installed at $CPUEVAL_DIR"
else
    info "Cloning vllm-cpu-perf-eval..."
    mkdir -p "$CPUEVAL_DIR"
    git clone https://github.com/redhat-et/vllm-cpu-perf-eval.git "$CPUEVAL_DIR/vllm-cpu-perf-eval" 2>&1 | tail -1
    ok "cpueval cloned"
fi

# --- Set up Python venv ---
VENV_DIR="$CPUEVAL_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

info "Installing dependencies..."
pip install --quiet ansible-core 2>&1 | tail -1
ok "ansible-core installed"

# Install Ansible collections
if [ -f "$CPUEVAL_DIR/vllm-cpu-perf-eval/automation/test-execution/ansible/requirements.yml" ]; then
    ansible-galaxy collection install -r "$CPUEVAL_DIR/vllm-cpu-perf-eval/automation/test-execution/ansible/requirements.yml" --force &>/dev/null
    ok "Ansible collections installed"
fi

# --- Run benchmark ---
header "Running benchmark against ${ENDPOINT}..."
echo
info "This will run a chat-smoke test suite against your running server."
info "Results include throughput (tok/s), time to first token (TTFT), and time per output token (TPOT)."
echo

cd "$CPUEVAL_DIR/vllm-cpu-perf-eval"

export VLLM_ENDPOINT_MODE=external
export VLLM_ENDPOINT_URL="$ENDPOINT"
export LOADGEN_HOSTNAME=localhost

./cpueval run --suite chat-smoke \
  --workload chat \
  --extra ansible_connection=local \
  --extra guidellm_use_container=false

# --- Show results ---
header "Benchmark results:"
echo

./cpueval results --last

# --- Summary ---
header "Done!"
echo
echo "  Results saved locally. Key metrics:"
echo "    TTFT  — Time to First Token (prompt evaluation delay)"
echo "    TPOT  — Time per Output Token (generation speed)"
echo "    Tok/s — Aggregate throughput across parallel streams"
echo
echo "  For deeper analysis and interactive dashboards, see the blog post:"
echo "    [From Zero to Benchmark — RHAIIS 3.5](URL)"
echo
echo "  Launch the interactive dashboard:"
echo "    cd $CPUEVAL_DIR/vllm-cpu-perf-eval && ./cpueval dashboard start"
echo
