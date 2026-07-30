# Serve AI Models on Any Linux Machine

_The Red Hat AI Inference Server in a single container. Open-weight models at zero cost per token._

> This quickstart is the companion to [From Zero to Benchmark: Deploying LLM Inference on CPU with RHAIIS 3.5](URL).

## Overview

The Red Hat AI Inference Server (RHAIIS) is a production-grade model serving engine — the same vLLM-based runtime that powers Red Hat AI on OpenShift. It also runs standalone on any Linux machine as a single container.

This quickstart walks you through it:

1. Pick a model (Granite 2B or Qwen 7B)
2. Pull the container image
3. Serve the model
4. Call it with the OpenAI-compatible API

The models are open-weight. The runtime is enterprise-supported. Everything runs on your CPU.

## What the result looks like

By the end, you'll have a model running locally that responds like this:

> **You ask:** "What are the advantages of running AI models on-premises?"
>
> **The model responds:** "1. Cost control — no per-token API fees. 2. Data sovereignty — sensitive data never leaves your network. 3. Customization — run any open-weight model, fine-tune for your domain."

The API is identical to OpenAI's — any application or framework that works with OpenAI works with this.

## One-click start

```bash
git clone https://github.com/jkershawrh/rhaiis-cpu-quickstart.git
cd rhaiis-cpu-quickstart
./start.sh
```

The script checks your system, asks which model to serve, handles credentials, pulls the server, loads the model, and runs your first inference call — including a classification demo.

To match the blog's exact configuration:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct ./start.sh
```

Prefer to do it step by step? Keep reading.

---

## What you need

| Requirement | Details |
|---|---|
| x86_64 Linux | RHEL 9/10, Fedora, Ubuntu 22.04+ (bare metal or VM — not Mac podman) |
| RAM | 16 GB minimum (Granite 2B) / 32 GB (Qwen 7B) |
| CPU cores | 4+ minimum, 16+ recommended |
| Podman or Docker | Any container runtime |
| Red Hat registry access | Free account at [access.redhat.com](https://access.redhat.com) for the container image |
| Hugging Face token | Free account at [huggingface.co](https://huggingface.co) to download model weights |
| Internet access | To pull the container image and model |

> **Why 16–32 GB?** The inference server loads the full model into memory and allocates a KV cache for processing requests. A 2B parameter model in bfloat16 needs ~4 GB for weights plus several GB for the cache. A 7B model needs ~14 GB for weights plus a 10 GB cache. This is the same architecture that serves models in production — it trades memory for speed and reliability.
>
> **Don't have 16 GB or Linux?** This quickstart is for the production runtime. If you want to experiment on a laptop first, [Ollama](https://ollama.com) runs the same Granite models with lower memory requirements on any OS.

---

## Pick your hardware

Not every model runs on every server. Here's what works where:

| Model Size | Minimum RAM | Intel Xeon Generation | Example Models |
|---|---|---|---|
| ≤1B (tiny) | 8 GB | 4th Gen+ (Sapphire Rapids) | granite-4.0-tiny, TinyLlama-1.1B |
| 2-3B (small) | 16 GB | 4th Gen+ (Sapphire Rapids) | granite-3.3-2b, qwen2.5-3b, phi-3-mini |
| 7-8B (medium) | 32 GB | 5th Gen+ (Emerald Rapids) | granite-3.3-8b, llama-3.1-8b, mistral-7b |
| 14B+ (large) | 64 GB+ | 6th Gen (Granite Rapids) recommended | qwen3-14b, deepseek-r1-14b |

> **Validated reference:** The blog post validated this quickstart on an AWS c8i.12xlarge (48 vCPUs, 96 GB RAM, Intel Xeon 6 / Granite Rapids).

**What matters most:**
- **RAM** is the hard constraint — the model must fit in memory plus KV cache overhead
- **Cores** affect throughput — more cores = more concurrent requests. At 8 cores, expect ~9 tok/s; at 16 cores, ~15 tok/s; at 32 cores, ~283 tok/s at concurrency=32
- **AMX** (available on 4th Gen+) accelerates bfloat16 inference ~2-3x over older AVX-512

---

## Before you start

Here's what the pieces are and how they fit together:

**Red Hat AI Inference Server (RHAIIS)** is the production model serving engine built on vLLM. It's the same runtime that Red Hat AI deploys on OpenShift — but it also runs standalone as a container on any Linux machine. Enterprise-supported, optimized for Intel and AMD CPUs, with automatic hardware acceleration detection. RHAIIS 3.5 automatically handles OpenMP thread optimization — no manual `LD_PRELOAD` configuration needed.

**The models** are open-weight — you can download, run, and modify them without paying per token. This quickstart offers two choices:
- **IBM Granite 2B** — lightweight, Apache 2.0 licensed, good for classification, extraction, and summarization
- **Qwen 2.5 7B** — higher quality reasoning, matches the RHAIIS 3.5 blog configuration

**The container image** (`registry.redhat.io/rhaii-early-access/vllm-cpu-rhel9`) bundles the inference server with all dependencies. Model weights are cached on your host so subsequent starts skip the download.

```
┌──────────────────────────────────────────┐
│  Your Linux machine (16-32 GB+ RAM)      │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Container                         │  │
│  │  ┌──────────────────────────────┐  │  │
│  │  │  Red Hat AI Inference Server │  │  │
│  │  │  (RHAIIS 3.5 / vLLM engine)  │  │  │
│  │  │  ┌────────────────────────┐  │  │  │
│  │  │  │ Granite 2B or Qwen 7B │  │  │  │
│  │  │  │ (open-weight)         │  │  │  │
│  │  │  └────────────────────────┘  │  │  │
│  │  └──────────────────────────────┘  │  │
│  │  API: localhost:8000               │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ~/rhaii-cache ← model weights persist   │
│  curl → localhost:8000                   │
└──────────────────────────────────────────┘
```

---

## Step 1: Log in to the registries

The container image comes from the Red Hat registry. The model weights come from Hugging Face. Log in to both:

```bash
podman login registry.redhat.io
```

Enter your Red Hat Customer Portal credentials (free account at [access.redhat.com](https://access.redhat.com)).

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN=your_hugging_face_token_here
```

You can create a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Create a persistent cache directory for model weights:

```bash
mkdir -p ~/rhaii-cache
```

## Step 2: Start the inference server

**Granite 2B** (16 GB RAM):

```bash
podman run -d --name inference-server \
  -p 8000:8000 \
  --shm-size=4g \
  --security-opt=label=disable \
  --userns=keep-id:uid=1001 \
  -v ~/rhaii-cache:/opt/app-root/src/.cache:Z \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "VLLM_CPU_KVCACHE_SPACE=4" \
  registry.redhat.io/rhaii-early-access/vllm-cpu-rhel9:3.5.0-ea.2-1782965184 \
  --model ibm-granite/granite-3.3-2b-instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --max-model-len 2048
```

**Qwen 7B** (32 GB RAM, matches the blog):

```bash
podman run -d --name inference-server \
  -p 8000:8000 \
  --shm-size=8g \
  --security-opt=label=disable \
  --userns=keep-id:uid=1001 \
  -v ~/rhaii-cache:/opt/app-root/src/.cache:Z \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "VLLM_CPU_KVCACHE_SPACE=10" \
  registry.redhat.io/rhaii-early-access/vllm-cpu-rhel9:3.5.0-ea.2-1782965184 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --max-model-len 4096
```

**What the flags do:**

| Flag | Purpose |
|------|---------|
| `-p 8000:8000` | Map port 8000 for the API |
| `--shm-size=4g/8g` | Shared memory for the inference engine (scale with model size) |
| `--security-opt=label=disable` | Bypass SELinux labeling for bind-mounted volumes (RHEL) |
| `--userns=keep-id:uid=1001` | Map host user to the container's application user |
| `-v ~/rhaii-cache:...:Z` | Persistent model cache — weights survive container restarts |
| `HF_TOKEN` | Authenticates with Hugging Face to download model weights |
| `VLLM_CPU_KVCACHE_SPACE` | KV cache size in GB (4 for 2B models, 10 for 7B) |
| `--host 0.0.0.0` | Listen on all interfaces |
| `--dtype bfloat16` | Use bfloat16 precision for CPU inference |
| `--max-model-len` | Limit context length to manage memory |

> **Using Docker?** Omit `--security-opt=label=disable` and `--userns=keep-id:uid=1001` — these are podman-specific flags for SELinux and user namespace mapping.

The first run downloads the model weights (~4 GB for Granite 2B, ~14 GB for Qwen 7B). Subsequent starts use the cached weights in `~/rhaii-cache`.

## Step 3: Wait for the model to load

Watch the logs:

```bash
podman logs -f inference-server
```

When you see this line, the server is ready:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Press `Ctrl+C` to stop watching. The server keeps running in the background.

> **How long does this take?** First run: 3-10 minutes (model download + loading). After that: 1-2 minutes (loading from cache). On newer CPUs with more cores, loading is faster.

## Step 4: Verify the server

Check the health endpoint:

```bash
curl -s http://localhost:8000/health && echo " healthy"
```

Confirm the model is loaded:

```bash
curl -s http://localhost:8000/v1/models | jq .
```

You should see your model ID in the response.

## Step 5: Ask a question

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.3-2b-instruct",
    "messages": [{"role": "user", "content": "What are the advantages of running AI on-premises? Answer in 3 bullet points."}],
    "max_tokens": 150
  }' | jq '.choices[0].message.content'
```

> **Using Qwen 7B?** Replace the model value with `Qwen/Qwen2.5-7B-Instruct` in the curl commands below.

You should get a coherent answer. This came from the Red Hat AI Inference Server running on your CPU — the same engine that powers production deployments on OpenShift.

## Step 6: Classify text

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.3-2b-instruct",
    "messages": [{"role": "user", "content": "Classify this text as positive, negative, or neutral. Respond with one word only.\n\nText: The new processor delivers exceptional performance at lower power consumption."}],
    "max_tokens": 5
  }' | jq '.choices[0].message.content'
```

Try a negative example:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.3-2b-instruct",
    "messages": [{"role": "user", "content": "Classify this text as positive, negative, or neutral. Respond with one word only.\n\nText: The deployment failed repeatedly and the documentation was outdated."}],
    "max_tokens": 5
  }' | jq '.choices[0].message.content'
```

One model, different prompts, different tasks — no retraining needed.

## Step 7: Extract structured data

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.3-2b-instruct",
    "messages": [{"role": "user", "content": "Extract all product names and companies from this text as a JSON array.\n\nText: Red Hat OpenShift serves IBM Granite models for production AI workloads."}],
    "max_tokens": 150
  }' | jq '.choices[0].message.content'
```

The same model handled question answering, classification, and data extraction.

---

## Clean up

```bash
podman stop inference-server && podman rm inference-server
```

The model weights in `~/rhaii-cache` are preserved for faster restarts. To remove them:

```bash
rm -rf ~/rhaii-cache
```

## What you just did

This quickstart is the "Hello World" of self-hosted AI — a quick validation that you have the inference server running, the model is serving, and the performance is roughly what you'd expect. You proved:

- The Red Hat AI Inference Server (RHAIIS 3.5) runs on your hardware
- Open-weight models handle real tasks (classification, extraction, Q&A)
- The API is OpenAI-compatible — anything you build here works everywhere
- Model weights are cached locally for fast restarts
- The cost per token is zero

The next question is: what do you build with it?

---

## Benchmark your hardware

Measure throughput and latency with the same toolchain used in the blog post:

```bash
./benchmark.sh
```

The script installs [cpueval](https://github.com/redhat-et/vllm-cpu-perf-eval), runs a chat-smoke test suite against your running server, and reports:

- **TTFT** — Time to First Token (prompt evaluation delay)
- **TPOT** — Time per Output Token (generation speed)
- **Tok/s** — Aggregate throughput across parallel streams

For detailed interpretation and advanced benchmarking (NUMA pinning, multi-node setups, interactive dashboards), see the [blog post](URL).

## Add a chat UI

[Open WebUI](https://github.com/open-webui/open-webui) provides a browser-based chat interface for your running server:

```bash
podman run -d --name open-webui \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://host.containers.internal:8000/v1 \
  ghcr.io/open-webui/open-webui:main
```

Open http://localhost:3000 in your browser. Create an admin account and start chatting.

> **Note:** `host.containers.internal` lets the Open WebUI container reach the inference server running in its own container. On Docker, use `host.docker.internal` instead.

---

## What makes this different from other runtimes

| | Red Hat AI Inference Server (RHAIIS) | Other runtimes (Ollama, llama.cpp) |
|---|---|---|
| **Engine** | vLLM — production throughput, continuous batching | llama.cpp — lightweight, single-user |
| **Support** | Enterprise-supported by Red Hat | Community-supported |
| **Path to production** | Same container deploys on OpenShift with scaling, monitoring, model management | Manual deployment |
| **Threading** | Automatic OpenMP optimization in 3.5 (no manual LD_PRELOAD) | Manual tuning |
| **Minimum RAM** | 16 GB (production architecture) | 4-8 GB |
| **Best for** | Teams evaluating production AI infrastructure | Individual experimentation |

> **Starting smaller?** [Ollama](https://ollama.com) runs the same IBM Granite models with lower memory on Mac, Linux, or Windows. Great for experimentation. When you're ready for production, the Red Hat AI Inference Server is the path forward.

## Try other models

The inference server runs any Hugging Face model. Swap the `--model` flag (or set `MODEL=` before running `start.sh`) to try these:

| Model | Params | License | Good for | Notes |
|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Apache 2.0 | Multilingual reasoning | **Matches the RHAIIS 3.5 blog post**. Needs 32 GB RAM |
| `ibm-granite/granite-3.3-2b-instruct` | 2B | Apache 2.0 | Classification, extraction, Q&A | **Default in this quickstart**. Runs on 16 GB |
| `ibm-granite/granite-3.3-8b-instruct` | 8B | Apache 2.0 | Better reasoning, longer outputs | Needs ~32 GB RAM |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Apache 2.0 | Multilingual, strong on benchmarks | Origin: CN — check your org's policy |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | MIT | Reasoning, chain-of-thought | Origin: CN |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | MIT | Compact, strong reasoning | |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Apache 2.0 | Ultra-lightweight, 8 GB RAM | Good for testing on lower-spec machines |

> **Regional note:** Some organizations restrict the use of models originating from specific countries. IBM Granite is a safe default for all geos.

## Want to go further?

- **Read the blog** — [From Zero to Benchmark: Deploying LLM Inference on CPU with RHAIIS 3.5](URL) covers advanced benchmarking, NUMA pinning, tool calling, and interactive dashboards
- **Build an AI agent** — wrap this model in tools and multi-step orchestration. Available as a hands-on lab on the [Red Hat Demo Platform](https://demo.redhat.com)
- **Deploy on OpenShift** — the same container, managed with auto-scaling and a model management dashboard. Available as a hands-on lab on the [Red Hat Demo Platform](https://demo.redhat.com)
- **Learn more** — [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/)
