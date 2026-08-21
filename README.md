# Serve AI Models on Any Linux Machine with an NVIDIA GPU

_The Red Hat AI Inference Server in a single container. Open-weight models at zero cost per token._

> This quickstart is a GPU-adapted companion to [From Zero to Benchmark: Deploying LLM Inference on CPU with RHAIIS 3.5](URL) — same workflow, backed by an NVIDIA GPU instead of CPU.

## Overview

The Red Hat AI Inference Server (RHAIIS) is a production-grade model serving engine — the same vLLM-based runtime that powers Red Hat AI on OpenShift. It also runs standalone on any Linux machine with a GPU, as a single container.

This quickstart walks you through it:

1. Pick a model (Granite 2B or Qwen3.8-27B-FP8)
2. Pull the container image
3. Serve the model on your GPU
4. Call it with the OpenAI-compatible API

## What the result looks like

By the end, you'll have a model running locally that responds like this:

> **You ask:** "What are the advantages of running AI models on-premises?"
>
> **The model responds:** "1. Cost control — no per-token API fees. 2. Data sovereignty — sensitive data never leaves your network. 3. Customization — run any open-weight model, fine-tune for your domain."

The API is identical to OpenAI's — any application or framework that works with OpenAI works with this.

## One-click start

```bash
git clone https://github.com/RMalone8/rhaiis-gpu-quickstart.git
cd rhaiis-gpu-quickstart
./start.sh
```

The script checks your system and GPU, asks which model to serve, handles credentials, pulls the server, loads the model, and runs your first inference call.

Once it's running, ask it anything with `./send-request.sh`.

Prefer to do the whole thing step by step? Keep reading.

## What you need

| Requirement | Details |
|---|---|
| Linux with an NVIDIA GPU | RHEL 9/10, Fedora, Ubuntu 22.04+ (bare metal or VM) |
| NVIDIA driver + Container Toolkit | With a CDI spec generated (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) |
| VRAM | 8 GB minimum (Granite 2B) / 40 GB+ (Qwen3.8-27B-FP8) |
| Podman or Docker | Any container runtime — CDI device requests (`--device nvidia.com/gpu=all`) require a recent version |
| Red Hat registry access | Free account at [access.redhat.com](https://access.redhat.com) for the container image |
| Hugging Face token | Free account at [huggingface.co](https://huggingface.co) to download model weights |
| Internet access | To pull the container image and model |

## Pick your hardware

Not every model runs on every GPU. Here's what works where:

| Model Size | Minimum VRAM | Example Models |
|---|---|---|
| ≤1B (tiny) | 8 GB | granite-4.0-tiny, TinyLlama-1.1B |
| 2-3B (small) | 8 GB | granite-3.3-2b, qwen2.5-3b, phi-3-mini |
| 7-13B (medium) | 16 GB | granite-3.3-8b, llama-3.1-8b, mistral-7b |
| 14-16B (large) | 20 GB | qwen3-14b, deepseek-r1-14b |
| 27B+ FP8 (extra-large) | 40 GB+ | Qwen3.8-27B-FP8 |

`start.sh` auto-detects the bucket from the model name and sets `--max-model-len`, `--shm-size`, and the VRAM pre-flight check accordingly.

**What matters most:**
- **VRAM** is the hard constraint — the model must fit in GPU memory plus KV cache overhead
- **`--gpu-memory-utilization`** controls how much of that VRAM vLLM reserves upfront (default `0.90`) — lower it if you're sharing the GPU with other workloads
- **FP8/quantized checkpoints** cut memory footprint roughly in half versus bf16 for the same parameter count, which is what makes a 27B model fit in 40 GB

> **No single GPU with enough VRAM?** If you have multiple GPUs but none of them individually meet a model's minimum, split the model across them with tensor parallelism: add `--tensor-parallel-size <N>` to the `podman run` command (e.g. `--tensor-parallel-size 2` for two GPUs, with `--device nvidia.com/gpu=all` already passing all of them into the container). vLLM shards both the weights and the KV cache across all `N` GPUs, so two 24 GB GPUs can serve a model that needs ~40 GB combined. `N` must evenly divide the model's attention head count — 2, 4, and 8 are the safest choices.

---

## Before you start

Here's what the pieces are and how they fit together:

**Red Hat AI Inference Server (RHAIIS)** is the production model serving engine built on vLLM. It's the same runtime that Red Hat AI deploys on OpenShift — but it also runs standalone as a container on any Linux machine with an NVIDIA GPU. Enterprise-supported, with the container passed direct GPU access via `--device nvidia.com/gpu=all` (NVIDIA Container Device Interface).

**The models** are open-weight — you can download, run, and modify them without paying per token. This quickstart offers two choices:
- **IBM Granite 2B** — lightweight, Apache 2.0 licensed, good for classification, extraction, and summarization
- **Qwen3.8-27B-FP8** — higher quality reasoning, general use, and agentic workloads.

**The container image** (`registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9`) bundles the inference server with all CUDA dependencies. Model weights are cached on your host so subsequent starts skip the download.



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

**Granite 2B** (8 GB+ VRAM):

```bash
podman run -d --name inference-server \
  -p 8000:8000 \
  --shm-size=4g \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --userns=keep-id:uid=1001 \
  -v ~/rhaii-cache:/opt/app-root/src/.cache:Z \
  -e "HF_TOKEN=$HF_TOKEN" \
  registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.5.0-ea.2-1782965184 \
  --model ibm-granite/granite-3.3-2b-instruct \
  --dtype auto \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048
```

**Qwen3.8-27B-FP8** (40 GB+ VRAM, matches the blog):

```bash
podman run -d --name inference-server \
  -p 8000:8000 \
  --shm-size=8g \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --userns=keep-id:uid=1001 \
  -v ~/rhaii-cache:/opt/app-root/src/.cache:Z \
  -e "HF_TOKEN=$HF_TOKEN" \
  registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.5.0-ea.2-1782965184 \
  --model Qwen/Qwen3.8-27B-FP8 \
  --dtype auto \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

**What the flags do:**

| Flag | Purpose |
|------|---------|
| `-p 8000:8000` | Map port 8000 for the API |
| `--shm-size=4g/8g` | Shared memory for the inference engine (scale with model size) |
| `--device nvidia.com/gpu=all` | Pass all NVIDIA GPUs into the container via CDI |
| `--security-opt=label=disable` | Bypass SELinux labeling for bind-mounted volumes (RHEL) |
| `--userns=keep-id:uid=1001` | Map host user to the container's application user |
| `-v ~/rhaii-cache:...:Z` | Persistent model cache — weights survive container restarts |
| `HF_TOKEN` | Authenticates with Hugging Face to download model weights |
| `--host 0.0.0.0` | Listen on all interfaces (required for the port mapping to reach the container) |
| `--dtype auto` | Let vLLM pick compute dtype from the model's config (important for pre-quantized checkpoints like FP8) |
| `--gpu-memory-utilization` | Fraction of GPU VRAM vLLM reserves upfront for weights + KV cache (default `0.90`) |
| `--max-model-len` | Limit context length to manage memory |

> **Using Docker?** Omit `--security-opt=label=disable` and `--userns=keep-id:uid=1001` — these are podman-specific flags for SELinux and user namespace mapping. `--device nvidia.com/gpu=all` works the same way on both, provided the NVIDIA Container Toolkit and a CDI spec are set up.

The first run downloads the model weights (~4 GB for Granite 2B, ~27 GB for Qwen3.8-27B-FP8). Subsequent starts use the cached weights in `~/rhaii-cache`.

## Step 3: Wait for the model to load

Poll the health endpoint until it responds (log message text isn't a reliable readiness signal — it varies by vLLM version):

```bash
until curl -sf http://127.0.0.1:8000/health &>/dev/null; do sleep 5; done && echo "ready"
```

Or watch the logs directly:

```bash
podman logs -f inference-server
```

Press `Ctrl+C` to stop watching. The server keeps running in the background.

> **How long does this take?** First run: 3-10 minutes (model download + loading, including CUDA graph capture). After that: 1-2 minutes (loading from cache).

## Step 4: Verify the server

Check the health endpoint:

```bash
curl -s http://127.0.0.1:8000/health && echo " healthy"
```

Confirm the model is loaded:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
```

You should see your model ID in the response.

## Step 5: Ask a question

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.3-2b-instruct",
    "messages": [{"role": "user", "content": "What are the advantages of running AI on-premises? Answer in 3 bullet points."}],
    "max_tokens": 150
  }' | jq '.choices[0].message.content'
```

> **Using Qwen3.8-27B-FP8?** Replace the model value with `Qwen/Qwen3.8-27B-FP8` in the curl commands below.

You should get a coherent answer. This came from the Red Hat AI Inference Server running on your GPU — the same engine that powers production deployments on OpenShift.

## Step 6: Classify text

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.3-2b-instruct",
    "messages": [{"role": "user", "content": "Classify this text as positive, negative, or neutral. Respond with one word only.\n\nText: The new processor delivers exceptional performance at lower power consumption."}],
    "max_tokens": 5
  }' | jq '.choices[0].message.content'
```

Try a negative example:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
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
curl -s http://127.0.0.1:8000/v1/chat/completions \
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

The script installs [GuideLLM](https://github.com/vllm-project/guidellm), sweeps a range of request loads against your running server, and reports:

- **TTFT** — Time to First Token (prompt evaluation delay)
- **TPOT** — Time per Output Token (generation speed)
- **Tok/s** — Aggregate throughput across parallel streams

Results are saved to `~/.guidellm/results/benchmark.json` and `benchmark.html`. Override the workload shape with env vars, e.g. `PROMPT_TOKENS=512 OUTPUT_TOKENS=256 MAX_SECONDS=60 ./benchmark.sh`.

For detailed interpretation and advanced benchmarking (multi-node setups, custom datasets), see the [blog post](URL).

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
| **GPU access** | Direct via CDI (`--device nvidia.com/gpu=all`) | Varies |
| **Minimum VRAM** | 8 GB (production architecture) | 4-8 GB (often CPU-only) |
| **Best for** | Teams evaluating production AI infrastructure | Individual experimentation |

> **Starting smaller?** [Ollama](https://ollama.com) runs the same IBM Granite models with lower memory on Mac, Linux, or Windows. Great for experimentation. When you're ready for production, the Red Hat AI Inference Server is the path forward.

## Try other models

The inference server runs any Hugging Face model. Swap the `--model` flag (or set `MODEL=` before running `start.sh`) to try these:

| Model | Params | License | Good for | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | 27B (FP8) | Apache 2.0 | Multilingual reasoning, agentic tasks | **Matches the RHAIIS 3.5 blog post**. Needs 40 GB+ VRAM |
| `ibm-granite/granite-3.3-2b-instruct` | 2B | Apache 2.0 | Classification, extraction, Q&A | **Default in this quickstart**. Runs on 8 GB+ VRAM |
| `ibm-granite/granite-3.3-8b-instruct` | 8B | Apache 2.0 | Better reasoning, longer outputs | Needs ~16 GB VRAM |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Apache 2.0 | Multilingual, strong on benchmarks | Origin: CN — check your org's policy |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | MIT | Reasoning, chain-of-thought | Origin: CN |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | MIT | Compact, strong reasoning | |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Apache 2.0 | Ultra-lightweight, 8 GB VRAM | Good for testing on lower-spec GPUs |

> **Regional note:** Some organizations restrict the use of models originating from specific countries. IBM Granite is a safe default for all geos.

## Want to go further?

- **Read the blog** — [From Zero to Benchmark: Deploying LLM Inference on CPU with RHAIIS 3.5](URL) covers advanced benchmarking, NUMA pinning, tool calling, and interactive dashboards
- **Build an AI agent** — wrap this model in tools and multi-step orchestration. Available as a hands-on lab on the [Red Hat Demo Platform](https://demo.redhat.com)
- **Deploy on OpenShift** — the same container, managed with auto-scaling and a model management dashboard. Available as a hands-on lab on the [Red Hat Demo Platform](https://demo.redhat.com)
- **Learn more** — [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/)
