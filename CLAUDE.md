# CLAUDE.md — RHAII CPU Quickstart

## What This Is

One-click quickstart for serving open-weight AI models on a Linux machine with an NVIDIA GPU using the Red Hat AI Inference Server (RHAII 3.5). Companion repo for the "From Zero to Benchmark" blog post.

No cluster. Just `podman run` + `curl`, backed by an NVIDIA GPU via `--device nvidia.com/gpu=all`.

## How to Run

```bash
# Default (Granite 2B, 8 GB+ VRAM)
./start.sh

# Match the blog (Qwen3.8-27B-FP8, 40 GB+ VRAM)
MODEL=Qwen/Qwen3.8-27B-FP8 ./start.sh

# Benchmark after the server is running
./benchmark.sh
```

Requires the NVIDIA driver and NVIDIA Container Toolkit on the host, with a CDI spec generated (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`).

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | (interactive prompt) | Model to serve. Set to skip the prompt. |
| `IMAGE` | `registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.5.0-ea.2-1782965184` | RHAII CUDA container image |
| `CACHE_DIR` | `~/rhaii-cache` | Persistent model weight cache |
| `HF_TOKEN` | (prompted) | Hugging Face access token |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM vLLM may reserve |
| `DTYPE` | `auto` | Model compute dtype passed to vLLM |
| `MAX_MODEL_LEN` | Auto (2048 for <=3B, up to 8192 for 27B+) | Max context length |
| `SHM_SIZE` | Auto (4g for <=3B, 8g for 7B+) | Container shared memory |

## Image Status

Using RHAII 3.5 Early Access. When GA drops, update IMAGE to:
`registry.redhat.io/rhaii/vllm-cuda-rhel9:3.5.0`

## Blog Alignment

This quickstart is the on-ramp to:
"From Zero to Benchmark: Deploying LLM Inference on CPU with RHAII 3.5"
by Maryam Tahhan, John Harrigan, Anton Ivanov
