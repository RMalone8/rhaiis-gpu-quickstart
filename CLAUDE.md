# CLAUDE.md — RHAIIS CPU Quickstart

## What This Is

One-click quickstart for serving open-weight AI models on any Linux machine using the Red Hat AI Inference Server (RHAIIS 3.5). Companion repo for the "From Zero to Benchmark" blog post.

No cluster, no GPU. Just `podman run` + `curl`.

## How to Run

```bash
# Default (Granite 2B, 16 GB RAM)
./start.sh

# Match the blog (Qwen 7B, 32 GB RAM)
MODEL=Qwen/Qwen2.5-7B-Instruct ./start.sh

# Benchmark after the server is running
./benchmark.sh
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | (interactive prompt) | Model to serve. Set to skip the prompt. |
| `IMAGE` | `registry.redhat.io/rhaii-early-access/vllm-cpu-rhel9:3.5.0-ea.2-1782965184` | RHAIIS container image |
| `CACHE_DIR` | `~/rhaii-cache` | Persistent model weight cache |
| `HF_TOKEN` | (prompted) | Hugging Face access token |
| `KVCACHE_SPACE` | Auto (4 for <=3B, 10 for 7B+) | KV cache size in GB |
| `MAX_MODEL_LEN` | Auto (2048 for <=3B, 4096 for 7B+) | Max context length |
| `SHM_SIZE` | Auto (4g for <=3B, 8g for 7B+) | Container shared memory |

## Image Status

Using RHAIIS 3.5 Early Access. When GA drops, update IMAGE to:
`registry.redhat.io/rhaii/vllm-cpu-rhel9:3.5.0`

## Blog Alignment

This quickstart is the on-ramp to:
"From Zero to Benchmark: Deploying LLM Inference on CPU with RHAIIS 3.5"
by Maryam Tahhan, John Harrigan, Anton Ivanov
