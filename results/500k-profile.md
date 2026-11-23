# 500K / 3-slot profile

Measured 2026-11-23 on the same two DGX Sparks, after the 262K / 4-slot matrix.
This is an additional serving profile; the published 21-cell matrix remains the
262K / 4-slot baseline.

## Configuration

| Setting | Value |
|---|---|
| Served window | **500,000 tokens** |
| Admission slots | **3** (`max-num-seqs 3`) |
| Prefill batch | **2048** (`max-num-batched-tokens 2048`) |
| KV sizing | auto (`--gpu-memory-utilization 0.85`, no `--kv-cache-memory-bytes`) |
| Speculative decoding | DFly, 5 tokens |
| Runtime | same local image, checkpoint and TP2 + EP layout |

## Capacity at the measured start

| Metric | Value |
|---|---:|
| Available KV cache memory | 19.32 GiB |
| KV blocks | 6,069 |
| KV tokens | **1,522,579** |
| Concurrency at 500K | **3.05×** |
| Headroom for three full 500K requests | ~22,000 tokens |

Three full-length requests fit, but the margin is small. If a workload needs
more than three 500K requests, this is not the profile; lower the served window
or use the 262K profile.

## Receipts

### 490K single request

Exact token-ID prompt from `benchmarks/matrix.py`, 16 forced output tokens.

| Prompt tokens | TTFT | Prefill | Output | Health |
|---:|---:|---:|---:|---:|
| 490,000 | 488.8 s | **1,002.4 tok/s** | 16/16 | 200 |

This is 98% of the 500K served window.

### Three concurrent short requests

| Requested | Wall | Per-request | Replies |
|---:|---:|---|---|
| 3 | **0.898 s** | 0.896 / 0.895 / 0.451 s | `OK 0`, `OK 1`, `OK 2` |

Receipt: `concurrency-3-500k.json`.

## Commands

```bash
WORKER=192.0.2.11 ./stop.sh
RECIPE=llm/hy4-flash-nvfp4-vllm-tp2-500k \
  HOST=0.0.0.0 WORKER=192.0.2.11 ./start.sh
```

The 500K profile has its own `cache/hy4-flash-nvfp4-vllm-tp2-500k/` compile
cache. A cold cache can spend ~25 minutes in FlashInfer/CuTeDSL autotune; a
warm restart took about 5–6 minutes in this installation.
