# Hy4-flash NVFP4 on 2 × DGX Spark

Run Tencent's **Hy4-flash** on two NVIDIA DGX Spark systems (GB10, 128 GB
unified memory each). The measured recipe serves the NVFP4 checkpoint through a
locally patched vLLM image with **TP2 + expert parallel** over the QSFP link,
the model's **DFly** drafter, a **1,501,761-token KV pool** and a
**262,144-token** served window.

The repository contains the pinned launcher, the exact recipe, the image build
context, a reproducible benchmark harness and the measured result set. Loading,
speed, KV capacity and feature correctness are reported as separate milestones.

## 1. Current status

| Item | Verified result or current limit |
| --- | --- |
| Model precision | `atakans/Hy4-flash-NVFP4`: NVFP4 routed experts, FP8 Engram tables, BF16 elsewhere. No local requantization. |
| Hardware | 2 × DGX Spark GB10, 128 GB unified memory each, QSFP link. TP2. |
| Engine | vLLM `0.1.dev25463+g77766db1f.d20261117` in `dgx-spark-recipes/vllm-hy4-flash:77766db`. |
| API model ID | `hy4-flash` |
| Served context | **262,144 tokens** per request. Native window is 1,048,576; it is not served here. |
| KV capacity | **1,501,761 tokens** at auto sizing / 0.85 utilization — **5.73×** a 262K window. |
| Admission slots | `max-num-seqs 4`; four simultaneous requests verified. |
| Speculative decoding | DFly, 5 draft tokens. |
| Speed | See the measured matrix below. |
| Long context | 250,000-token matrix cell; 200k needle check. |
| Features | Text, JSON schema, tool calls, single-image vision, Anthropic Messages, reasoning-effort control. |
| Reliability | One boot; no one-hour soak, rollback or multi-user agent trace yet. |

**Scope:** the publishable claim is the measured two-Spark vLLM recipe. The
KV/batch tuning history is in [`EVAL-REPORT.md`](EVAL-REPORT.md).

## 2. Start the service

Requirements: two DGX Sparks with ~110 GiB free unified memory each, Docker
with the NVIDIA runtime, key-based ssh from the head to the worker, the QSFP
link up, and network access to Docker Hub and Hugging Face.

```bash
git clone https://github.com/tedngai/hy4-flash-2x-dgx-spark.git
cd hy4-flash-2x-dgx-spark
WORKER=<ssh-host-of-second-spark> ./start.sh --setup   # first run
```

On later boots:

```bash
WORKER=192.0.2.11 ./start.sh
WORKER=192.0.2.11 ./stop.sh
```

`start.sh` uses the vendored `run-recipe.sh` from
[`atakann/dgx-spark-recipes`](https://github.com/atakann/dgx-spark-recipes) and
the published recipe in
[`recipes/llm/hy4-flash-nvfp4-vllm-tp2.yaml`](recipes/llm/hy4-flash-nvfp4-vllm-tp2.yaml).
The first boot after a cache-clearing build takes ~25 minutes for
FlashInfer/CuTeDSL autotune; cached restarts take ~2–3 minutes.

The API binds to port **8000** on the head only; there is no authentication in
this recipe.

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash",
       "messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":256,
       "chat_template_kwargs":{"reasoning_effort":"no_think"}}'
```

## 3. Configuration

The published profile is the tuned recipe, not the upstream stock defaults.

| Setting | Published value | Upstream recipe | Why |
| --- | --- | --- | --- |
| `max-model-len` | **262144** | 262144 | Same served window; measured up to 250K. |
| `max-num-batched-tokens` | **4096** | 8192 | Frees peak activation memory for a 25% larger KV pool at ~1% prefill cost. |
| `max-num-seqs` | **4** | 4 | Four admission slots. |
| `gpu-memory-utilization` | **0.85** | 0.85 | Leaves host headroom on the 128 GB unified-memory Spark. |
| `kv-cache-memory-bytes` | **unset** | 6G | The hard cap overrides utilization and blocks the pool from growing. |
| Tensor parallel | **2** | 2 | One rank per Spark. |
| Expert parallel | **on** | on | Without it each Spark holds all experts. |
| DFly | 5 tokens | 5 tokens | Day-0 drafter. |
| Prefix caching | on | on | Keeps repeated system prompts cheap. |
| Parsers | `hy_v4` reasoning + tool | same | Day-0 recipe. |

**Capacity is not the same as context.** The KV pool holds 1,501,761 tokens.
A single request can use up to 262,144; four requests cannot each use 262,144.
At 250K the pool fits six full sessions; `max-num-seqs 4` is the admission cap,
not a reservation.

**Thinking control.** The chat template accepts `reasoning_effort` values
`no_think`, `high` and `max`; the default is `high`. `no_think` emits empty
think tags. A small `max_tokens` budget can be consumed by the reasoning block
and return no final content; give chat requests headroom.

## 4. Measured speed

<!-- MATRIX-TABLE -->

**These are code-shaped raw completions, not chat.** DFly acceptance on
open-ended prose is lower than on the matrix prompt, so a live chat stream is
slower than the short-context matrix decode rate. The chat smoke numbers are in
[`EVAL-REPORT.md`](EVAL-REPORT.md).

## 5. KV tuning history

The final pool came from three changes, measured separately:

| Config | KV tokens | Concurrency @ 262K | KV memory |
|---|---:|---:|---:|
| published 6G / 8192 | 578,345 | 2.21× | 6.0 GiB |
| 10G / 4096 | 1,020,326 | 3.89× | 10.0 GiB |
| 14G / 2048 | 1,471,508 | 5.61× | 14.0 GiB |
| 16G / 2048 | 1,681,864 | 6.42× | 16.0 GiB |
| **auto / 4096 (published)** | **1,501,761** | **5.73×** | **18.26 GiB** |
| auto / 2048 | 1,741,790 | 6.64× | 19.08 GiB |
| auto / 8192 | 1,199,665 | 4.58× | 16.5 GiB |

`--kv-cache-memory-bytes` **overrides** `gpu-memory-utilization`; the published
recipe's 6G cap reserved exactly 6 GiB and skipped memory profiling. Removing
the cap lets vLLM size the pool at 0.85 utilization. Lowering the prefill batch
from 8192 to 4096 then creates the peak-activation headroom for 18.26 GiB.

- 8192 → 4096: **+25% KV tokens**, ~1% slower 32K prefill.
- 4096 → 2048: **+16% KV tokens**, ~8% slower 32K prefill.
- 2048 → 1024: only ~1.5% more tokens; not worth it.

## 6. Feature acceptance

<!-- FEATURE-TABLE -->

These are smoke checks, not quality scores.

## 7. Reproduce benchmarks and stop

Capture the launch identity first; the matrix hashes `results/launch.json`.

```bash
python3 benchmarks/capture_env.py
```

Run the main matrix (512 forced output tokens; cells above the KV guard are
skipped):

```bash
OUTPUT_TOKENS=512 \
PREFILL_SIZES=512,2048,8192,32768,131072,200000,250000 \
CONCURRENCIES=1,2,4 \
python3 benchmarks/matrix.py
```

Recompute the table from saved token events and audit:

```bash
./benchmarks/finalize.sh results/matrix-<timestamp>
```

Run the feature checks only when the matrix is idle:

```bash
python3 benchmarks/smoke.py
```

Stop the service:

```bash
WORKER=192.0.2.11 ./stop.sh
```

The harness writes:

- `results/launch.json` — model revision, tokenizer/config/chat-template
  hashes, image ID and build-context hash, exact command, KV configuration,
  both distinct TP2 ranks and host state.
- `results/matrix-<timestamp>/<size>-c<n>.json` — raw per-request events.
- `results/matrix-<timestamp>/summary.json` — one row per completed cell.
- `results/matrix-<timestamp>/TABLE.md` — the rendered table.
- `results/feature-acceptance.json`, `needle-200k.json`,
  `reasoning-effort.json`, `concurrency-4.json`, `protocol-audit.json`.

## 8. Provenance

| Component | Pinned source / attribution |
| --- | --- |
| Model | [atakans/Hy4-flash-NVFP4](https://huggingface.co/atakans/Hy4-flash-NVFP4), measured at `30f201acf0b4f9ba178817bcbc04ccfa3ec473f9`. |
| Base model | [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash). |
| Runtime image | `vllm/vllm-openai@sha256:5e2c1badbca60595d6591c60feb4b8547f59da96e7a91710d40239250e721aba`, vLLM `0.1.dev25463+g77766db1f`. |
| Image patches | `images/vllm-hy4-flash/` (six Python files), tree hash in `results/launch.json`. |
| Launcher | [atakann/dgx-spark-recipes](https://github.com/atakann/dgx-spark-recipes), vendored at `run-recipe.sh`. |
| Recipe | `recipes/llm/hy4-flash-nvfp4-vllm-tp2.yaml`. |
| Repository pattern | [tedngai/inkling-2-mini-dgx-spark](https://github.com/tedngai/inkling-2-mini-dgx-spark), [tedngai/glm-5.3-flash-2x-dgx-spark](https://github.com/tedngai/glm-5.3-flash-2x-dgx-spark). |
| Hardware | 2 × NVIDIA DGX Spark GB10, 128 GB unified memory each, QSFP link. |

Weights are downloaded at runtime and retain their upstream Tencent Hy license.
They are not included in this repository.

## Study protocol and limitations

The workload, ranking rules and evidence audit are in
[`STUDY-PROTOCOL.md`](STUDY-PROTOCOL.md). The short version:

- A throughput winner requires repeatable matched measurements; this matrix is
  a baseline, not a final sweep.
- Differences under 5% between repeated key cells are treated as noise.
- Forced-output tokens are not quality scores.
- NVFP4 expert quantization is lossy; the quality gate is smoke checks plus
  needles, not a benchmark suite.
- The 262K window, single-image vision, one tool round trip and the Anthropic
  surface are qualified only to the extent shown above. Native 1M context,
  multi-image/video, a one-hour soak and rollback remain open.
