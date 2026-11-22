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
| Admission slots | `max-num-seqs 4`; four simultaneous short requests verified (0.58 s wall). |
| Speculative decoding | DFly, 5 draft tokens. |
| Speed | C1 decode 36–55 tok/s from 512 to 250K; C4 aggregate 89–136 tok/s for 512–32K. Chat: JSON 44.8, code 45.6, rewrite 24.4, explain 28.9, story 16.9 tok/s. |
| Long context | 250,000-token C1/C2/C4 matrix cells completed; 200,024-token needle pass. |
| Features | Text, JSON schema, tool call round trip, single-image vision, Anthropic Messages, reasoning-effort control; 13/13 smoke checks. |
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

All **21 cells completed** (49 requests, no failures; `results/protocol-audit.json` passes). The full table is below. Aggregate decode is left empty where no positive common decode window exists; the per-request column is still defined.

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Shared decode/request tok/s | Per-request decode tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 955.7 | 35.2 | 35.2 | 36.2 | 0.5 | 14.5 | 1 | 1/1 |
| 512 | 2 | 987.0 | 75.2 | 37.6 | 38.5 | 1.0 | 12.2 | 2 | 2/2 |
| 512 | 4 | 1241.1 | 117.1 | 28.8 | 29.8 | 1.6 | 16.1 | 4 | 4/4 |
| 2,048 | 1 | 1272.8 | 40.6 | 40.6 | 41.8 | 1.6 | 12.6 | 1 | 1/1 |
| 2,048 | 2 | 1530.0 | 85.1 | 42.5 | 43.5 | 2.7 | 11.4 | 2 | 2/2 |
| 2,048 | 4 | 1335.8 | 114.6 | 27.8 | 28.8 | 4.3 | 16.1 | 4 | 4/4 |
| 8,192 | 1 | 1164.0 | 47.6 | 47.6 | 47.5 | 7.0 | 10.7 | 1 | 1/1 |
| 8,192 | 2 | 1227.0 | 71.5 | 35.8 | 37.2 | 9.8 | 12.7 | 2 | 2/2 |
| 8,192 | 4 | 1186.5 | 89.4 | 24.2 | 23.2 | 19.6 | 18.4 | 4 | 4/4 |
| 32,768 | 1 | 1476.4 | 50.6 | 50.6 | 50.8 | 22.2 | 10.1 | 1 | 1/1 |
| 32,768 | 2 | 1486.6 | 93.2 | 46.6 | 47.4 | 33.1 | 9.9 | 2 | 2/2 |
| 32,768 | 4 | 1471.6 | 136.4 | 33.8 | 35.7 | 57.4 | 10.8 | 4 | 4/4 |
| 131,072 | 1 | 1409.3 | 52.9 | 52.9 | 52.9 | 93.0 | 9.7 | 1 | 1/1 |
| 131,072 | 2 | 1293.7 | 82.3 | 41.1 | 25.7 | 151.2 | 8.2 | 2 | 2/2 |
| 131,072 | 4 | 1155.6 | — | — | 8.9 | 290.1 | 0.0 | 3 | 4/4 |
| 200,000 | 1 | 1162.8 | 47.0 | 47.0 | 45.5 | 172.0 | 10.9 | 1 | 1/1 |
| 200,000 | 2 | 1357.0 | 81.7 | 40.9 | 25.7 | 218.4 | 6.2 | 2 | 2/2 |
| 200,000 | 4 | 1310.4 | — | — | 3.2 | 380.3 | 0.0 | 3 | 4/4 |
| 250,000 | 1 | 1378.5 | 54.6 | 54.6 | 54.7 | 181.4 | 9.4 | 1 | 1/1 |
| 250,000 | 2 | 1309.6 | 72.9 | 36.5 | 28.9 | 283.1 | 3.4 | 2 | 2/2 |
| 250,000 | 4 | 1276.7 | — | — | 3.0 | 483.9 | 0.0 | 2 | 4/4 |

**Reading the table.** Effective prefill includes queueing and mixed decode work until the last request reaches its first token. C1 decode is 36–55 tok/s from 512 to 250K. C4 aggregate is 89–136 tok/s for 512–32K prompts. At 131K/C4, 200K/C4 and 250K/C4 all four requests completed, but the scheduler admitted the long prefills in waves: peak client overlap was 2–3 and there was no common decode interval, so aggregate decode is left empty. The per-request medians are low because three requests decode while another request prefill runs; the last request in each wave reached 41–42 tok/s at 200K/250K (17.4 tok/s at 131K). C1 and C2 are the practical long-context modes.

### Chat-shaped decode

Five chat prompts, 300 forced output tokens, thinking off, median of three:

| JSON | Code | Rewrite | Explain | Story |
|---:|---:|---:|---:|---:|
| 44.8 | 45.6 | 24.4 | 28.9 | 16.9 tok/s |

TTFT was 0.20–0.43 s on these short prompts. DFly acceptance is the difference between the matrix and the prose rows: the raw code-completion prompt accepts 5.45–5.50 tokens per draft step, while story/prose accepts 1.6–1.8.

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

`smoke.py` passed **13/13** checks on one boot.

| Check | Result | Evidence |
| --- | --- | --- |
| Health / models | pass | `/health` 200; `hy4-flash` listed |
| Arithmetic | pass | `19 + 23` → `42` |
| JSON schema | pass | valid city/temperature/conditions object |
| Tool call + result | pass | `get_weather({"city":"Paris"})` round trip |
| Single-image vision | pass | red square → `Red` |
| Anthropic count_tokens | pass | 23 input tokens |
| Anthropic messages | pass | thinking block + reply |
| Prompt reuse, 32K | pass | 21.77 s → 1.26 s |
| Needle, 200K | pass | 200,024 prompt tokens, `ZEBRA-9174`, 141.8 s |
| Reasoning-effort scale | pass | `no_think 0` / `high 136` / `max 187` tokens |
| 4-way concurrency | pass | wall 0.58 s for 4 requests |

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
python3 benchmarks/chat_smoke.py
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
  `reasoning-effort.json`, `concurrency-4.json`, `chat-smoke.json`,
  `protocol-audit.json`.

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
