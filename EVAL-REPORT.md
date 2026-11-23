# Evaluation report — Hy4-flash NVFP4 on two DGX Sparks

Measured 2026-11-22 on two NVIDIA DGX Spark GB10 systems, 128 GB unified
memory each, joined by the QSFP link. The serving stack is vLLM
`0.1.dev25463+g77766db1f.d20261117` in a locally built image, TP2 with expert
parallel, and the model's DFly drafter with five tokens.

This report separates four questions:

1. Does the checkpoint load and serve correctly on two Sparks?
2. How large can the KV pool be made without breaking prefill?
3. What does the burst-serving matrix measure?
4. Which features pass a bounded acceptance check?

## 1. Provenance

| Component | Pin |
|---|---|
| Model | `atakans/Hy4-flash-NVFP4@30f201acf0b4f9ba178817bcbc04ccfa3ec473f9` |
| Base runtime | `vllm/vllm-openai@sha256:5e2c1badbca60595d6591c60feb4b8547f59da96e7a91710d40239250e721aba` |
| Local image | `dgx-spark-recipes/vllm-hy4-flash:77766db`, image ID `sha256:562aab44…` |
| Image build context | `images/vllm-hy4-flash/`, tree hash in `results/launch.json` |
| vLLM | `0.1.dev25463+g77766db1f.d20261117` |
| Checkpoint size | 164.7 GB (153.4 GiB) on disk |
| Parallelism | TP2, expert parallel, one rank per Spark |
| Speculative decoding | DFly, 5 draft tokens |
| Served window | 262,144 tokens |

The model is Tencent's Hy4-flash: 220B total parameters with 64B of Engram
tables and 12B active. This checkpoint is an NVFP4 build made with NVIDIA
ModelOpt 0.48: routed experts are NVFP4, the Engram tables are FP8 with one
scale per row, and the rest stays BF16. The model repository and its weights
retain the upstream Tencent Hy license; weights are not in this repository.

## 2. Why a local image

The public `vllm/vllm-openai:hy4-flash` image contains Hy4-flash support, but
its attention path is FP8-only and targets Hopper or data-center Blackwell.
Two DGX Sparks are SM120. The image build adds six Python files:

| File | Change |
|---|---|
| `sm120.py` | Hy4's per-head learnable attention sink through FlashInfer's SM120 sparse MLA kernel. |
| `attention.py` | Layer dispatch to that backend on compute capability 12, for sparse and sliding-window layers. |
| `hy_v4_attention.py` | The same dispatch for the DFly drafter's layers. |
| `mla_attention.py` | `fp8_ds_mla` KV cache on sliding-window MLA. |
| `model.py` | ModelOpt NVFP4 expert loading: weight, block scale, global scale and input scale per expert. |
| `dfly_config.py` | DFly accepts a ModelOpt NVFP4 target; the drafter's own layers stay BF16. |

`--enable-expert-parallel` is required. Without it each Spark holds all
experts, which does not fit.

## 3. KV pool and prefill-batch sweep

The upstream recipe pins `--kv-cache-memory-bytes 6G`. That flag **overrides**
`--gpu-memory-utilization`; vLLM reserves exactly 6 GiB and skips memory
profiling. The first measured baseline was therefore 578,345 KV tokens,
2.21× a 262K request.

The pool was grown in two dimensions: the KV cap and
`--max-num-batched-tokens`. Lowering the prefill batch reduces peak activation
and workspace memory, which leaves more room for KV.

| Config | KV memory | KV tokens | Concurrency @262K | 32K prefill |
|---|---:|---:|---:|---:|
| published 6G / 8192 | 6.0 GiB | 578,345 | 2.21× | 1,461 tok/s |
| 10G / 8192 | 10.0 GiB | 963,984 | 3.68× | — |
| 10G / 4096 | 10.0 GiB | 1,020,326 | 3.89× | — |
| 12G / 4096 | 12.0 GiB | 1,224,535 | 4.67× | — |
| 12G / 2048 | 12.0 GiB | 1,261,398 | 4.81× | — |
| 14G / 2048 | 14.0 GiB | 1,471,508 | 5.61× | — |
| 14G / 1024 | 14.0 GiB | 1,493,995 | 5.70× | — |
| 16G / 2048 | 16.0 GiB | 1,681,864 | 6.42× | — |
| 16G / 1024 | 16.0 GiB | 1,707,566 | 6.51× | — |
| **auto / 4096** | **18.26 GiB** | **1,501,761** | **5.73×** | **1,551 tok/s** |
| auto / 2048 | 19.08 GiB | 1,741,790 | 6.64× | 1,426 tok/s |
| auto / 8192 | 16.5–16.55 GiB | 1,166,868–1,199,665 | 4.45–4.58× | 1,568 tok/s |

"auto" means `--kv-cache-memory-bytes` is omitted and
`--gpu-memory-utilization 0.85` sizes the pool. `--max-num-seqs 4` caps
admission at four requests regardless of the token multiple.

The published choice is **auto / 4096**: a 2.6× larger pool than the 6G
recipe for about 1% prefill loss against batch 8192. Batch 2048 gives another
16% KV tokens but costs about 8% prefill; batch 1024 adds only 1.5% tokens
over 2048 and is not worth it.

### Long-context prefill at auto/4096

| Prompt tokens | TTFT | Prefill |
|---:|---:|---:|
| 27,477 | 17.9 s | 1,531 tok/s |
| 109,770 | 104.3 s | 1,052 tok/s |
| 214,348 | 217.3 s | 986 tok/s |
| 250,000 (matrix C1) | 181.4 s | 1,378.5 tok/s |

All returned normally with health 200 after each. The 250K matrix cell is the
end-to-end receipt for the served window.

## 4. Burst-serving matrix

All **21 cells completed** (49 requests, zero failed requests). `results/protocol-audit.json` reports `status: pass`; the only warnings are the three long-context C4 cells with no positive common decode window. The full table is in the README and `results/hy4-flash-table.md`.

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

### Reading the matrix

- **C1 decode** is 36.2 tok/s at 512, rises to 47.5 at 8K and 50.8 at 32K, then 52.9 at 131K, 45.5 at 200K and 54.7 at 250K. The variation at long context is DFly acceptance, not a monotonic context penalty.
- **C4 aggregate** is 117.1 / 114.6 / 89.4 / 136.4 tok/s at 512 / 2K / 8K / 32K. Above 32K the long prefills stagger the wave.
- **Effective prefill** is 1,164–1,530 tok/s at short context and 1,163–1,409 tok/s for C1 at 200K/250K. The 250K/C1 TTFT is 181.4 s; the 250K/C4 median TTFT is 483.9 s.
- **Long-context C4 is an admission receipt, not an aggregate throughput result.** All four requests completed at 131K/C4, 200K/C4 and 250K/C4, but peak client overlap was 2–3 and no common decode interval existed, so total and shared decode are left empty. The per-request decode medians are low because three requests decode while another prefill is running. The last request in each wave decodes alone at 41–42 tok/s at 200K/250K (17.4 tok/s at 131K). C1 and C2 are the practical long-context modes.
- **Capacity is not context.** The pool holds 1,501,761 tokens; four 250K requests fit in aggregate, but `max-num-seqs 4` and the 4,096-token prefill budget make the scheduler admit them in waves. A C4 wave at 250K is not four simultaneous full-speed streams.

### Raw per-request decode at the long C4 cells

| Cell | Request 0 | Request 1 | Request 2 | Request 3 | Median |
|---|---:|---:|---:|---:|---:|
| 131K / C4 | 1.68 | 3.29 | 14.56 | 17.36 | 8.9 |
| 200K / C4 | 1.61 | 41.34 | 1.62 | 4.81 | 3.2 |
| 250K / C4 | 41.69 | 1.60 | 1.70 | 4.30 | 3.0 |

The fast request is the one that finishes last and decodes after the other prefills have cleared. The slow requests are decoding while the next 131K–250K prefill is consuming the GPU.

## 5. Feature acceptance

`smoke.py` passed **13/13** checks on one boot. These are bounded acceptance checks, not quality scores.

| Check | Result | Evidence |
| --- | --- | --- |
| Health / models | pass | `/health` 200; `hy4-flash` listed |
| Arithmetic | pass | `19 + 23` → `42` |
| JSON schema | pass | valid city/temperature/conditions object |
| Tool call request + round trip | pass | `get_weather({"city":"Paris"})` |
| Single-image red square | pass | `Red` |
| Anthropic count_tokens | pass | 23 input tokens |
| Anthropic messages | pass | text `Hello`; 2.5 s |
| Prompt reuse, 32K | pass | 26.99 s → 1.10 s |
| Needle, 200K | pass | 200,024 prompt tokens, `ZEBRA-9174`, 166.2 s |
| Reasoning-effort scale | pass | 0 / 173 / 180 reasoning tokens |
| 4-way concurrency | pass | wall 0.47 s; four `OK n` replies |

### Chat-shaped decode

Five prompts, 300 forced output tokens, thinking off, median of three:

| Workload | TTFT | Decode tok/s | DFly mean acceptance |
|---|---:|---:|---:|
| JSON | 0.30–0.35 s | 44.8 | 2.94–3.09 |
| Code | 0.21–0.41 s | 45.6 | 2.94–3.09 |
| Rewrite | 0.32–0.43 s | 24.4 | — |
| Explain | 0.28–0.41 s | 28.9 | — |
| Story | 0.20–0.30 s | 16.9 | 1.61–1.81 |

The matrix's raw code-completion prompt accepts 5.45–5.50 tokens per draft step (89–90% draft acceptance); chat prose accepts 1.6–1.8 (12–16%). That is the main reason chat prose runs at 17–29 tok/s while the matrix code prompt runs at 36–55 tok/s. JSON/code chat still reaches ~45 tok/s.

## 6. 500K / 3-slot profile

A second profile serves **500,000 tokens with 3 admission slots**. It keeps
auto KV at 0.85 and lowers the prefill batch to 2048:

| Setting | 262K / 4 profile | 500K / 3 profile |
|---|---:|---:|
| `max-model-len` | 262,144 | **500,000** |
| `max-num-seqs` | 4 | **3** |
| `max-num-batched-tokens` | 4,096 | **2,048** |
| KV tokens at measured start | 1,501,761 | **1,522,579** |
| Concurrency at max window | 5.73× | **3.05×** |

The 500K pool leaves about 22K tokens of headroom for three full-length
requests. A 490,000-token C1 request completed with **488.8 s TTFT** and
**1,002.4 tok/s prefill**; three short concurrent requests passed at 0.898 s
wall. The 500K profile has its own compile-cache directory, so a cold first
boot may autotune for ~25 minutes. Receipt: `results/500k-profile.md`.

## 7. Limitations

- Forced-output completions are not a quality score.
- NVFP4 expert quantization is lossy; no GPQA Diamond or IFEval run was made.
- One synthetic image is qualified; multi-image and video are not.
- One 200k needle is a retrieval sanity check, not a long-context benchmark.
- The 1,048,576-token native window is not served. The recipe trades it for a
  262,144-token window and a 1.50M-token KV pool.
- No one-hour soak, multi-user agent trace or rollback exercise was run.

## 8. Reproduction

```bash
python3 benchmarks/capture_env.py
OUTPUT_TOKENS=512 \
PREFILL_SIZES=512,2048,8192,32768,131072,200000,250000 \
CONCURRENCIES=1,2,4 \
python3 benchmarks/matrix.py
./benchmarks/finalize.sh results/matrix-<timestamp>
python3 benchmarks/smoke.py
```

The KV/batch sweep transcript and per-cell logs from the tuning session are in
`results/kv-batch-sweep.md`.
