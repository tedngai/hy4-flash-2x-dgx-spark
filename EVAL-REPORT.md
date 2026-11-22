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
| 250,000 (matrix C1) | see matrix | see matrix |

All returned normally with health 200 after each. The 250K matrix cell is the
end-to-end receipt for the served window.

## 4. Burst-serving matrix

<!-- MATRIX-SECTION -->

## 5. Feature acceptance

<!-- FEATURE-SECTION -->

## 6. Limitations

- Forced-output completions are not a quality score.
- NVFP4 expert quantization is lossy; no GPQA Diamond or IFEval run was made.
- One synthetic image is qualified; multi-image and video are not.
- One 200k needle is a retrieval sanity check, not a long-context benchmark.
- The 1,048,576-token native window is not served. The recipe trades it for a
  262,144-token window and a 1.50M-token KV pool.
- No one-hour soak, multi-user agent trace or rollback exercise was run.

## 7. Reproduction

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
