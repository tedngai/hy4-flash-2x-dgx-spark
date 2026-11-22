# KV pool × prefill-batch sweep

Measured 2026-11-22 on the same two Sparks with the published image and
checkpoint. Each cell restarts the TP2 pair, waits for `/v1/models` 200, records
the vLLM KV lines, runs a short chat smoke test, then stops the pair.

`auto` means `--kv-cache-memory-bytes` is omitted; `--gpu-memory-utilization
0.85` sizes the pool.

| KV cap | Batch | KV memory | KV tokens | Concurrency @262K | Host available | Health |
|---:|---:|---:|---:|---:|---:|---:|
| 6G | 8192 | 6.0 GiB | 578,345 | 2.21× | 14 GB | 200 |
| 10G | 8192 | 10.0 GiB | 963,984 | 3.68× | 10 GB | 200 |
| 10G | 4096 | 10.0 GiB | 1,020,326 | 3.89× | 12 GB | 200 |
| 12G | 4096 | 12.0 GiB | 1,224,535 | 4.67× | 12 GB | 200 |
| 12G | 2048 | 12.0 GiB | 1,261,398 | 4.81× | 11 GB | 200 |
| 14G | 2048 | 14.0 GiB | 1,471,508 | 5.61× | 10 GB | 200 |
| 14G | 1024 | 14.0 GiB | 1,493,995 | 5.70× | 9 GB | 200 |
| 16G | 2048 | 16.0 GiB | 1,681,864 | 6.42× | 8 GB | 200 |
| 16G | 1024 | 16.0 GiB | 1,707,566 | 6.51× | 8 GB | 200 |
| **auto** | **4096** | **18.26 GiB** | **1,501,761** | **5.73×** | **7 GB** | **200** |
| auto | 2048 | 19.08 GiB | 1,741,790 | 6.64× | 8 GB | 200 |
| auto | 8192 | 16.5–16.55 GiB | 1,166,868–1,199,665 | 4.45–4.58× | 10 GB | 200 |

## Reading the sweep

- The published 6G cap is a hard reservation. It overrides
  `gpu-memory-utilization`, so the pool cannot grow while the flag is present.
- At a fixed 10G cap, dropping the batch from 8192 to 4096 raised KV tokens by
  5.8% and recovered two GiB of host memory.
- With the cap removed, batch 8192 yields 16.5 GiB / 1.20M tokens, batch 4096
  yields 18.26 GiB / 1.50M tokens, and batch 2048 yields 19.08 GiB / 1.74M
  tokens. The batch reduction is what creates the headroom.
- The 32K prefill cost is ~1% from 8192 to 4096 (1,568 → 1,551 tok/s) and ~8%
  from 4096 to 2048 (1,551 → 1,426 tok/s). 2048 → 1024 adds only ~1.5% KV
  tokens, so it was not adopted.
- Every cell reached health 200 and passed the short chat smoke test.

## Receipts

- `kv-batch-sweep.log` — full sweep transcript.
- `kv-batch/cell-<cap>-<batch>.log` — the KV lines, host memory and health for
  each cell.
- `../benchmarks/kv_batch_sweep.sh` — the reproducible sweep driver.

The sweep was run from the upstream `dgx-spark-recipes` checkout before this
project was split out; the recipe and launcher are vendored here unchanged
apart from the tuned recipe values.
