# Results

Every published number has a raw receipt here.

| Path | Contents |
|---|---|
| `launch.json` | Launch identity: both TP2 ranks, model revision, tokenizer/config/chat-template hashes, image ID and build-context hash, exact command, KV cache config, host state. |
| `config.json` | Short published configuration. |
| `matrix-<timestamp>/<size>-c<n>.json` | Raw per-request events for one input-length / concurrency cell. |
| `matrix-<timestamp>/summary.json` | One row per completed cell. |
| `matrix-<timestamp>/TABLE.md` | Rendered table. |
| `hy4-flash-summary.json` | Stable copy of the matrix summary. |
| `hy4-flash-table.md` | Stable copy of the table. |
| `feature-acceptance.json` | Smoke checks, including the 200k needle. |
| `needle-200k.json` | Long-context retrieval receipt. |
| `reasoning-effort.json` | `no_think` / `high` / `max` reasoning-token probe. |
| `concurrency-4.json` | Four simultaneous requests against the four admission slots. |
| `chat-smoke.json` | Five chat-shaped prompt types × 3, thinking off, with TTFT and decode tok/s. |
| `smoke-transcript.txt`, `chat-smoke-transcript.txt` | Human-readable check output. |
| `kv-batch-sweep.md`, `kv-batch/`, `kv-batch-sweep.log` | KV-cap × prefill-batch tuning receipts. |
| `protocol-audit.json` | Structural audit of the matrix (counts, hashes, failures). |

The matrix is a throughput and latency test, not a quality score. Decode is
measured inside the common interval where all requests are emitting; when long
prefills stagger a wave, the aggregate decode columns are left empty rather
than fabricated.
