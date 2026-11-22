# Study protocol

This repository documents **one measured serving recipe** for Hy4-flash NVFP4
on two DGX Sparks. It is not a leaderboard entry and not a claim that the
recipe is optimal for every workload.

## Fixed platform

- Hardware: two NVIDIA DGX Spark GB10 systems, 128 GB unified memory each,
  joined by the QSFP link.
- Runtime: `dgx-spark-recipes/vllm-hy4-flash:77766db`, vLLM
  `0.1.dev25463+g77766db1f.d20261117`.
- Image: `vllm/vllm-openai@sha256:5e2c1bad…` plus the six Hy4-flash SM120 /
  ModelOpt files in `images/vllm-hy4-flash/`.
- Model: `atakans/Hy4-flash-NVFP4`, measured at revision
  `30f201acf0b4f9ba178817bcbc04ccfa3ec473f9`.
- Parallelism: TP2 with expert parallel; one rank per Spark.
- KV cache: auto-sized at `gpu-memory-utilization 0.85`; 1,501,761 tokens
  (18.26 GiB) at the measured start.
- Served window: 262,144 tokens per request.
- Speculative decoding: DFly, 5 draft tokens.
- Admission: `max-num-seqs 4`.

## Throughput matrix

- Prompts are built from exact token IDs, not text-and-hope token counts.
- Every request starts with a unique nonce so prefix caching cannot turn a
  cell into a cache-hit measurement.
- The matrix uses raw `/v1/completions` with `ignore_eos` and a forced output
  budget. Chat-template effects, reasoning blocks and tool parsing are
  therefore **not** part of the throughput number.
- `OUTPUT_TOKENS=512` keeps the sweep tractable on two Sparks. Decode is
  measured inside the common interval where all requests are emitting, not
  from total wall time.
- Input sizes are `512, 2,048, 8,192, 32,768, 131,072, 200,000, 250,000`.
  The last is 95% of the 262,144-token window.
- Concurrencies are `1, 2, 4`, matching the four admission slots.
- Cells whose aggregate prompt tokens exceed 1.3M are skipped rather than
  letting the scheduler queue or retract them. With this matrix the largest
  cell is 1.0M, so all cells run.
- TTFT is measured from request start to the first streamed usage event.
  Effective prefill includes queueing and any mixed decode work until the last
  request in the wave reaches its first token.

## Quality and feature checks

- Forced-output throughput is not a quality score.
- `smoke.py` is bounded acceptance, not an evaluation suite: arithmetic, JSON
  schema, one tool-call round trip, one image, Anthropic Messages, prompt
  reuse, a 200k needle, the reasoning-effort scale and four concurrent
  requests.
- The model is a vision-language checkpoint, but only one small synthetic
  image is qualified here. Multi-image and video are not.
- Thinking is on by default. `chat_template_kwargs {"reasoning_effort":
  "no_think"}` disables it; the other accepted levels are `high` and `max`.
  A small `max_tokens` budget can be consumed by the reasoning block and
  return no final content.

## Repeatability and reporting

- A difference below 5% between repeated key cells is treated as noise.
- Every published matrix cell has a raw event file; `common_window.py`
  recomputes the table without rerunning inference.
- `audit.py` checks exact prompt/output counts, successful completion, common
  decode windows and manifest hashes.
- Missing measurements remain unknown. No synthetic number is substituted.
- A cell can complete all requests yet have no positive common decode window;
  long prefills stagger the streams. Such a cell is reported as an
  admission/completion receipt and its aggregate decode is left empty.

## Capacity is not context

The KV pool holds 1,501,761 tokens. A single request can use up to 262,144;
four requests cannot each use 262,144. At 250K the pool fits six full
sessions; `max-num-seqs 4` is the admission cap, not a reservation.

## Open questions

- Quality after NVFP4 expert quantization on a real evaluation suite (GPQA
  Diamond, IFEval, long-context retrieval beyond one needle).
- Full 1,048,576-token native window. The checkpoint supports it; this recipe
  serves 262,144 and the KV pool would not hold a 1M-token request plus
  headroom at the current batch.
- Multi-image and video input.
- A one-hour soak, multi-user agent traces, and rollback exercise.
- Whether a larger KV pool with batch 2048 is worth the measured ~8% prefill
  cost for the intended workload.
