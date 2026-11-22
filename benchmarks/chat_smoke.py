#!/usr/bin/env python3
"""Chat-shaped decode smoke for Hy4-flash, thinking off.

Runs five prompt types three times each with a 300-token output budget and
records TTFT and decode tokens/second. This is the model-card workload, not the
forced-output matrix in `matrix.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
API = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "hy4-flash")
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "300"))
REPEATS = int(os.environ.get("REPEATS", "3"))

PROMPTS = {
    "json": "Return a JSON object for a weather report for Paris with city, "
            "temperature_c, conditions and a three-day forecast array. Return JSON only.",
    "code": "Write a Python function that merges two sorted lists and include "
            "three test cases. Return code only.",
    "rewrite": "Rewrite this paragraph in a concise, professional style: "
               "The team looked at the numbers and it seemed like the launch was "
               "probably going to happen later than everyone first thought because "
               "several dependencies were not ready yet.",
    "explain": "Explain how a bloom filter works, including false positives, "
               "sizing and two practical uses.",
    "story": "Write a short story about a lighthouse keeper who finds a message "
             "in a bottle.",
}


def one(name: str, prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
    }
    request = urllib.request.Request(
        API + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start = time.time()
    first = None
    usage = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if (delta.get("content") or delta.get("reasoning")) and first is None:
                    first = time.time()
    end = time.time()
    completion = (usage or {}).get("completion_tokens", 0)
    return {
        "prompt": name,
        "ttft_s": round(first - start, 3) if first else None,
        "wall_s": round(end - start, 3),
        "completion_tokens": completion,
        "decode_tok_s": (
            round((completion - 1) / (end - first), 2)
            if first and completion > 1 and end > first else None
        ),
    }


def main() -> int:
    rows = []
    for name, prompt in PROMPTS.items():
        for _ in range(REPEATS):
            row = one(name, prompt)
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for name in PROMPTS:
        rates = [r["decode_tok_s"] for r in rows if r["prompt"] == name and r["decode_tok_s"]]
        summary[name] = round(statistics.median(rates), 2) if rates else None
    result = {
        "model": MODEL,
        "output_tokens": OUTPUT_TOKENS,
        "repeats": REPEATS,
        "reasoning_effort": "no_think",
        "median_decode_tok_s": summary,
        "rows": rows,
    }
    (RESULTS / "chat-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"median_decode_tok_s": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
