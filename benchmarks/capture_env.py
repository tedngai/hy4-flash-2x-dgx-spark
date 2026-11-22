#!/usr/bin/env python3
"""Capture the identity and launch configuration of the running Hy4-flash TP2 recipe.

Writes `results/launch.json` (hashed by the matrix) and `results/config.json`
(the short published configuration). The head and the worker are both recorded.
No `.env` file is read and no environment values are printed.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
CONTAINER = os.environ.get("CONTAINER", "hy4-flash-nvfp4-vllm-tp2")
WORKER = os.environ.get("WORKER", "192.0.2.11")
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
MODEL_REPO = "atakans/Hy4-flash-NVFP4"
CACHE_MODEL = HF_HOME / "hub" / "models--atakans--Hy4-flash-NVFP4"


def run(command: str, timeout: int = 30) -> str:
    try:
        return subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True,
            timeout=timeout,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - capture, do not fail the report
        return f"<unavailable: {error}>"


def remote(command: str, timeout: int = 30) -> str:
    try:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", WORKER, command],
            check=True, text=True, capture_output=True, timeout=timeout,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"<unavailable: {error}>"


def get_json(path: str) -> dict:
    with urllib.request.urlopen(API_BASE + path, timeout=20) as response:
        return json.loads(response.read())


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def tree_sha256(folder: Path) -> str | None:
    if not folder.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


model = get_json("/v1/models")["data"][0]
snapshot = Path(model["root"])
revision = snapshot.name
if not snapshot.exists():
    matches = sorted(CACHE_MODEL.glob("snapshots/*"))
    if not matches:
        raise SystemExit(f"model snapshot not found under {CACHE_MODEL}")
    snapshot = matches[-1]
    revision = snapshot.name
try:
    version = get_json("/version").get("version")
except Exception:  # noqa: BLE001
    version = "<unavailable>"
metrics = urllib.request.urlopen(API_BASE + "/metrics", timeout=20).read().decode()
cache_line = next(
    (line for line in metrics.splitlines() if line.startswith("vllm:cache_config_info{")),
    "",
)
cache_info = dict(re.findall(r'(\w+)="([^"]*)"', cache_line))

processes = run(f"docker exec {CONTAINER} ps -eo pid,args")
command = next(
    (line.split(None, 1)[1] for line in processes.splitlines() if "vllm serve" in line),
    "<not found>",
)
worker_processes = remote(
    f"docker exec {CONTAINER} ps -eo pid,args"
)
worker_command = next(
    (line.split(None, 1)[1] for line in worker_processes.splitlines() if "vllm serve" in line),
    "<not found>",
)
image_id = run(f"docker inspect {CONTAINER} --format '{{{{.Image}}}}'")
image_tag = run(f"docker inspect {CONTAINER} --format '{{{{.Config.Image}}}}'")
image_digests = run(
    f"docker image inspect {image_id} --format '{{{{json .RepoDigests}}}}'"
)
worker_image_id = remote(f"docker inspect {CONTAINER} --format '{{{{.Image}}}}'")
worker_image_tag = remote(f"docker inspect {CONTAINER} --format '{{{{.Config.Image}}}}'")
worker_image_digests = remote(
    f"docker image inspect {worker_image_id} --format '{{{{json .RepoDigests}}}}'"
)

dockerfile = (ROOT / "images" / "vllm-hy4-flash" / "Dockerfile").read_text()
base_image = re.search(r"^FROM\s+(\S+)", dockerfile, re.M)
base_image = base_image.group(1) if base_image else None
image_context_sha256 = tree_sha256(ROOT / "images" / "vllm-hy4-flash")

gpu_query = "name,memory.total,driver_version"
launch = {
    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "recipe": "Hy4-flash NVFP4 on 2x DGX Spark (GB10), TP2 + expert parallel, DFly 5",
    "hostname": run("hostname"),
    "uname": run("uname -a"),
    "hardware": "2x NVIDIA DGX Spark GB10, 128 GB unified memory each, QSFP link",
    "model": model["id"],
    "model_root": model.get("root"),
    "model_repo": MODEL_REPO,
    "model_revision": revision,
    "model_snapshot": str(snapshot),
    "config_sha256": sha256(snapshot / "config.json"),
    "tokenizer_sha256": sha256(snapshot / "tokenizer.json"),
    "chat_template_sha256": sha256(snapshot / "chat_template.jinja"),
    "quant_config_sha256": sha256(snapshot / "hf_quant_config.json"),
    "image": image_tag,
    "image_id": image_id,
    "image_digests": image_digests,
    "image_context_sha256": image_context_sha256,
    "base_image": base_image,
    "worker": {
        "host": WORKER,
        "hostname": remote("hostname"),
        "image": worker_image_tag,
        "image_id": worker_image_id,
        "image_digests": worker_image_digests,
        "command": worker_command,
        "gpu": remote(
            f"nvidia-smi --query-gpu={gpu_query} --format=csv,noheader"
        ),
        "host_mem": remote("free -m"),
        "kernel": remote("uname -a"),
    },
    "head": {
        "hostname": run("hostname"),
        "gpu": run(f"nvidia-smi --query-gpu={gpu_query} --format=csv,noheader"),
        "host_mem": run("free -m"),
    },
    "vllm_version": version,
    "command": command,
    "cache_config_info": cache_info,
    "container_mem_usage": run(
        f"docker stats --no-stream --format '{{{{.MemUsage}}}}' {CONTAINER}"
    ),
}
(RESULTS / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")


def flag_value(flag: str, default: str = "—") -> str:
    """Read a vLLM flag from the captured command line."""
    try:
        parts = command.split()
        index = parts.index(flag)
        return parts[index + 1]
    except (ValueError, IndexError):
        return default


specs = re.findall(r"--speculative-config\s+(\{.*?\})\s+--", command + " --")
config = {
    "model": MODEL_REPO,
    "model_revision": revision,
    "served_model_name": model["id"],
    "runtime_image": image_tag,
    "runtime_image_id": image_id,
    "base_image": base_image,
    "vllm_version": version,
    "hardware": "2x NVIDIA DGX Spark (GB10), 128 GB unified memory each, QSFP link",
    "tensor_parallel": flag_value("--tensor-parallel-size"),
    "enable_expert_parallel": "--enable-expert-parallel" in command,
    "max_model_len": int(flag_value("--max-model-len", "0") or 0),
    "max_num_seqs": int(flag_value("--max-num-seqs", "0") or 0),
    "max_num_batched_tokens": int(flag_value("--max-num-batched-tokens", "0") or 0),
    "gpu_memory_utilization": float(flag_value("--gpu-memory-utilization", "0") or 0),
    "kv_cache_memory_bytes": (
        int(cache_info["kv_cache_memory_bytes"])
        if cache_info.get("kv_cache_memory_bytes", "None") != "None" else None
    ),
    "kv_cache_size_tokens": int(cache_info.get("kv_cache_size_tokens", 0) or 0),
    "kv_cache_max_concurrency_at_max_len": float(
        cache_info.get("kv_cache_max_concurrency", 0) or 0
    ),
    "kv_cache_dtype": cache_info.get("cache_dtype"),
    "block_size": int(cache_info.get("block_size", 0) or 0),
    "enable_prefix_caching": cache_info.get("enable_prefix_caching") == "True",
    "speculative_config": json.loads(specs[-1]) if specs else None,
    "quantization": "modelopt NVFP4 experts; FP8 Engram tables; BF16 elsewhere",
    "output_tokens": int(os.environ.get("OUTPUT_TOKENS", "512")),
    "ignore_eos": True,
}
(RESULTS / "config.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps(config, indent=2))
