from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import yaml

from controlflow.v22.candidate import CandidateModelBundle


def _probed_vllm_version(root: Path) -> str:
    evidence = root / "state/v22_environment_evidence/wsl_probe.txt"
    for line in evidence.read_text(encoding="utf-8").splitlines():
        if '"vllm"' in line:
            return str(json.loads(line)["vllm"])
    raise RuntimeError("LIVE_VLLM_VERSION_EVIDENCE_MISSING")


def verify_live_bundle_server(root: Path, bundle: CandidateModelBundle) -> dict[str, Any]:
    """Fail closed unless the live process and endpoint match bundle bindings."""
    config = yaml.safe_load(bundle.artifact_path("serving_config").read_text(encoding="utf-8"))
    scalar_matches = all(
        (
            config["model"] == bundle.payload["qwen_model"],
            config["revision"] == bundle.payload["qwen_revision"],
            config["structured_output_backend"] == bundle.payload["structured_output_backend"],
            _probed_vllm_version(root) == bundle.payload["vllm_version"],
        )
    )
    endpoint_root = f"http://{config['host']}:{config['port']}"
    health = httpx.get(f"{endpoint_root}/v1/models", timeout=10)
    health.raise_for_status()
    served_models = {str(item["id"]) for item in health.json()["data"]}
    process_probe = subprocess.run(
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "bash",
            "-lc",
            f"pgrep -af 'vllm serve {bundle.payload['qwen_model']}'",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required_arguments = (
        str(bundle.payload["qwen_revision"]),
        f"--dtype {config['dtype']}",
        f"--gpu-memory-utilization {config['gpu_memory_utilization']}",
        f"--max-model-len {config['max_model_len']}",
        f"--max-num-seqs {config['concurrency']}",
        f"--structured-outputs-config.backend {bundle.payload['structured_output_backend']}",
        f"--served-model-name {bundle.payload['qwen_served_model']}",
        f"--host {config['host']}",
        f"--port {config['port']}",
    )
    verified = all(
        (
            scalar_matches,
            bundle.payload["qwen_served_model"] in served_models,
            all(item in process_probe for item in required_arguments),
        )
    )
    if not verified:
        raise RuntimeError("LIVE_VLLM_BUNDLE_MISMATCH")
    return {
        "config": config,
        "endpoint_root": endpoint_root,
        "served_models": sorted(served_models),
        "process_command": process_probe,
        "required_arguments": list(required_arguments),
        "verified": True,
    }
