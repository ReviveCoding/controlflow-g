"""Record current V2.4 execution environment without changing it."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    gpu = _run("nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits")
    consumers = _run(
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"
    )
    wsl_python = _run("wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-lc", "python3 --version")
    vllm = _run(
        "wsl.exe",
        "-d",
        "Ubuntu-22.04",
        "--",
        "bash",
        "-lc",
        "/home/bjw-0/.venvs/controlflow-g-v2/bin/python -c 'import vllm; print(vllm.__version__)'",
    )
    inventory = _run(
        "wsl.exe",
        "-d",
        "Ubuntu-22.04",
        "--",
        "bash",
        "-lc",
        "/home/bjw-0/.venvs/controlflow-g-v2/bin/python -m pip freeze",
    )
    inventory_path = ROOT / "state/v24_wsl_pip_freeze.txt"
    inventory_path.write_text(inventory + "\n", encoding="utf-8")
    atomic_write_json(
        ROOT / "state/v24_environment_manifest.json",
        {
            "schema_version": 1,
            "observed_at": utc_now(),
            "host_python": sys.version,
            "host_platform": platform.platform(),
            "gpu_query": gpu,
            "compute_consumers_at_probe": consumers.splitlines() if consumers else [],
            "wsl_distribution": "Ubuntu-22.04",
            "wsl_python": wsl_python,
            "vllm_version": vllm,
            "wsl_dependency_inventory_sha256": sha256_file(inventory_path),
            "dependency_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
    )
    print(json.dumps({"gpu": gpu, "vllm_version": vllm, "compute_consumers": consumers.splitlines()}))


if __name__ == "__main__":
    main()
