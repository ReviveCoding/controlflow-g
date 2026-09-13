from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v23.telemetry import sample_nvidia

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "state/v23_environment_evidence"


def run(name: str, argv: list[str], timeout: int = 90) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    try:
        process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        stdout, stderr, exit_code = process.stdout, process.stderr, process.returncode
    except Exception as exc:
        stdout, stderr, exit_code = "", f"{type(exc).__name__}: {exc}", None
    path = EVIDENCE / f"{name}.txt"
    path.write_text(stdout + (f"\nSTDERR:\n{stderr}" if stderr else ""), encoding="utf-8")
    return {
        "name": name,
        "argv": argv,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "exit_code": exit_code,
        "evidence_path": path.relative_to(ROOT).as_posix(),
        "evidence_sha256": sha256_file(path),
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    disk = psutil.disk_usage(str(ROOT))
    ram = psutil.virtual_memory()
    consumers = run(
        "nvidia_processes",
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    wsl = run(
        "wsl_probe",
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "bash",
            "-lc",
            'uname -a; "$HOME/.venvs/controlflow-g-v2/bin/python" -c \'import json,torch,vllm; '
            'print(json.dumps({"torch":torch.__version__,"cuda_compiled":torch.version.cuda,'
            '"cuda_available":torch.cuda.is_available(),"vllm":vllm.__version__}))\'; '
            "df -B1 /mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2; free -b",
        ],
    )
    help_probe = run(
        "vllm_help_relevant",
        [
            "wsl.exe",
            "-d",
            "Ubuntu-22.04",
            "--",
            "bash",
            "-lc",
            '"$HOME/.venvs/controlflow-g-v2/bin/vllm" serve --help=all 2>&1 | tr -d "\\r" | '
            'grep -i -C 2 -E "performance.mode|batched.tokens|prefix.cach|optimization.level|'
            'speculative|per.request.metrics"',
        ],
    )
    consumer_text = (ROOT / consumers["evidence_path"]).read_text(encoding="utf-8").strip()
    budget = yaml.safe_load((ROOT / "configs/resource_budget.yaml").read_text(encoding="utf-8"))
    reserve = int(budget["minimum_free_space_gib"]) * 1024**3
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "probe_status": "COMPLETE" if all(x["exit_code"] == 0 for x in (consumers, wsl, help_probe)) else "PARTIAL",
        "gpu": sample_nvidia(),
        "background_gpu_consumers": [line for line in consumer_text.splitlines() if line.strip()],
        "gpu_workload_admission": "DEFER_GPU" if consumer_text else "AVAILABLE_AT_PROBE",
        "ram_total_bytes": ram.total,
        "ram_available_bytes": ram.available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "minimum_free_space_bytes": reserve,
        "free_space_reserve_satisfied": disk.free >= reserve,
        "resource_budget_sha256": sha256_file(ROOT / "configs/resource_budget.yaml"),
        "commands": [consumers, wsl, help_probe],
    }
    atomic_write_json(ROOT / "state/v23_environment_manifest.json", manifest)
    if not manifest["free_space_reserve_satisfied"]:
        raise RuntimeError("RESOURCE_RESERVE_VIOLATION")


if __name__ == "__main__":
    main()
