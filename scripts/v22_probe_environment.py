from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from controlflow.core.state import atomic_write_json, sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str, argv: list[str], evidence_dir: Path) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    try:
        process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)
        stdout, stderr, exit_code = process.stdout, process.stderr, process.returncode
    except Exception as exc:
        stdout, stderr, exit_code = "", f"{type(exc).__name__}: {exc}", None
    path = evidence_dir / f"{name}.txt"
    path.write_text(stdout + ("\nSTDERR:\n" + stderr if stderr else ""), encoding="utf-8")
    return {
        "name": name,
        "argv": argv,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "exit_code": exit_code,
        "evidence_path": path.relative_to(ROOT).as_posix(),
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-wsl", action="store_true")
    args = parser.parse_args()
    evidence_dir = ROOT / "state/v22_environment_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    python_code = (
        "import json,platform,psutil,importlib.metadata as m; import torch,xgboost; "
        "d=psutil.disk_usage('.'); r=psutil.virtual_memory(); "
        "print(json.dumps({'platform':platform.platform(),'python':platform.python_version(),"
        "'torch':torch.__version__,'torch_compiled_cuda':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),"
        "'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,"
        "'xgboost':xgboost.__version__,'cryptography':m.version('cryptography'),"
        "'ram_total_bytes':r.total,'ram_available_bytes':r.available,'disk_total_bytes':d.total,"
        "'disk_free_bytes':d.free},sort_keys=True))"
    )
    commands = [
        _run("windows_ver", ["cmd.exe", "/c", "ver"], evidence_dir),
        _run("host_python", [str(ROOT / ".venv/Scripts/python.exe"), "-c", python_code], evidence_dir),
        _run(
            "nvidia_gpu",
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,uuid,name,memory.total,memory.free,memory.used,driver_version,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            evidence_dir,
        ),
        _run(
            "nvidia_processes",
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            evidence_dir,
        ),
    ]
    if args.include_wsl:
        commands.extend(
            [
                _run("wsl_list", ["wsl.exe", "--list", "--verbose"], evidence_dir),
                _run(
                    "wsl_probe",
                    [
                        "wsl.exe",
                        "-d",
                        "Ubuntu-22.04",
                        "--",
                        "bash",
                        "-lc",
                        """uname -a; python3 --version;
"$HOME/.venvs/controlflow-g-v2/bin/python" -c '
import json,torch,vllm,xgboost
print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,
"cuda_available":torch.cuda.is_available(),"vllm":vllm.__version__,
"xgboost":xgboost.__version__}))
';
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,driver_version \
--format=csv,noheader,nounits;
df -B1 /mnt/c/Users/bjw-0/Downloads/ControlFlow-G-v2;
free -b""",
                    ],
                    evidence_dir,
                ),
            ]
        )
    host = next(item for item in commands if item["name"] == "host_python")
    parsed_host = json.loads(host["stdout"]) if host["exit_code"] == 0 else None
    gpu = next(item for item in commands if item["name"] == "nvidia_gpu")
    processes = next(item for item in commands if item["name"] == "nvidia_processes")
    wsl_probe = next((item for item in commands if item["name"] == "wsl_probe"), None)
    budget = yaml.safe_load((ROOT / "configs/resource_budget.yaml").read_text(encoding="utf-8"))
    reserve_gib = int(budget["minimum_free_space_gib"])
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "probe_status": "COMPLETE" if all(item["exit_code"] == 0 for item in commands) else "PARTIAL",
        "host": parsed_host,
        "nvidia_gpu_raw": gpu["stdout"].strip() if gpu["exit_code"] == 0 else None,
        "background_gpu_consumers": [line for line in processes["stdout"].splitlines() if line.strip()],
        "gpu_workload_admission": "DEFER_GPU" if processes["stdout"].strip() else "AVAILABLE_AT_PROBE",
        "wsl_probe_raw": None if wsl_probe is None or wsl_probe["exit_code"] != 0 else wsl_probe["stdout"],
        "resource_budget_sha256": sha256_file(ROOT / "configs/resource_budget.yaml"),
        "minimum_free_space_gib": reserve_gib,
        "free_space_reserve_satisfied": bool(parsed_host and parsed_host["disk_free_bytes"] >= reserve_gib * 1024**3),
        "commands": [
            {key: value for key, value in item.items() if key not in {"stdout", "stderr"}} for item in commands
        ],
    }
    atomic_write_json(ROOT / "state/v22_environment_manifest.json", manifest)


if __name__ == "__main__":
    main()
