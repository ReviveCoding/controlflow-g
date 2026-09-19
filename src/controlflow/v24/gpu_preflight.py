"""Verify that the exact repository-owned server exclusively occupies the GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _descends(pid: int, ancestor: int, parents: dict[int, int]) -> bool:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = parents.get(pid, 0)
    return False


def verify_gpu_ownership(root: Path, server_pid: int) -> dict[str, Any]:
    lock = root / "state/gpu.lock/owner.pid"
    if not lock.is_file() or int(lock.read_text(encoding="utf-8").strip()) != server_pid:
        raise RuntimeError("V24_GPU_SEMAPHORE_OWNER_INVALID")
    command = (
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits; "
        "printf 'PARENT_TABLE_BEGIN\\n'; ps -eo pid=,ppid="
    )
    result = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-lc", command],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    gpu_lines, separator, process_lines = result.partition("PARENT_TABLE_BEGIN")
    if not separator:
        raise RuntimeError("V24_GPU_PROCESS_INVENTORY_INVALID")
    consumers = [int(line.strip()) for line in gpu_lines.splitlines() if line.strip().isdigit()]
    parents = {
        int(parts[0]): int(parts[1])
        for line in process_lines.splitlines()
        if len(parts := line.split()) == 2 and all(part.isdigit() for part in parts)
    }
    if not consumers or any(not _descends(pid, server_pid, parents) for pid in consumers):
        raise RuntimeError(f"V24_UNRELATED_GPU_CONSUMER_OR_NO_SERVER:{consumers}")
    return {"server_pid": server_pid, "gpu_compute_pids": consumers, "all_server_descendants": True}
