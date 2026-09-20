"""Recompute contamination after generation exits, then admit one sealed identity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from controlflow.core.state import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v26/prior_runtime_manifest.yaml"


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("qualification", "final"), required=True)
    role = parser.parse_args().role
    identity = "V26FINAL" if role == "final" else "V26QUAL"
    manifest_path = ROOT / f"state/v26_{role}_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "ADMITTED_PENDING_TRANSITION" or manifest.get("dataset_identity") != identity:
        raise RuntimeError("V26_SEALED_TRANSITION_NOT_ELIGIBLE")
    data = ROOT / f"data/v26/{role}/{identity}"
    results = ROOT / f"results/v26/{role}"
    runtime = data / "runtime_cases.parquet"
    inventory = results / "prior_inventory.json"
    stored = results / "contamination.json"
    fresh = results / "contamination_recheck.json"
    receipt = results / "transition_receipt.json"
    trailing = [str(runtime), str(CONFIG), str(inventory), str(stored), str(fresh), str(manifest_path), str(receipt)]
    try:
        if sha256_file(data / "manifest.json") != manifest["dataset_manifest_sha256"]:
            raise RuntimeError("V26_SEALED_DATASET_MUTATED")
        command = [sys.executable, "-m", "scripts.v26_contamination"]
        subprocess.run([*command, "recheck", *trailing], cwd=ROOT, check=True)
        subprocess.run([*command, "verify", *trailing], cwd=ROOT, check=True)
        verified = _json(receipt)
        if verified["status"] != "PASS":
            raise RuntimeError("V26_SEALED_TRANSITION_FAILED")
        manifest["status"] = "ADMITTED_PENDING_FREEZE" if role == "final" else "ADMITTED_NOT_EXECUTED"
        manifest["fresh_contamination_sha256"] = sha256_file(fresh)
        manifest["transition_receipt_sha256"] = sha256_file(receipt)
        atomic_write_json(manifest_path, manifest)
    except BaseException:
        manifest = _json(manifest_path)
        manifest["status"] = "NO_PROMOTE"
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
