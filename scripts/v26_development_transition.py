"""One-shot fresh development generation and separate-process transition rehearsal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from controlflow.core.state import atomic_write_json, sha256_file, utc_now
from controlflow.v26.admission import structural_admission
from controlflow.v26.generator import generate_v26_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v26/prior_runtime_manifest.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    if seed < 26011:
        raise ValueError("V26_DEVELOPMENT_SEED_INVALID")
    data = ROOT / f"data/v26/development/rehearsal_{seed}"
    result_dir = ROOT / f"results/v26/development/rehearsal_{seed}"
    state = ROOT / f"state/v26_development_rehearsal_{seed}_manifest.json"
    if data.exists() or result_dir.exists() or state.exists():
        raise RuntimeError("V26_DEVELOPMENT_IDENTITY_ALREADY_OPENED")
    atomic_write_json(
        state,
        {
            "schema_version": 1,
            "status": "GENERATION_OPENED",
            "seed": seed,
            "created_at": utc_now(),
            "one_shot_opened": True,
        },
    )
    try:
        dataset = generate_v26_split(data, role="DEVELOPMENT", seed=seed, root=ROOT, count=60, prefix=f"V26DEV{seed}")
        result_dir.mkdir(parents=True)
        runtime = data / "runtime_cases.parquet"
        inventory = result_dir / "prior_inventory.json"
        stored = result_dir / "contamination.json"
        fresh = result_dir / "contamination_recheck.json"
        receipt = result_dir / "transition_receipt.json"
        command = [sys.executable, str(ROOT / "scripts/v26_contamination.py")]
        trailing = [str(runtime), str(CONFIG), str(inventory), str(stored), str(fresh), str(state), str(receipt)]
        subprocess.run([*command, "generate", *trailing], cwd=ROOT, check=True)
        contamination = json.loads(stored.read_text(encoding="utf-8"))
        admission = structural_admission(
            data, ROOT, contamination, result_dir / "structural_admission.json", role="DEVELOPMENT"
        )
        atomic_write_json(
            state,
            {
                "schema_version": 1,
                "status": "ADMITTED_NOT_EXECUTED" if admission["status"] == "ADMITTED" else "V26_GENERATION_INVALID",
                "seed": seed,
                "one_shot_opened": True,
                "dataset": dataset,
                "dataset_manifest_sha256": sha256_file(data / "manifest.json"),
                "contamination_sha256": sha256_file(stored),
                "contamination_semantic_digest": contamination["semantic_digest"],
                "prior_inventory_sha256": sha256_file(inventory),
                "structural_admission": admission,
            },
        )
        if admission["status"] != "ADMITTED":
            raise RuntimeError(f"V26_DEVELOPMENT_ADMISSION_INVALID:{admission['failures']}")
        # The generation process has finished before this fresh process starts.
        subprocess.run([*command, "recheck", *trailing], cwd=ROOT, check=True)
        subprocess.run([*command, "verify", *trailing], cwd=ROOT, check=True)
        record = json.loads(receipt.read_text(encoding="utf-8"))
        current = json.loads(state.read_text(encoding="utf-8"))
        current["status"] = "TRANSITION_PASS" if record["status"] == "PASS" else "V26_DEVELOPMENT_NO_GO"
        current["transition_receipt_sha256"] = sha256_file(receipt)
        current["fresh_contamination_sha256"] = sha256_file(fresh)
        atomic_write_json(state, current)
    except BaseException:
        current = json.loads(state.read_text(encoding="utf-8"))
        current["status"] = "V26_DEVELOPMENT_NO_GO"
        atomic_write_json(state, current)
        raise


if __name__ == "__main__":
    main()
