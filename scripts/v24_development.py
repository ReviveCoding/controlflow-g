"""Development-only stratification tests; never creates a sealed holdout."""

from __future__ import annotations

import json
from pathlib import Path

from controlflow.core.state import atomic_write_json, utc_now
from controlflow.v24.admission import structural_admission
from controlflow.v24.generator import generate_v24_split

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (24201, 24202, 24203)


def main() -> None:
    outcomes = []
    for seed in SEEDS:
        directory = ROOT / f"data/v24/development/seed_{seed}_v3"
        if not directory.exists():
            generate_v24_split(directory, role="QUALIFICATION", seed=seed, root=ROOT, prefix=f"V24DEV{seed}")
        output = ROOT / f"results/v24/development_admission_{seed}_v3.json"
        result = structural_admission(directory, ROOT, {"leakage_findings": 0}, output)
        outcomes.append({"seed": seed, "status": result["status"], "counts": result.get("independent_counts")})
        if result["status"] != "ADMITTED":
            raise RuntimeError(f"V24_DEVELOPMENT_GENERATOR_INVALID:{seed}:{result['failures']}")
    atomic_write_json(
        ROOT / "results/v24/development_generator_validation.json",
        {"schema_version": 1, "created_at": utc_now(), "role": "DEVELOPMENT_ONLY", "outcomes": outcomes},
    )
    print(json.dumps(outcomes))


if __name__ == "__main__":
    main()
