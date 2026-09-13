from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlflow.core.state import atomic_write_json, utc_now

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    path = ROOT / "state/v22_final_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("one_shot_opened") and not manifest.get("one_shot_executed"):
        raise RuntimeError("FINAL_INVALIDATION_REQUIRES_OPENED_FINAL")
    manifest.update(
        {
            "status": "INVALID",
            "invalidated_at": utc_now(),
            "invalidation_reason": args.reason,
            "rerun_same_final_permitted": False,
        }
    )
    atomic_write_json(path, manifest)


if __name__ == "__main__":
    main()
