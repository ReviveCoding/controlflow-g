from __future__ import annotations

import json
from pathlib import Path

import yaml

from controlflow.v22.integrity import derive_integrity

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))
    namespace = config["development"]["namespace"]
    report = derive_integrity(ROOT, ledger_path=ROOT / f"artifacts/v22/{namespace}/validation.sqlite")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
