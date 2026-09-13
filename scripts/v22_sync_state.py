from __future__ import annotations

import json
from pathlib import Path

from controlflow.core.state import atomic_write_json, utc_now

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    v22 = json.loads((ROOT / "state/v22_execution_state.json").read_text(encoding="utf-8"))
    event = {
        "schema_version": 1,
        "timestamp": utc_now(),
        "program": "ControlFlow-G",
        "version": "2.2",
        "phase": v22["current_phase"],
        "status": v22["status"],
        "qualification_generation_permitted": v22["qualification_generation_permitted"],
        "final_holdout_creation_permitted": v22["final_holdout_creation_permitted"],
    }
    atomic_write_json(ROOT / "state/execution_state.json", event)
    journal_path = ROOT / "state/execution_journal.jsonl"
    with journal_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
