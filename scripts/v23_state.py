from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlflow.core.state import atomic_write_json, utc_now

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete", required=True)
    parser.add_argument("--next", required=True)
    parser.add_argument("--status", default="V23_DEVELOPMENT_IN_PROGRESS")
    args = parser.parse_args()
    path = ROOT / "state/v23_execution_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    phases = list(state.get("completed_phases", []))
    newly_completed = args.complete not in phases
    if newly_completed:
        phases.append(args.complete)
    now = utc_now()
    state.update(
        {
            "updated_at": now,
            "status": args.status,
            "current_phase": args.next,
            "completed_phases": phases,
        }
    )
    atomic_write_json(path, state)
    global_state = {
        "schema_version": 1,
        "timestamp": now,
        "program": "ControlFlow-G",
        "version": "2.3",
        "phase": args.next,
        "status": args.status,
        "qualification_generation_permitted": bool(state.get("qualification_generation_permitted", False)),
        "final_holdout_creation_permitted": bool(state.get("final_generation_permitted", False)),
    }
    atomic_write_json(ROOT / "state/execution_state.json", global_state)
    if newly_completed:
        journal = ROOT / "state/execution_journal.jsonl"
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**global_state, "completed_phase": args.complete}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
