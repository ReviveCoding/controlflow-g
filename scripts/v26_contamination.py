"""Separate-process contamination generation and transition verification."""

from __future__ import annotations

import argparse
from pathlib import Path

from controlflow.core.state import atomic_write_json
from controlflow.v26.contamination_evidence import generate_evidence
from controlflow.v26.transition import verify_transition

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "recheck", "verify"))
    parser.add_argument("candidate", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("stored", type=Path)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        result = verify_transition(
            ROOT,
            candidate_path=args.candidate,
            config_path=args.config,
            inventory_path=args.inventory,
            stored_path=args.stored,
            fresh_path=args.fresh,
            manifest_path=args.manifest,
            receipt_path=args.receipt,
        )
        if result["status"] != "PASS":
            raise RuntimeError(f"V26_TRANSITION_INVALID:{result['failures']}")
        return
    evidence, inventory = generate_evidence(ROOT, args.candidate, args.config)
    output = args.stored if args.mode == "generate" else args.fresh
    atomic_write_json(output, evidence.model_dump(mode="json"))
    if args.mode == "generate":
        atomic_write_json(args.inventory, [entry.model_dump() for entry in inventory])


if __name__ == "__main__":
    main()
