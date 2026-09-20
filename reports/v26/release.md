# ControlFlow-G V2.6 synthetic research release

Decision: **PROMOTE** the reproducible synthetic research artifact. This does not authorize real-bank deployment or financial actions.

V26QUAL (seed 26061) and the one-shot V26FINAL (seed 26062) each completed the sealed path. The final run used the frozen Qwen3/vLLM configuration at observed maximum concurrency 2. All 17 predeclared final gates passed; the independent correctness, methodology, and security/reliability outcome audits found no BLOCKER or HIGH finding.

| Frozen measure | V26QUAL | V26FINAL | Gate |
| --- | ---: | ---: | ---: |
| Core STC | 491/600 (0.8183) | 489/600 (0.8150) | ≥ 0.80 |
| Binary critical recall | 134/136 (0.9853) | 141/144 (0.9792) | ≥ 0.95 |
| Typed critical recall | 134/136 (0.9853) | 141/144 (0.9792) | ≥ 0.92 |
| Stale-policy errors | 0/150 | 0/150 | ≤ 0.02 rate |
| Structured-output failures | 0/600 | 0/600 | ≤ 0.01 rate |
| Unauthorized committed actions | 0/81 | 0/83 | 0 |
| Approval-bypass commits | 0/255 | 0/255 | 0 |
| Total p95 latency, concurrency 2 | 6.833 s | 7.392 s | ≤ 15 s |

Final structural admission was ADMITTED for 600 cases. The stored full contamination-report hash authenticated the generation artifact; the independently recomputed semantic digest matched, while volatile report provenance remained distinct. The typed producer checkpoint covered all 600 cases. Independent denominators were VALID with zero aggregate mismatch. SQLite passed integrity and foreign-key checks, recorded 255 review commits and 255 matching approval consumptions, completed `wal_checkpoint(TRUNCATE)`, had no WAL/SHM sidecars after close, and retained a stable file hash. The artifact scan was CLEAN; post-close verification, closure receipt, and terminal verifier passed.

All figures above are from `results/v26/{qualification,final}/gate_decision.json`, `results/v26/final/denominators.json`, `results/v26/final/sqlite_finalization.json`, and the bound manifests and receipts in `state/`. Static/type checks and the full 240-test suite passed before V26QUAL in `state/v26_static_quality.json`. The exact evidence hashes and decision are recorded in `state/v26_release_decision.json`.
