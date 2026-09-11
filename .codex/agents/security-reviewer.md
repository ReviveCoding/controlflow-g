# Security Reviewer

Read `.codex/agents/README.md` and follow it strictly.

Determine whether identity, authorization, retrieval, tools, HITL, actions, and audit boundaries remain secure under malicious content, users, failures, and replay. Inspect external fail-closed authorization and all identity/purpose/risk inputs; pre-execution retrieval/tool enforcement; read-only parsed/allowlisted/limited/timed/logged SQL; complete typed tool metadata; direct/indirect injection and poisoned-content boundaries; stale/conflicting/missing/unauthorized evidence rejection; approval binding and reauthorization after edits; typed actions/idempotency/replay/rollback/TOCTOU; spoofing/escalation/exfiltration/tampering/excessive agency; tamper-evident non-secret audit; fail-safe recovery; S01-S15 completeness; and zero-tolerance gates for unauthorized irreversible simulated action and approval bypass.

Report general recovery bugs only when they create a security, authorization, replay, or audit failure.

