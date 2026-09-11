from __future__ import annotations

import os
import secrets
from pathlib import Path

from filelock import FileLock

from controlflow.core.state import ProjectPaths
from controlflow.hitl.approval import ApprovalAuthority


def configured_recovery_authority() -> ApprovalAuthority:
    """Load the runtime-pinned verifier for exceptional audit-anchor recovery.

    This is a local simulation trust root. Production deployments must replace
    it with an independently administered KMS/HSM-backed approval verifier.
    """
    trust = Path(
        os.environ.get(
            "CONTROLFLOW_AUDIT_TRUST_DIR",
            str(ProjectPaths.discover().state / "audit_trust"),
        )
    )
    trust.mkdir(parents=True, exist_ok=True)
    key_path = trust / "recovery-approval.key"
    with FileLock(str(trust / ".recovery-approval-key.lock")):
        if not key_path.exists():
            with key_path.open("xb") as handle:
                handle.write(secrets.token_bytes(32))
                handle.flush()
                os.fsync(handle.fileno())
            key_path.chmod(0o400)
        secret = key_path.read_bytes()
    if len(secret) != 32:
        raise RuntimeError("invalid audit recovery authority key")
    return ApprovalAuthority(
        secret,
        reviewer_entitlements={"audit-recovery-reviewer": ("Risk Manager", "enterprise")},
    )
