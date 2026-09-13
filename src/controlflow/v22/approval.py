from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from controlflow.core.state import canonical_json, sha256_file
from controlflow.v22.schemas import ApprovalPayload, PolicyResult, ProposedAction, SignedApproval


def proposed_action_hash(action: ProposedAction) -> str:
    return hashlib.sha256(canonical_json(action.model_dump(mode="json"))).hexdigest()


class ApprovalIssuer:
    """Logically separate reviewer service. PEP never receives this private key."""

    def __init__(self, private_key: Ed25519PrivateKey, *, key_id: str = "v22-dev-reviewer") -> None:
        self.__private_key = private_key
        self.key_id = key_id

    @classmethod
    def create_ephemeral(cls, key_directory: Path, public_key_path: Path) -> ApprovalIssuer:
        key_directory.mkdir(parents=True, exist_ok=True)
        private_path = key_directory / "approval_ed25519.private.pem"
        if private_path.exists():
            private = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
            if not isinstance(private, Ed25519PrivateKey):
                raise TypeError("approval private key is not Ed25519")
        else:
            private = Ed25519PrivateKey.generate()
            private_path.write_bytes(
                private.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            with contextlib.suppress(OSError):
                os.chmod(private_path, 0o600)
        public_key_path.parent.mkdir(parents=True, exist_ok=True)
        public_key_path.write_bytes(
            private.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return cls(private)

    def issue(
        self,
        *,
        action: ProposedAction,
        policy: PolicyResult,
        reviewer_id: str,
        approve: bool,
        now: datetime | None = None,
        lifetime: timedelta = timedelta(minutes=15),
    ) -> SignedApproval:
        issued_at = now or datetime.now(UTC)
        payload = ApprovalPayload(
            token_id=str(uuid.uuid4()),
            case_id=action.case_id,
            action_hash=proposed_action_hash(action),
            reviewer_id=reviewer_id,
            review_decision="APPROVE" if approve else "REJECT",
            policy_decision_id=policy.policy_decision_id,
            authorization_snapshot_hash=policy.authorization_context_hash,
            policy_version=policy.policy_version,
            action_registry_version=policy.action_registry_version,
            workflow_version=action.workflow_version,
            issued_at=issued_at,
            expires_at=issued_at + lifetime,
            nonce=uuid.uuid4().hex,
        )
        signature = self.__private_key.sign(canonical_json(payload.model_dump(mode="json")))
        return SignedApproval(payload=payload, signature_b64=base64.b64encode(signature).decode(), key_id=self.key_id)


class ApprovalVerifier:
    def __init__(self, public_key_path: Path, *, key_id: str = "v22-dev-reviewer") -> None:
        public = serialization.load_pem_public_key(public_key_path.read_bytes())
        if not isinstance(public, Ed25519PublicKey):
            raise TypeError("approval public key is not Ed25519")
        self.public_key = public
        self.key_id = key_id
        self.public_key_sha256 = sha256_file(public_key_path)

    def verify(self, approval: SignedApproval) -> bool:
        if approval.key_id != self.key_id:
            return False
        try:
            self.public_key.verify(
                base64.b64decode(approval.signature_b64, validate=True),
                canonical_json(approval.payload.model_dump(mode="json")),
            )
        except (InvalidSignature, ValueError):
            return False
        return True

    @staticmethod
    def signature_hash(approval: SignedApproval) -> str:
        try:
            value = base64.b64decode(approval.signature_b64, validate=True)
        except ValueError:
            value = approval.signature_b64.encode()
        return hashlib.sha256(value).hexdigest()
