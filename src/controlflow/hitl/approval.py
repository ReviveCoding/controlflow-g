from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from controlflow.audit.ledger import action_key
from controlflow.core.state import canonical_json


class InvalidApproval(PermissionError):
    pass


@dataclass(frozen=True)
class ApprovalAuthority:
    """Issues short-lived tokens bound to the exact action and policy context."""

    secret: bytes

    def issue(
        self,
        *,
        case_id: str,
        action_type: str,
        payload: dict[str, Any],
        workflow_version: str,
        reviewer_id: str,
        policy_version: str,
        evidence_hash: str,
        ttl_seconds: int = 900,
    ) -> str:
        body = {
            "action_key": action_key(case_id, action_type, payload, workflow_version),
            "reviewer_id": reviewer_id,
            "policy_version": policy_version,
            "evidence_hash": evidence_hash,
            "expires_at": (datetime.now(UTC) + timedelta(seconds=ttl_seconds)).isoformat(),
        }
        encoded = base64.urlsafe_b64encode(canonical_json(body)).decode()
        signature = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def verify(
        self, token: str, *, expected_action_key: str, policy_version: str, evidence_hash: str
    ) -> dict[str, Any]:
        try:
            encoded, signature = token.split(".", 1)
            expected = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise InvalidApproval("invalid approval signature")
            body = cast(dict[str, Any], json.loads(base64.urlsafe_b64decode(encoded.encode())))
            if (
                body["action_key"] != expected_action_key
                or body["policy_version"] != policy_version
                or body["evidence_hash"] != evidence_hash
            ):
                raise InvalidApproval("approval is not bound to this action context")
            if datetime.fromisoformat(body["expires_at"]) <= datetime.now(UTC):
                raise InvalidApproval("approval expired")
            return body
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, InvalidApproval):
                raise
            raise InvalidApproval("malformed approval token") from exc
