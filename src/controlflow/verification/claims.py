from __future__ import annotations

import hashlib

from controlflow.schemas import AgentState, ClaimVerification, VerificationResult


def verify_claims(state: AgentState, claims: dict[str, tuple[str, tuple[str, ...]]]) -> VerificationResult:
    evidence = {item.evidence_id: item for item in state.retrieved_evidence}
    verified: list[ClaimVerification] = []
    for claim_id, (text, evidence_ids) in claims.items():
        items = [evidence[item] for item in evidence_ids if item in evidence]
        complete = len(items) == len(evidence_ids) and bool(items)
        temporal = complete and all(item.valid_at(state.event_time, state.system_time) for item in items)
        authorized = complete and all(
            state.identity_context.clearance >= item.classification
            and (not item.authorized_roles or state.identity_context.role in item.authorized_roles)
            for item in items
        )
        hashes_valid = complete and all(
            hashlib.sha256(item.text.encode()).hexdigest() == item.content_sha256 for item in items
        )
        normalized = [item.text.casefold().strip() for item in items]
        conflict = any(text.startswith("[contradicts]") for text in normalized)
        # Claim support is a governed ingestion relation, never a magic token
        # embedded in untrusted text. All required evidence must come from the
        # trusted corpus and be explicitly related to this claim.
        support = complete and all(item.trusted_ingestion and claim_id in item.claim_relations for item in items)
        status = (
            "VERIFIED"
            if complete and temporal and authorized and hashes_valid and support and not conflict
            else "REJECTED"
        )
        confidence = 1.0 if status == "VERIFIED" else 0.0
        verified.append(
            ClaimVerification(
                claim_id=claim_id,
                claim_text=text,
                supporting_evidence_ids=evidence_ids,
                source_provenance=tuple(item.source for item in items),
                temporal_validity=temporal,
                authorization_validity=authorized,
                conflict_status=conflict,
                verification_status=status,
                confidence=confidence,
            )
        )
    all_verified = bool(verified) and all(item.verification_status == "VERIFIED" for item in verified)
    return VerificationResult(claims=tuple(verified), all_verified=all_verified, sufficient_evidence=all_verified)
