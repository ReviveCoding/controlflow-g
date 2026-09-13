from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimAssessment:
    claim: str
    status: str
    evidence_ids: tuple[str, ...]


def decompose_claims(rationale: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r"(?<=[.!?])\s+|;\s*", rationale) if part.strip())


def assess_claims(rationale: str, evidence: dict[str, str]) -> tuple[ClaimAssessment, ...]:
    """Conservative lexical entailment: unsupported claims never pass by ID presence alone."""
    assessments: list[ClaimAssessment] = []
    for claim in decompose_claims(rationale):
        claim_tokens = set(re.findall(r"[a-z0-9]+", claim.casefold())) - {"the", "a", "an", "is", "and", "because"}
        mapped = tuple(identifier for identifier in evidence if identifier.casefold() in claim.casefold())
        evidence_tokens = set(re.findall(r"[a-z0-9]+", " ".join(evidence.values()).casefold()))
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        contradiction = any(phrase in claim.casefold() for phrase in ("not supported", "contradicts evidence"))
        status = "CONTRADICTED" if contradiction else "SUPPORTED" if mapped and overlap >= 0.5 else "UNSUPPORTED"
        assessments.append(ClaimAssessment(claim=claim, status=status, evidence_ids=mapped))
    return tuple(assessments)


def deterministic_rationale(severity: str, policy_decision: str, evidence_ids: tuple[str, ...]) -> str:
    cited = ", ".join(evidence_ids) if evidence_ids else "the available runtime evidence"
    return f"{policy_decision.replace('_', ' ').title()} because {cited} supports a {severity} control exception."
