from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from controlflow.retrieval.core import GovernedBM25Retriever
from controlflow.schemas import IdentityContext, TemporalEvidence


def evidence(identifier: str, text: str, classification: int = 1, start: str = "2024-01-01") -> TemporalEvidence:
    return TemporalEvidence(
        evidence_id=identifier,
        source="test",
        text=text,
        classification=classification,
        business_valid_from=datetime.fromisoformat(start).replace(tzinfo=UTC),
        system_known_from=datetime.fromisoformat(start).replace(tzinfo=UTC),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_governed_bm25_never_indexes_unauthorized_evidence() -> None:
    corpus = [
        evidence("public", "access control account review"),
        evidence("secret", "access control account review privileged", 5),
    ]
    identity = IdentityContext(
        user_id="u",
        role="Control Analyst",
        business_unit="consumer",
        region="US",
        clearance=2,
        purpose="investigate",
        session_id="s",
    )
    retriever = GovernedBM25Retriever(
        corpus,
        identity=identity,
        event_time=datetime(2025, 1, 1, tzinfo=UTC),
        known_time=datetime(2025, 1, 1, tzinfo=UTC),
    )
    governed = retriever.search("privileged access control", 2)
    assert [item.evidence_id for item in retriever.corpus] == ["public"]
    assert [hit.evidence.evidence_id for hit in governed] == ["public"]
