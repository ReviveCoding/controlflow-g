from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from controlflow.retrieval.core import BM25Retriever, governed_filter
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


def test_bm25_and_governed_filter_remove_unauthorized() -> None:
    corpus = [
        evidence("public", "access control account review"),
        evidence("secret", "access control account review privileged", 5),
    ]
    hits = BM25Retriever(corpus).search("privileged access control", 2)
    identity = IdentityContext(
        user_id="u",
        role="Control Analyst",
        business_unit="consumer",
        region="US",
        clearance=2,
        purpose="investigate",
        session_id="s",
    )
    governed = governed_filter(
        hits,
        identity=identity,
        event_time=datetime(2025, 1, 1, tzinfo=UTC),
        known_time=datetime(2025, 1, 1, tzinfo=UTC),
        k=2,
    )
    assert [hit.evidence.evidence_id for hit in governed] == ["public"]
