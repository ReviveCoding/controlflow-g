from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from controlflow.v22.schemas import PolicyDocument


def load_policy_corpus(path: Path) -> tuple[PolicyDocument, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(PolicyDocument.model_validate(item) for item in payload["policies"])


class CandidateTemporalRetriever:
    """Candidate-side bitemporal retrieval; evaluator truth is authored in DGP fixtures."""

    def __init__(self, documents: tuple[PolicyDocument, ...]) -> None:
        self.documents = documents

    def select(self, event_time: datetime, system_time: datetime, *, temporal_filter: bool = True) -> str | None:
        if not temporal_filter:
            return max(self.documents, key=lambda item: item.system_known_from).policy_id
        visible = [
            item
            for item in self.documents
            if item.business_valid_from <= event_time
            and (item.business_valid_to is None or event_time < item.business_valid_to)
            and item.system_known_from <= system_time
            and (item.system_known_to is None or system_time < item.system_known_to)
        ]
        if not visible:
            return None
        best_rank = max(item.correction_rank for item in visible)
        ranked = [item for item in visible if item.correction_rank == best_rank]
        if len(ranked) > 1:
            return None
        return ranked[0].policy_id
