from __future__ import annotations

import re
from collections.abc import Iterable

from controlflow.v22.schemas import AuthenticatedContext, EvidenceDocument, RuntimeCase


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_-]+", text.casefold()))


class EvidenceRetriever:
    def __init__(
        self,
        documents: Iterable[EvidenceDocument],
        *,
        top_k: int = 3,
        lexical_weight: float = 1.0,
        control_metadata_weight: float = 0.5,
        case_metadata_weight: float = 0.0,
        minimum_score: float = 3.0,
    ) -> None:
        self.documents = tuple(documents)
        self.top_k = top_k
        self.lexical_weight = lexical_weight
        self.control_metadata_weight = control_metadata_weight
        self.case_metadata_weight = case_metadata_weight
        self.minimum_score = minimum_score

    def retrieve(
        self, case: RuntimeCase, context: AuthenticatedContext, *, temporal_filter: bool = True
    ) -> tuple[str, ...]:
        if context.requested_scope != context.business_unit:
            return ()
        query = _tokens(case.evidence_query)
        reference_tokens = {token for token in query if token.startswith("incident-")}
        ranked: list[tuple[float, str]] = []
        for document in self.documents:
            if document.business_unit != context.business_unit or document.classification > context.clearance:
                continue
            if temporal_filter and not (
                document.valid_from <= case.event_time
                and (document.valid_to is None or case.event_time < document.valid_to)
            ):
                continue
            if self.lexical_weight > 0 and not (reference_tokens & _tokens(document.text)):
                continue
            overlap = self.lexical_weight * len(query & _tokens(document.text))
            metadata = self.control_metadata_weight * int(document.control_family == case.control_family)
            # Case IDs are deliberately absent from the searchable corpus in V2.2.
            case_match = self.case_metadata_weight * int(document.case_id == case.case_id)
            ranked.append((float(overlap + metadata + case_match), document.document_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(document_id for score, document_id in ranked[: self.top_k] if score >= self.minimum_score)

    def sufficient(self, case: RuntimeCase, document_ids: tuple[str, ...]) -> bool:
        selected = {item.document_id: item for item in self.documents if item.document_id in document_ids}
        reference_tokens = {token for token in _tokens(case.evidence_query) if token.startswith("incident-")}
        corroborating = sum(bool(reference_tokens & _tokens(document.text)) for document in selected.values())
        return corroborating >= 2
