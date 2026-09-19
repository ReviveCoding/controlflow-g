"""Pre-execution proof of model-independent authoritative PDP opportunities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.v22.policy import ActionRegistry, AuthorizationState, PolicyDecisionPoint
from controlflow.v22.schemas import AuthenticatedContext, PolicyDecision, PolicyInput, Severity


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9_-]+", str(value).casefold()))


def _evidence_proof(case: Any, evidence: pd.DataFrame, config: dict[str, Any]) -> bool:
    references = {token for token in _tokens(case.evidence_query) if token.startswith("incident-")}
    if len(references) != 1:
        return False
    matching = evidence[evidence.text.astype(str).map(lambda value: bool(references & _tokens(value)))]
    if len(matching) != 2 or int(config["top_k"]) < 2:
        return False
    query = _tokens(case.evidence_query)
    for item in matching.itertuples(index=False):
        if (
            item.business_unit != case.business_unit
            or int(item.classification) > int(case.clearance)
            or pd.Timestamp(item.valid_from) > pd.Timestamp(case.event_time)
            or (item.valid_to is not None and pd.Timestamp(case.event_time) >= pd.Timestamp(item.valid_to))
        ):
            return False
        score = float(config["lexical_weight"]) * len(query & _tokens(item.text))
        score += float(config["control_metadata_weight"]) * int(item.control_family == case.control_family)
        if score < float(config["minimum_score"]):
            return False
    return True


def certify_opportunities(
    root: Path, runtime: pd.DataFrame, evidence: pd.DataFrame, auth: pd.DataFrame
) -> dict[str, Any]:
    config = yaml.safe_load((root / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))["retrieval"]
    registry = ActionRegistry(root / "configs/v22/action_registry.yaml")
    state = AuthorizationState(
        {
            str(item.identity): {
                "active": bool(item.active),
                "role": str(item.role),
                "business_unit": str(item.business_unit),
                "region": str(item.region),
                "clearance": int(item.clearance),
                "authorization_version": str(item.authorization_version),
                "case_risk_version": str(item.case_risk_version),
            }
            for item in auth.itertuples(index=False)
        }
    )
    pdp = PolicyDecisionPoint(policy_path=root / "configs/v22/policy.yaml", registry=registry, authorization=state)
    review: list[str] = []
    deny: list[str] = []
    actions = tuple(registry.actions)
    for case in runtime.itertuples(index=False):
        context = AuthenticatedContext(
            identity=str(case.authenticated_identity),
            role=str(case.role),
            business_unit=str(case.business_unit),
            region=str(case.region),
            clearance=int(case.clearance),
            purpose=str(case.purpose),
            requested_scope=str(case.requested_scope),
            data_classification=int(case.data_classification),
        )
        scope_denied = case.requested_scope != case.business_unit or int(case.clearance) < int(case.data_classification)
        evidence_proven = (
            not scope_denied
            and "conflicting" in str(case.narrative).casefold()
            and _evidence_proof(case, evidence, config)
        )
        if not scope_denied and not evidence_proven:
            continue
        decisions = {
            pdp.decide(
                PolicyInput(
                    case_id=str(case.case_id),
                    context=context,
                    severity=Severity.LOW,
                    critical_probability=0.0,
                    evidence_sufficient=evidence_proven,
                    conflict_state=evidence_proven,
                    novelty=0.0,
                    uncertainty=0.0,
                    proposed_action=action,
                )
            ).decision
            for action in actions
        }
        if scope_denied and decisions == {PolicyDecision.DENY}:
            deny.append(str(case.case_id))
        if evidence_proven and decisions == {PolicyDecision.REQUIRE_REVIEW}:
            review.append(str(case.case_id))
    return {
        "schema_version": 1,
        "review_certified_count": len(review),
        "deny_certified_count": len(deny),
        "review_case_ids": review,
        "deny_case_ids": deny,
        "proof": (
            "all frozen candidate actions evaluated by authoritative PDP; review evidence sufficiency "
            "proved from two uniquely referenced accessible corpus records"
        ),
    }
