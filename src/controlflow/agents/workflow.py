from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from controlflow.audit.ledger import ActionLedger
from controlflow.authorization.policy import LocalPolicyBackend, ToolPolicyInput
from controlflow.core.state import atomic_write_json, canonical_json
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.modeling.supervised import LABELS, _preprocessor, _xy
from controlflow.retrieval.core import (
    BM25Retriever,
    NeuralHybridRetriever,
    authorized_evidence_partition,
)
from controlflow.schemas import (
    AgentState,
    AnomalySignal,
    Disposition,
    FinalAgentOutput,
    IdentityContext,
    ProposedAction,
    RiskPrediction,
    Severity,
    TemporalEvidence,
)
from controlflow.tools.registry import ToolRegistry, ToolSpec
from controlflow.verification.claims import verify_claims

CONTROL = re.compile(r"\b[A-Z]{2}-\d+(?:\.\d+)?\b")
REGULATION = re.compile(r"\b12CFR-\d+\b")
INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?previous|disregard\s+(?:all\s+)?earlier|override\s+(?:the\s+)?system|"
    r"unrestricted\s+tool|role\s*=\s*administrator|i\s*g\s*n\s*o\s*r\s*e",
    re.I,
)


class EvidenceSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


class EvidenceSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str]


class ActionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    status: str


class ActionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executed: bool
    action_id: int


class CaseLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str


class StructuredResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: dict[str, float | int | str | list[str]]


class EvidenceBundleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str]
    bundle_sha256: str


class RiskToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    probabilities: list[float]
    severity: str
    confidence: float


class AnomalyToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float
    threshold: float
    is_anomaly: bool


@dataclass(frozen=True)
class WorkflowConfig:
    retrieval: bool = True
    temporal_retrieval: bool = True
    reranker: bool = True
    ml_risk: bool = True
    calibration: bool = True
    anomaly: bool = True
    verifier: bool = True
    authorization: bool = True
    hitl: bool = True
    structured_output: bool = True
    bounded_tools: bool = True
    tools_enabled: bool = True
    point_in_time_features: bool = True


class RiskService:
    def __init__(self, training: pd.DataFrame) -> None:
        x, y = _xy(training)
        from sklearn.linear_model import LogisticRegression

        self.model = Pipeline(
            [
                ("features", _preprocessor()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17)),
            ]
        )
        self.model.fit(x, y)
        numeric = training[["amount", "repeat_count", "historical_failures", "data_sensitivity"]]
        self.anomaly = IsolationForest(contamination=0.05, random_state=17, n_jobs=1).fit(numeric)
        self.anomaly_threshold = float(pd.Series(-self.anomaly.score_samples(numeric)).quantile(0.95))
        self.calibrator: object | None = None
        self.review_threshold = 0.8

    def predict_details(self, row: pd.Series, *, calibrated: bool = True) -> tuple[np.ndarray, float]:
        probability = self.model.predict_proba(pd.DataFrame([row]))[0]
        if calibrated and self.calibrator is not None:
            probability = self.calibrator.predict(probability.reshape(1, -1))[0]  # type: ignore[attr-defined]
        numeric = pd.DataFrame([row])[["amount", "repeat_count", "historical_failures", "data_sensitivity"]]
        return probability, float(-self.anomaly.score_samples(numeric)[0])

    def predict(self, row: pd.Series, *, calibrated: bool = True) -> tuple[str, float, float]:
        probability, anomaly = self.predict_details(row, calibrated=calibrated)
        return (
            LABELS[int(probability.argmax())],
            float(probability.max()),
            anomaly,
        )


@dataclass
class WorkflowTrace:
    case_id: str
    predicted_severity: str
    predicted_disposition: str
    risk_confidence: float
    anomaly_score: float
    retrieved_ids: list[str]
    tool_calls: list[str]
    evidence_verified: bool
    temporal_correct: bool
    feature_event_timestamp: str
    feature_system_known_at: str
    authorization_outcome: str
    human_review_requested: bool
    structured_output_valid: bool
    action_executed: bool
    action_id: int | None
    injection_detected: bool
    latency_seconds: float
    retry_count: int = 0
    tool_argument_errors: int = 0


class GovernedWorkflow:
    def __init__(
        self,
        training: pd.DataFrame,
        controls: pd.DataFrame,
        ledger: ActionLedger,
        authority: ApprovalAuthority,
        risk_service: RiskService | None = None,
        state_dir: Path | None = None,
        regulations: pd.DataFrame | None = None,
        require_cuda_retrieval: bool = False,
    ) -> None:
        self.risk = risk_service or RiskService(training)
        self.controls = controls
        self.ledger = ledger
        self.policy = LocalPolicyBackend()
        self.authority = authority
        self.state_dir = state_dir
        self.require_cuda_retrieval = require_cuda_retrieval
        self.training = training.set_index("case_id", drop=False)
        self._retriever_cache: dict[tuple[str, int, int, bool, bool], BM25Retriever | NeuralHybridRetriever] = {}
        origin = datetime(2020, 1, 1, tzinfo=UTC)
        self.evidence_corpus: list[TemporalEvidence] = []
        for control in controls.itertuples():
            text = f"Control {control.control_id}: {control.title}. {control.description}"
            self.evidence_corpus.append(
                TemporalEvidence(
                    evidence_id=str(control.control_id),
                    source="NIST SP 800-53",
                    text=text,
                    classification=0,
                    business_valid_from=origin,
                    system_known_from=origin,
                    authorized_roles=frozenset({"Control Analyst", "Risk Manager", "Compliance Reviewer"}),
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    trusted_ingestion=True,
                    claim_relations=frozenset({"applicability"}),
                )
            )
        if regulations is not None:
            for regulation in regulations.head(2_000).itertuples():
                text = str(regulation.text)[:1600]
                timestamp = pd.Timestamp(regulation.business_valid_from)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                regulation_valid_from = timestamp.to_pydatetime()
                self.evidence_corpus.append(
                    TemporalEvidence(
                        evidence_id=str(regulation.regulation_id),
                        source="official eCFR annual edition",
                        text=text,
                        classification=0,
                        business_valid_from=regulation_valid_from,
                        business_valid_to=regulation_valid_from.replace(year=regulation_valid_from.year + 1),
                        system_known_from=regulation_valid_from,
                        authorized_roles=frozenset({"Control Analyst", "Risk Manager", "Compliance Reviewer"}),
                        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        trusted_ingestion=True,
                        claim_relations=frozenset({"regulatory_context"}),
                    )
                )

    @staticmethod
    def identity(row: pd.Series) -> IdentityContext:
        return IdentityContext(
            user_id="analyst-1",
            role="Control Analyst",
            business_unit=str(row.business_unit),
            region="US",
            clearance=2,
            purpose="control exception investigation",
            session_id="validation-session",
        )

    def _evidence(
        self, row: pd.Series, config: WorkflowConfig, identity: IdentityContext
    ) -> tuple[list[TemporalEvidence], bool]:
        missing_evidence = str(row.get("evidence_status", "AVAILABLE")) == "MISSING"
        if not config.retrieval or missing_evidence:
            # Absence is a correct temporal state for an explicit missing-
            # evidence case; the verifier will force insufficient evidence.
            return [], missing_evidence
        origin = datetime(2020, 1, 1, tzinfo=UTC)
        event_time = pd.Timestamp(row.event_timestamp).to_pydatetime()
        known_time = event_time
        authorized = authorized_evidence_partition(
            self.evidence_corpus,
            identity=identity,
            event_time=event_time if config.temporal_retrieval else datetime(2025, 1, 1, tzinfo=UTC),
            known_time=known_time if config.temporal_retrieval else datetime(2025, 1, 1, tzinfo=UTC),
        )
        cache_key = (identity.role, identity.clearance, event_time.year, config.temporal_retrieval, config.reranker)
        retriever = self._retriever_cache.get(cache_key)
        if retriever is None:
            retriever = (
                NeuralHybridRetriever(authorized, require_cuda=self.require_cuda_retrieval)
                if config.reranker
                else BM25Retriever(authorized)
            )
            self._retriever_cache[cache_key] = retriever
        hits = retriever.search(str(row.narrative), 8)
        selected = [hit.evidence for hit in hits]
        control_match = CONTROL.search(str(row.narrative))
        if control_match and control_match.group(0) not in {item.evidence_id for item in selected}:
            exact_control = next(
                (
                    item
                    for item in authorized
                    if item.source == "NIST SP 800-53" and item.evidence_id == control_match.group(0)
                ),
                None,
            )
            if exact_control is not None:
                selected.insert(0, exact_control)
        regulation_match = REGULATION.search(str(row.narrative))
        if regulation_match:
            regulation = regulation_match.group(0)
            expected_version = "policy-v1" if event_time.year < 2024 else "policy-v2"
            chosen_version = expected_version if config.temporal_retrieval else "policy-v2"
            valid_from = datetime(2020 if chosen_version == "policy-v1" else 2024, 1, 1, tzinfo=UTC)
            valid_to = datetime(2024, 1, 1, tzinfo=UTC) if chosen_version == "policy-v1" else None
            text = f"Regulation {regulation} {chosen_version} applies"
            selected.append(
                TemporalEvidence(
                    evidence_id=f"{regulation}:{chosen_version}",
                    source="governed bitemporal policy registry",
                    text=text,
                    classification=0,
                    business_valid_from=valid_from,
                    business_valid_to=valid_to,
                    system_known_from=valid_from,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    trusted_ingestion=True,
                    claim_relations=frozenset({"applicability"}),
                )
            )
        if str(row.get("evidence_status", "AVAILABLE")) == "CONFLICT" and selected:
            control_id = CONTROL.search(str(row.narrative)).group(0)  # type: ignore[union-attr]
            text = f"[CONTRADICTS] control {control_id} does not apply"
            selected.append(
                TemporalEvidence(
                    evidence_id=f"{control_id}:CONFLICT",
                    source="synthetic conflict fixture",
                    text=text,
                    classification=0,
                    business_valid_from=origin,
                    system_known_from=origin,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    trusted_ingestion=True,
                    claim_relations=frozenset({"applicability"}),
                )
            )
        temporal = all(
            item.valid_at(
                pd.Timestamp(row.event_timestamp).to_pydatetime(), pd.Timestamp(row.event_timestamp).to_pydatetime()
            )
            for item in selected
        )
        return selected, temporal

    def context_for_llm(self, row: pd.Series, config: WorkflowConfig) -> str:
        evidence, _ = self._evidence(row, config, self.identity(row))
        controls = "\n".join(item.text[:300] for item in evidence if item.source == "NIST SP 800-53")
        regulations = "\n".join(item.text[:300] for item in evidence if item.source != "NIST SP 800-53")
        risk_observation = ""
        anomaly_observation = ""
        if config.ml_risk:
            feature_row = row.copy()
            feature_row["historical_failures"] = (
                row["pit_historical_failures"] if config.point_in_time_features else row["future_failures"]
            )
            severity, confidence, anomaly = self.risk.predict(feature_row, calibrated=config.calibration)
            risk_observation = f"severity={severity}; confidence={confidence:.6f}"
            anomaly_observation = f"anomaly_score={anomaly:.6f}"
        return json.dumps(
            {
                "search_controls": controls,
                "search_regulations": regulations,
                "compute_risk": risk_observation,
                "compute_anomaly": anomaly_observation,
                "propose_case_update": "write action requires the workflow authorization boundary",
            },
            sort_keys=True,
        )

    def execute(
        self,
        row: pd.Series,
        config: WorkflowConfig,
        llm_prediction: tuple[str, str, bool],
        llm_tool_requests: list[str] | None = None,
        llm_tool_arguments: list[dict[str, object]] | None = None,
    ) -> WorkflowTrace:
        started = time.perf_counter()
        identity = self.identity(row)
        tool_calls: list[str] = []
        requested = set(llm_tool_requests) if llm_tool_requests is not None else None
        requested_arguments = {
            name: arguments for name, arguments in zip(llm_tool_requests or [], llm_tool_arguments or [], strict=False)
        }
        tool_argument_errors = 0
        risk_requested = requested is None or bool(requested & {"compute_risk", "compute_anomaly"})
        retrieval_requested = requested is None or bool(
            requested
            & {
                "search_controls",
                "search_regulations",
                "search_cases",
                "get_policy_at_time",
                "query_case_data",
                "query_transactions",
                "generate_evidence_bundle",
            }
        )
        action_requested = requested is None or "propose_case_update" in requested
        llm_severity, llm_disposition, llm_valid = llm_prediction
        risk_probabilities: np.ndarray | None = None
        service_registry = ToolRegistry(self.policy)
        feature_row = row.copy()
        if config.point_in_time_features:
            feature_row["historical_failures"] = row["pit_historical_failures"]
        else:
            feature_row["historical_failures"] = row["future_failures"]

        def compute_risk(_request: CaseLookupInput) -> RiskToolOutput:
            probabilities, _anomaly_value = self.risk.predict_details(feature_row, calibrated=config.calibration)
            return RiskToolOutput(
                probabilities=probabilities.tolist(),
                severity=LABELS[int(probabilities.argmax())],
                confidence=float(probabilities.max()),
            )

        def compute_anomaly(_request: CaseLookupInput) -> AnomalyToolOutput:
            _probability_value, score = self.risk.predict_details(feature_row, calibrated=config.calibration)
            return AnomalyToolOutput(
                score=score,
                threshold=self.risk.anomaly_threshold,
                is_anomaly=score > self.risk.anomaly_threshold,
            )

        for risk_tool_name, risk_output_model, risk_implementation in (
            ("compute_risk", RiskToolOutput, compute_risk),
            ("compute_anomaly", AnomalyToolOutput, compute_anomaly),
        ):
            service_registry.register(
                ToolSpec(
                    name=risk_tool_name,
                    input_model=CaseLookupInput,
                    output_model=risk_output_model,
                    risk_tier=0,
                    read_only=True,
                    allowed_roles=frozenset({"Control Analyst"}),
                    allowed_data_scopes=frozenset({identity.business_unit}),
                    human_review_required=False,
                    timeout_seconds=2.0,
                    max_retries=1,
                    implementation=risk_implementation,
                )
            )
        if config.ml_risk and risk_requested:
            try:
                risk_result = service_registry.invoke(
                    "compute_risk",
                    requested_arguments.get("compute_risk", {"case_id": str(row.case_id)}),
                    identity=identity,
                    scope=identity.business_unit,
                    data_classification=int(row.get("data_sensitivity", 0)),
                    severity=Severity.LOW,
                )
            except (ValidationError, PermissionError):
                risk_result = None
                tool_argument_errors += 1
            if risk_result is None:
                severity, confidence, anomaly_score = llm_severity, 0.5, 0.0
                risk_requested = False
            else:
                risk_probabilities = np.asarray(risk_result.probabilities)  # type: ignore[attr-defined]
                severity = str(risk_result.severity)  # type: ignore[attr-defined]
                confidence = float(risk_result.confidence)  # type: ignore[attr-defined]
                try:
                    anomaly_result = service_registry.invoke(
                        "compute_anomaly",
                        requested_arguments.get("compute_anomaly", {"case_id": str(row.case_id)}),
                        identity=identity,
                        scope=identity.business_unit,
                        data_classification=int(row.get("data_sensitivity", 0)),
                        severity=Severity(severity),
                    )
                    anomaly_score = float(anomaly_result.score)  # type: ignore[attr-defined]
                    if config.anomaly and bool(anomaly_result.is_anomaly):  # type: ignore[attr-defined]
                        severity = LABELS[min(3, LABELS.index(severity) + 1)]
                except (ValidationError, PermissionError):
                    anomaly_score = 0.0
                    tool_argument_errors += 1
                tool_calls.append("compute_risk")
                if config.anomaly and tool_argument_errors == 0:
                    tool_calls.append("compute_anomaly")
        else:
            severity = llm_severity
            confidence, anomaly_score = 0.5, 0.0
        evidence, temporal = (
            self._evidence(row, config, identity) if retrieval_requested and requested is None else ([], True)
        )
        feature_temporal = pd.Timestamp(row.get("feature_event_timestamp", row.event_timestamp)) <= pd.Timestamp(
            row.event_timestamp
        ) and pd.Timestamp(row.get("feature_system_known_at", row.event_timestamp)) <= pd.Timestamp(row.event_timestamp)
        temporal = temporal and feature_temporal and config.point_in_time_features
        if config.retrieval and retrieval_requested:
            registry = ToolRegistry(self.policy)

            def evidence_search(selected_tool: str) -> Callable[[EvidenceSearchInput], EvidenceSearchOutput]:
                def search(_request: EvidenceSearchInput) -> EvidenceSearchOutput:
                    return EvidenceSearchOutput(
                        evidence_ids=[
                            item.evidence_id
                            for item in self._evidence(row, config, identity)[0]
                            if (
                                (selected_tool == "search_controls" and item.source == "NIST SP 800-53")
                                or (selected_tool == "search_regulations" and item.source != "NIST SP 800-53")
                            )
                        ]
                    )

                return search

            for tool_name in ("search_controls", "search_regulations"):
                if requested is not None and tool_name not in requested:
                    continue
                registry.register(
                    ToolSpec(
                        name=tool_name,
                        input_model=EvidenceSearchInput,
                        output_model=EvidenceSearchOutput,
                        risk_tier=0,
                        read_only=True,
                        allowed_roles=frozenset({"Control Analyst"}),
                        allowed_data_scopes=frozenset({identity.business_unit}),
                        human_review_required=False,
                        timeout_seconds=2.0,
                        max_retries=1,
                        implementation=evidence_search(tool_name),
                    )
                )
                try:
                    raw_arguments = requested_arguments.get(tool_name, {"query": str(row.narrative)})
                    result = (
                        registry.invoke(
                            tool_name,
                            raw_arguments,
                            identity=identity,
                            scope=identity.business_unit,
                            data_classification=int(row.get("data_sensitivity", 0)),
                            severity=Severity(severity),
                        )
                        if config.authorization
                        else evidence_search(tool_name)(EvidenceSearchInput.model_validate(raw_arguments))
                    )
                    selected_ids = set(result.evidence_ids)  # type: ignore[attr-defined]
                    selected, selected_temporal = self._evidence(row, config, identity)
                    evidence.extend(item for item in selected if item.evidence_id in selected_ids)
                    temporal = temporal and selected_temporal
                    tool_calls.append(tool_name)
                except (ValidationError, PermissionError):
                    tool_argument_errors += 1
            auxiliary_tools: tuple[tuple[str, type[BaseModel], Callable[[CaseLookupInput], BaseModel]], ...] = (
                (
                    "search_cases",
                    StructuredResult,
                    lambda _: StructuredResult(
                        values={
                            "matching_case_ids": self.training.loc[
                                self.training["business_unit"].eq(identity.business_unit)
                            ]
                            .head(5)["case_id"]
                            .astype(str)
                            .tolist()
                        }
                    ),
                ),
                (
                    "get_policy_at_time",
                    EvidenceSearchOutput,
                    lambda _: EvidenceSearchOutput(
                        evidence_ids=[
                            item.evidence_id
                            for item in evidence
                            if item.source == "governed bitemporal policy registry"
                        ]
                    ),
                ),
                (
                    "query_case_data",
                    StructuredResult,
                    lambda _: StructuredResult(
                        values={
                            "case_id": str(row.case_id),
                            "business_unit": str(row.business_unit),
                            "event_timestamp": str(row.event_timestamp),
                        }
                    ),
                ),
                (
                    "query_transactions",
                    StructuredResult,
                    lambda _: StructuredResult(values={"case_id": str(row.case_id), "amount": float(row.amount)}),
                ),
                (
                    "generate_evidence_bundle",
                    EvidenceBundleOutput,
                    lambda _: EvidenceBundleOutput(
                        evidence_ids=[item.evidence_id for item in evidence],
                        bundle_sha256=hashlib.sha256(
                            canonical_json(sorted(item.evidence_id for item in evidence))
                        ).hexdigest(),
                    ),
                ),
            )
            for tool_name, output_model, implementation in auxiliary_tools:
                registry.register(
                    ToolSpec(
                        name=tool_name,
                        input_model=CaseLookupInput,
                        output_model=output_model,
                        risk_tier=0,
                        read_only=True,
                        allowed_roles=frozenset({"Control Analyst"}),
                        allowed_data_scopes=frozenset({identity.business_unit}),
                        human_review_required=False,
                        timeout_seconds=2.0,
                        max_retries=1,
                        implementation=implementation,
                    )
                )
                if requested is None or tool_name in requested:
                    try:
                        registry.invoke(
                            tool_name,
                            requested_arguments.get(tool_name, {"case_id": str(row.case_id)}),
                            identity=identity,
                            scope=identity.business_unit,
                            data_classification=int(row.get("data_sensitivity", 0)),
                            severity=Severity(severity),
                        )
                        tool_calls.append(tool_name)
                    except (ValidationError, PermissionError):
                        tool_argument_errors += 1
        state = AgentState(
            case_id=str(row.case_id),
            workflow_version="agent-graph-v3",
            event_time=pd.Timestamp(row.event_timestamp).to_pydatetime(),
            system_time=pd.Timestamp(row.event_timestamp).to_pydatetime(),
            identity_context=identity,
            purpose=identity.purpose,
            case_context={"business_unit": row.business_unit},
            retrieved_evidence=evidence,
        )
        required_ids = []
        control_match = CONTROL.search(str(row.narrative))
        regulation_match = REGULATION.search(str(row.narrative))
        if control_match:
            required_ids.append(control_match.group(0))
        expected_policy = "policy-v1" if state.event_time.year < 2024 else "policy-v2"
        if regulation_match:
            required_ids.append(f"{regulation_match.group(0)}:{expected_policy}")
        if str(row.get("evidence_status", "AVAILABLE")) == "CONFLICT" and control_match:
            required_ids.append(f"{control_match.group(0)}:CONFLICT")
        claims = {
            "applicability": (
                "control "
                f"{control_match.group(0) if control_match else 'unknown'} regulation "
                f"{regulation_match.group(0) if regulation_match else 'unknown'} applies",
                tuple(required_ids),
            )
        }
        verification = verify_claims(state, claims) if config.verifier else None
        evidence_verified = verification.all_verified if verification else bool(evidence)
        injection_detected = bool(INJECTION.search(str(row.narrative))) or any(
            INJECTION.search(item.text) is not None for item in evidence
        )
        requested_scope = (
            "restricted" if injection_detected else str(row.get("requested_scope", identity.business_unit))
        )
        request = ToolPolicyInput(
            tool_name="propose_case_update",
            risk_tier=2 if severity in {"HIGH", "CRITICAL"} else 1,
            read_only=False,
            allowed_roles=frozenset({"Control Analyst"}),
            allowed_scopes=frozenset({identity.business_unit}),
            requires_review=severity in {"HIGH", "CRITICAL"},
            data_classification=int(row.get("data_sensitivity", 0)),
            requested_scope=requested_scope,
            case_severity=Severity(severity),
            arguments={"case_id": str(row.case_id)},
        )
        authorization = self.policy.authorize(identity, request) if config.authorization else None
        authorization_outcome = authorization.outcome.value if authorization else "ALLOW"
        human_review = authorization_outcome == "REQUIRE_REVIEW" and config.hitl
        if injection_detected or authorization_outcome == "DENY":
            disposition = "DENY"
        elif config.verifier and not evidence_verified:
            disposition = "INSUFFICIENT_EVIDENCE"
        elif human_review or (config.ml_risk and config.calibration and confidence < self.risk.review_threshold):
            disposition = "REVIEW_REQUIRED"
        elif config.ml_risk:
            disposition = "REVIEW_REQUIRED" if severity in {"HIGH", "CRITICAL"} and config.hitl else "AUTO"
        else:
            disposition = llm_disposition
        if config.structured_output:
            try:
                FinalAgentOutput(
                    case_id=str(row.case_id),
                    severity=Severity(severity),
                    confidence=confidence,
                    controls=(control_match.group(0),) if control_match else (),
                    regulations=(regulation_match.group(0),) if regulation_match else (),
                    evidence=tuple(item.evidence_id for item in evidence),
                    root_cause_hypotheses=("control execution variance",),
                    recommended_actions=(
                        ProposedAction(
                            action_type="propose_case_update",
                            payload={"status": "investigated"},
                            risk_tier=2 if severity in {"HIGH", "CRITICAL"} else 1,
                            rollback_available=True,
                        ),
                    ),
                    automation_decision=Disposition(disposition),
                    human_review_required=disposition == "REVIEW_REQUIRED",
                )
                structured_valid = True
            except (ValidationError, ValueError):
                structured_valid = False
        else:
            structured_valid = llm_valid
        action_executed = False
        action_id = None
        if disposition == "REVIEW_REQUIRED" and config.tools_enabled and config.hitl and action_requested:
            pending = self.ledger.request_review(
                case_id=str(row.case_id),
                action_type="propose_case_update",
                payload={"status": "investigated"},
                workflow_version="agent-graph-v3",
            )
            action_id = pending.action_id
        if (
            disposition == "AUTO"
            and config.tools_enabled
            and action_requested
            and ((config.bounded_tools and authorization_outcome == "ALLOW") or not config.bounded_tools)
        ):
            payload = {"status": "investigated"}
            evidence_hash = hashlib.sha256(canonical_json(sorted(item.evidence_id for item in evidence))).hexdigest()
            # Reauthorize at the side-effect boundary against the same protected
            # identity, arguments, classification, and risk tier.
            final_authorization = self.policy.authorize(identity, request) if config.authorization else None
            final_outcome = final_authorization.outcome.value if final_authorization else "ALLOW"
            if config.bounded_tools and final_outcome != "ALLOW":
                raise PermissionError(f"write requires ALLOW, received {final_outcome}")
            token = self.authority.issue_system(
                case_id=str(row.case_id),
                action_type="propose_case_update",
                payload=payload,
                workflow_version="agent-graph-v3",
                policy_version="local-policy-v1",
                evidence_hash=evidence_hash,
                risk_tier=1,
                authorization_outcome=final_outcome,
            )

            def execute_action(_: ActionInput) -> ActionOutput:
                receipt = self.ledger.execute_simulated(
                    case_id=str(row.case_id),
                    action_type="propose_case_update",
                    payload=payload,
                    workflow_version="agent-graph-v3",
                    authorization_token=token,
                    approval_authority=self.authority,
                    evidence_hash=evidence_hash,
                )
                return ActionOutput(executed=receipt.executed, action_id=receipt.action_id)

            if config.authorization and config.bounded_tools:
                action_registry = ToolRegistry(self.policy)
                action_registry.register(
                    ToolSpec(
                        name="propose_case_update",
                        input_model=ActionInput,
                        output_model=ActionOutput,
                        risk_tier=1,
                        read_only=False,
                        allowed_roles=frozenset({"Control Analyst"}),
                        allowed_data_scopes=frozenset({identity.business_unit}),
                        human_review_required=False,
                        timeout_seconds=2.0,
                        max_retries=0,
                        implementation=execute_action,
                    )
                )
                result = action_registry.invoke(
                    "propose_case_update",
                    {"case_id": str(row.case_id), "status": "investigated"},
                    identity=identity,
                    scope=identity.business_unit,
                    data_classification=int(row.get("data_sensitivity", 0)),
                    severity=Severity(severity),
                )
                receipt_executed, receipt_action_id = result.executed, result.action_id  # type: ignore[attr-defined]
            else:
                unrestricted = execute_action(ActionInput(case_id=str(row.case_id), status="investigated"))
                receipt_executed, receipt_action_id = unrestricted.executed, unrestricted.action_id
            action_executed, action_id = receipt_executed, receipt_action_id
            tool_calls.append("propose_case_update")
        state.applicable_controls = [control_match.group(0)] if control_match else []
        state.applicable_regulations = [regulation_match.group(0)] if regulation_match else []
        state.structured_evidence = {
            item.evidence_id: {"source": item.source, "content_sha256": item.content_sha256} for item in evidence
        }
        if risk_probabilities is not None:
            state.risk_prediction = RiskPrediction(
                model_id="calibrated-logistic-risk-v1",
                probabilities={Severity(label): float(risk_probabilities[index]) for index, label in enumerate(LABELS)},
                predicted_severity=Severity(severity),
                calibrated=config.calibration,
                prediction_time=state.system_time,
            )
            state.risk_uncertainty = 1.0 - confidence
        if config.anomaly and risk_requested:
            state.anomaly_signal = AnomalySignal(
                model_id="isolation-forest-signal-v1",
                score=anomaly_score,
                threshold=self.risk.anomaly_threshold,
                is_anomaly=anomaly_score > self.risk.anomaly_threshold,
            )
        state.proposed_actions = [
            ProposedAction(
                action_type="propose_case_update",
                payload={"status": "investigated"},
                risk_tier=2 if severity in {"HIGH", "CRITICAL"} else 1,
                rollback_available=True,
            )
        ]
        state.verification_result = verification
        state.authorization_result = authorization
        state.final_disposition = Disposition(disposition)
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            safe_case_key = hashlib.sha256(str(row.case_id).encode()).hexdigest()
            atomic_write_json(
                self.state_dir / f"{safe_case_key}.json",
                state.model_dump(mode="json"),
            )
        return WorkflowTrace(
            case_id=str(row.case_id),
            predicted_severity=severity,
            predicted_disposition=disposition,
            risk_confidence=confidence,
            anomaly_score=anomaly_score,
            retrieved_ids=[item.evidence_id for item in evidence],
            tool_calls=tool_calls,
            evidence_verified=evidence_verified,
            temporal_correct=temporal,
            feature_event_timestamp=str(row.get("feature_event_timestamp", row.event_timestamp)),
            feature_system_known_at=str(row.get("feature_system_known_at", row.event_timestamp)),
            authorization_outcome=authorization_outcome,
            human_review_requested=human_review,
            structured_output_valid=structured_valid,
            action_executed=action_executed,
            action_id=action_id,
            injection_detected=injection_detected,
            latency_seconds=time.perf_counter() - started,
            tool_argument_errors=tool_argument_errors,
        )
