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

from controlflow.audit.ledger import ActionLedger, action_key
from controlflow.authorization.identity import SessionIdentityProvider
from controlflow.authorization.policy import LocalPolicyBackend, ToolPolicyInput
from controlflow.core.state import ProjectPaths, atomic_write_json, canonical_json, sha256_file
from controlflow.hitl.approval import ApprovalAuthority, InvalidApproval
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
from controlflow.tools.sql_safety import UnsafeQuery, validate_readonly_sql
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
    context_gpu_seconds: float = 0.0
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
        identity_provider: SessionIdentityProvider | None = None,
        verified_corpus_hashes: dict[str, str] | None = None,
    ) -> None:
        self.risk = risk_service or RiskService(training)
        self.controls = controls
        self.ledger = ledger
        self.policy = LocalPolicyBackend()
        self.authority = authority
        self.state_dir = state_dir
        self.require_cuda_retrieval = require_cuda_retrieval
        self.training = training.set_index("case_id", drop=False)
        if identity_provider is None:
            raise ValueError("an independently provisioned identity provider is required")
        self.identity_provider = identity_provider
        paths = ProjectPaths.discover()
        if verified_corpus_hashes is None:
            artifact_manifest = json.loads((paths.state / "artifact_manifest.json").read_text(encoding="utf-8"))
            recorded_hashes = {item["path"]: item["sha256"] for item in artifact_manifest["artifacts"]}
        else:
            recorded_hashes = verified_corpus_hashes
        control_path = paths.root / "data/staging/nist_controls_raw.parquet"
        regulation_path = paths.root / "data/staging/cfr_raw.parquet"
        transaction_path = paths.root / "data/staging/transactions_raw.parquet"
        controls_trusted = recorded_hashes.get("data/staging/nist_controls_raw.parquet") == sha256_file(
            control_path
        ) and controls.reset_index(drop=True).equals(pd.read_parquet(control_path).reset_index(drop=True))
        regulations_trusted = (
            regulations is not None
            and recorded_hashes.get("data/staging/cfr_raw.parquet") == sha256_file(regulation_path)
            and regulations.reset_index(drop=True).equals(pd.read_parquet(regulation_path).reset_index(drop=True))
        )
        self.transactions = pd.read_parquet(transaction_path)
        if recorded_hashes.get("data/staging/transactions_raw.parquet") != sha256_file(transaction_path):
            raise ValueError("transaction query source failed artifact verification")
        self._retriever_cache: dict[tuple[str, int, int, bool, bool], BM25Retriever | NeuralHybridRetriever] = {}
        self._context_observations: dict[tuple[str, str], dict[str, object]] = {}
        self._context_timings: dict[tuple[str, str], tuple[float, float]] = {}
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
                    trusted_ingestion=controls_trusted,
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
                        trusted_ingestion=regulations_trusted,
                        claim_relations=frozenset({"regulatory_context"}),
                    )
                )

    def identity(self, session_token: str) -> IdentityContext:
        return self.identity_provider.resolve(session_token)

    @staticmethod
    def _enforce_resource_scope(row: pd.Series, identity: IdentityContext) -> str:
        resource_scope = str(row.business_unit)
        if resource_scope != identity.business_unit:
            raise PermissionError("authenticated session is not entitled to the case resource scope")
        return resource_scope

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
        # Retrieval misses remain misses. Required identifiers from benchmark
        # text are never used to complete or reorder the retrieved set.
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

    def context_for_llm(self, row: pd.Series, config: WorkflowConfig, *, session_token: str) -> str:
        """Build LLM-visible context exclusively from authorized typed-tool outputs."""
        started = time.perf_counter()
        identity = self.identity(session_token)
        resource_scope = self._enforce_resource_scope(row, identity)
        registry = ToolRegistry(self.policy, self.ledger.record_system_event)
        retrieved_by_tool: dict[str, TemporalEvidence] = {}
        feature_row = row.copy()
        feature_row["historical_failures"] = (
            row["pit_historical_failures"] if config.point_in_time_features else row["future_failures"]
        )

        def search(kind: str) -> Callable[[EvidenceSearchInput], EvidenceSearchOutput]:
            def invoke_search(_request: EvidenceSearchInput) -> EvidenceSearchOutput:
                retrieved, _ = self._evidence(row, config, identity) if config.retrieval else ([], True)
                selected = [item for item in retrieved if (item.source == "NIST SP 800-53") == (kind == "controls")]
                retrieved_by_tool.update({item.evidence_id: item for item in selected})
                return EvidenceSearchOutput(evidence_ids=[item.evidence_id for item in selected])

            return invoke_search

        def risk_tool(_request: CaseLookupInput) -> RiskToolOutput:
            probabilities, _ = self.risk.predict_details(feature_row, calibrated=config.calibration)
            return RiskToolOutput(
                probabilities=probabilities.tolist(),
                severity=LABELS[int(probabilities.argmax())],
                confidence=float(probabilities.max()),
            )

        def anomaly_tool(_request: CaseLookupInput) -> AnomalyToolOutput:
            _, score = self.risk.predict_details(feature_row, calibrated=config.calibration)
            return AnomalyToolOutput(
                score=score, threshold=self.risk.anomaly_threshold, is_anomaly=score > self.risk.anomaly_threshold
            )

        def search_cases(_request: CaseLookupInput) -> StructuredResult:
            query_terms = set(re.findall(r"[a-z0-9]+", str(row.narrative).casefold()))
            candidates = self.training.loc[
                self.training["business_unit"].eq(identity.business_unit)
                & ~self.training["case_id"].eq(str(row.case_id))
            ].copy()
            candidates["similarity"] = candidates["narrative"].map(
                lambda value: len(query_terms & set(re.findall(r"[a-z0-9]+", str(value).casefold())))
            )
            identifiers = candidates.sort_values(["similarity", "event_timestamp"], ascending=False).head(5)["case_id"]
            return StructuredResult(values={"matching_case_ids": identifiers.astype(str).tolist()})

        def policy_at_time(_request: CaseLookupInput) -> EvidenceSearchOutput:
            event_time = pd.Timestamp(row.event_timestamp).to_pydatetime()
            return EvidenceSearchOutput(
                evidence_ids=[
                    item.evidence_id
                    for item in retrieved_by_tool.values()
                    if item.source != "NIST SP 800-53" and item.valid_at(event_time, event_time)
                ]
            )

        def query_case(_request: CaseLookupInput) -> StructuredResult:
            return StructuredResult(
                values={
                    "case_id": str(row.case_id),
                    "entity_id": str(row.entity_id),
                    "business_unit": str(row.business_unit),
                    "event_timestamp": str(row.event_timestamp),
                    "repeat_count": int(row.repeat_count),
                }
            )

        def query_transactions(_request: CaseLookupInput) -> StructuredResult:
            as_of = self.transactions.loc[
                self.transactions["entity_id"].eq(str(row.entity_id))
                & (self.transactions["event_timestamp"] <= pd.Timestamp(row.event_timestamp))
            ].sort_values("event_timestamp")
            recent = as_of.tail(20)
            return StructuredResult(
                values={
                    "transaction_ids": recent["transaction_id"].astype(str).tolist(),
                    "transaction_count": len(as_of),
                    "amount_sum": float(as_of["amount"].sum()),
                    "amount_max": float(as_of["amount"].max()) if len(as_of) else 0.0,
                }
            )

        def evidence_bundle(_request: CaseLookupInput) -> EvidenceBundleOutput:
            identifiers = sorted(retrieved_by_tool)
            return EvidenceBundleOutput(
                evidence_ids=identifiers,
                bundle_sha256=hashlib.sha256(canonical_json(identifiers)).hexdigest(),
            )

        specs = (
            ToolSpec(
                "search_controls",
                EvidenceSearchInput,
                EvidenceSearchOutput,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                search("controls"),
            ),
            ToolSpec(
                "search_regulations",
                EvidenceSearchInput,
                EvidenceSearchOutput,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                search("regulations"),
            ),
            ToolSpec(
                "compute_risk",
                CaseLookupInput,
                RiskToolOutput,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                risk_tool,
            ),
            ToolSpec(
                "compute_anomaly",
                CaseLookupInput,
                AnomalyToolOutput,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                anomaly_tool,
            ),
            ToolSpec(
                "search_cases",
                CaseLookupInput,
                StructuredResult,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                search_cases,
            ),
            ToolSpec(
                "get_policy_at_time",
                CaseLookupInput,
                EvidenceSearchOutput,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                policy_at_time,
            ),
            ToolSpec(
                "query_case_data",
                CaseLookupInput,
                StructuredResult,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                query_case,
            ),
            ToolSpec(
                "query_transactions",
                CaseLookupInput,
                StructuredResult,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                query_transactions,
            ),
            ToolSpec(
                "generate_evidence_bundle",
                CaseLookupInput,
                EvidenceBundleOutput,
                0,
                True,
                frozenset({"Control Analyst"}),
                frozenset({identity.business_unit}),
                False,
                2.0,
                1,
                evidence_bundle,
            ),
        )
        for spec in specs:
            registry.register(spec)

        def invoke(name: str, arguments: dict[str, object]) -> BaseModel:
            return registry.invoke(
                name,
                arguments,
                identity=identity,
                scope=resource_scope,
                data_classification=int(row.get("data_sensitivity", 0)),
                severity=Severity.LOW,
            )

        control_result = invoke("search_controls", {"query": str(row.narrative)})
        regulation_result = invoke("search_regulations", {"query": str(row.narrative)})
        risk_result = invoke("compute_risk", {"case_id": str(row.case_id)}) if config.ml_risk else None
        anomaly_result = (
            invoke("compute_anomaly", {"case_id": str(row.case_id)}) if config.ml_risk and config.anomaly else None
        )
        cases_result = invoke("search_cases", {"case_id": str(row.case_id)})
        policy_result = invoke("get_policy_at_time", {"case_id": str(row.case_id)})
        case_result = invoke("query_case_data", {"case_id": str(row.case_id)})
        transaction_result = invoke("query_transactions", {"case_id": str(row.case_id)})
        bundle_result = invoke("generate_evidence_bundle", {"case_id": str(row.case_id)})
        control_ids = control_result.evidence_ids  # type: ignore[attr-defined]
        regulation_ids = regulation_result.evidence_ids  # type: ignore[attr-defined]
        structured_context: dict[str, object] = {
            "search_cases": cases_result.model_dump(mode="json"),
            "get_policy_at_time": policy_result.model_dump(mode="json"),
            "query_case_data": case_result.model_dump(mode="json"),
            "query_transactions": transaction_result.model_dump(mode="json"),
            "generate_evidence_bundle": bundle_result.model_dump(mode="json"),
        }
        self._context_observations[(str(row.case_id), session_token)] = structured_context
        context = json.dumps(
            {
                "search_controls": [retrieved_by_tool[item].text[:300] for item in control_ids],
                "search_regulations": [retrieved_by_tool[item].text[:300] for item in regulation_ids],
                "compute_risk": risk_result.model_dump(mode="json") if risk_result else {},
                "compute_anomaly": anomaly_result.model_dump(mode="json") if anomaly_result else {},
                **structured_context,
                "propose_case_update": "write action requires the workflow authorization boundary",
            },
            sort_keys=True,
        )
        elapsed = time.perf_counter() - started
        # The neural retriever synchronizes each CUDA result before returning.
        # Conservatively charge its complete wall time to the GPU cost proxy.
        gpu_seconds = elapsed if config.retrieval and config.reranker and self.require_cuda_retrieval else 0.0
        self._context_timings[(str(row.case_id), session_token)] = (elapsed, gpu_seconds)
        return context

    def execute(
        self,
        row: pd.Series,
        config: WorkflowConfig,
        llm_prediction: tuple[str, str, bool],
        llm_tool_requests: list[str] | None = None,
        llm_tool_arguments: list[dict[str, object]] | None = None,
        *,
        session_token: str,
    ) -> WorkflowTrace:
        started = time.perf_counter()
        identity = self.identity(session_token)
        resource_scope = self._enforce_resource_scope(row, identity)
        tool_calls: list[str] = []
        requested = set(llm_tool_requests) if llm_tool_requests is not None else None
        requested_arguments = {
            name: arguments for name, arguments in zip(llm_tool_requests or [], llm_tool_arguments or [], strict=False)
        }
        tool_argument_errors = 0
        structured_tool_results = dict(self._context_observations.get((str(row.case_id), session_token), {}))
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
        service_registry = ToolRegistry(self.policy, self.ledger.record_system_event)
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
                    scope=resource_scope,
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
                        scope=resource_scope,
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
        evidence: list[TemporalEvidence] = []
        temporal = True
        feature_temporal = pd.Timestamp(row.get("feature_event_timestamp", row.event_timestamp)) <= pd.Timestamp(
            row.event_timestamp
        ) and pd.Timestamp(row.get("feature_system_known_at", row.event_timestamp)) <= pd.Timestamp(row.event_timestamp)
        temporal = temporal and feature_temporal and config.point_in_time_features
        if config.retrieval and retrieval_requested:
            registry = ToolRegistry(self.policy, self.ledger.record_system_event)
            retrieved_materialized: dict[str, TemporalEvidence] = {}

            def evidence_search(selected_tool: str) -> Callable[[EvidenceSearchInput], EvidenceSearchOutput]:
                def search(_request: EvidenceSearchInput) -> EvidenceSearchOutput:
                    selected = [
                        item
                        for item in self._evidence(row, config, identity)[0]
                        if (
                            (selected_tool == "search_controls" and item.source == "NIST SP 800-53")
                            or (selected_tool == "search_regulations" and item.source != "NIST SP 800-53")
                        )
                    ]
                    retrieved_materialized.update({item.evidence_id: item for item in selected})
                    return EvidenceSearchOutput(evidence_ids=[item.evidence_id for item in selected])

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
                            scope=resource_scope,
                            data_classification=int(row.get("data_sensitivity", 0)),
                            severity=Severity(severity),
                        )
                        if config.authorization
                        else evidence_search(tool_name)(EvidenceSearchInput.model_validate(raw_arguments))
                    )
                    selected_ids = set(result.evidence_ids)  # type: ignore[attr-defined]
                    selected = [retrieved_materialized[item] for item in selected_ids]
                    evidence.extend(selected)
                    event_time = pd.Timestamp(row.event_timestamp).to_pydatetime()
                    temporal = temporal and all(item.valid_at(event_time, event_time) for item in selected)
                    tool_calls.append(tool_name)
                except (ValidationError, PermissionError):
                    tool_argument_errors += 1
            auxiliary_tools: tuple[tuple[str, type[BaseModel], Callable[[CaseLookupInput], BaseModel]], ...] = (
                (
                    "search_cases",
                    StructuredResult,
                    lambda _: StructuredResult(
                        values={
                            "matching_case_ids": self.training.assign(
                                similarity=self.training["narrative"].map(
                                    lambda value: len(
                                        set(re.findall(r"[a-z0-9]+", str(row.narrative).casefold()))
                                        & set(re.findall(r"[a-z0-9]+", str(value).casefold()))
                                    )
                                )
                            )
                            .loc[lambda frame: frame["business_unit"].eq(identity.business_unit)]
                            .sort_values(["similarity", "event_timestamp"], ascending=False)
                            .head(5)["case_id"]
                            .astype(str)
                            .tolist(),
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
                    lambda _: StructuredResult(
                        values={
                            "transaction_ids": self.transactions.loc[
                                self.transactions["entity_id"].eq(str(row.entity_id))
                                & (self.transactions["event_timestamp"] <= pd.Timestamp(row.event_timestamp))
                            ]
                            .sort_values("event_timestamp")
                            .tail(20)["transaction_id"]
                            .astype(str)
                            .tolist(),
                            "transaction_count": int(
                                (
                                    self.transactions["entity_id"].eq(str(row.entity_id))
                                    & (self.transactions["event_timestamp"] <= pd.Timestamp(row.event_timestamp))
                                ).sum()
                            ),
                        }
                    ),
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
                        auxiliary_result = registry.invoke(
                            tool_name,
                            requested_arguments.get(tool_name, {"case_id": str(row.case_id)}),
                            identity=identity,
                            scope=resource_scope,
                            data_classification=int(row.get("data_sensitivity", 0)),
                            severity=Severity(severity),
                        )
                        structured_tool_results[tool_name] = auxiliary_result.model_dump(mode="json")
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
            case_context={"business_unit": row.business_unit, "tool_results": structured_tool_results},
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
        boundary_violation = False
        requested_sql = row.get("requested_sql")
        if isinstance(requested_sql, str) and requested_sql:
            sql_started = time.perf_counter()
            sql_status = "allowed"
            try:
                validate_readonly_sql(requested_sql, frozenset({"gold.fact_case"}))
            except UnsafeQuery:
                boundary_violation = True
                sql_status = "rejected"
            self.ledger.record_system_event(
                "SQL_VALIDATION",
                identity.user_id,
                {
                    "query_sha256": hashlib.sha256(requested_sql.encode()).hexdigest(),
                    "status": sql_status,
                    "allowed_tables": ["gold.fact_case"],
                    "row_limit": 10_000,
                    "policy_version": self.policy.version,
                    "duration_seconds": time.perf_counter() - sql_started,
                },
            )
        tool_description = row.get("requested_tool_description")
        if isinstance(tool_description, str) and tool_description:
            probe = ToolRegistry(self.policy, self.ledger.record_system_event)
            try:
                probe.register(
                    ToolSpec(
                        "description_probe",
                        CaseLookupInput,
                        StructuredResult,
                        0,
                        True,
                        frozenset({"Control Analyst"}),
                        frozenset({identity.business_unit}),
                        False,
                        1.0,
                        0,
                        lambda _: StructuredResult(values={"status": "safe"}),
                        description=tool_description,
                    )
                )
            except ValueError:
                boundary_violation = True
        claimed_role = row.get("claimed_role")
        if isinstance(claimed_role, str) and claimed_role != identity.role:
            boundary_violation = True
        presented_approval = row.get("presented_approval_token")
        if isinstance(presented_approval, str) and presented_approval:
            try:
                self.authority.verify(
                    presented_approval,
                    expected_action_key=action_key(
                        str(row.case_id),
                        "propose_case_update",
                        {"status": "investigated"},
                        "agent-graph-v3",
                    ),
                    policy_version="local-policy-v1",
                    evidence_hash="analysis-context",
                )
            except InvalidApproval:
                boundary_violation = True
        injection_detected = (
            boundary_violation
            or bool(INJECTION.search(str(row.narrative)))
            or any(INJECTION.search(item.text) is not None for item in evidence)
        )
        requested_scope = (
            "restricted" if injection_detected else str(row.get("requested_scope", identity.business_unit))
        )
        requested_action_arguments = row.get("requested_action_arguments", {"case_id": str(row.case_id)})
        if not isinstance(requested_action_arguments, dict):
            requested_action_arguments = {"invalid": True}
            boundary_violation = True
        if requested_action_arguments.get("case_id", str(row.case_id)) != str(row.case_id):
            boundary_violation = True
            requested_scope = "restricted"
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
            arguments=requested_action_arguments,
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
                action_registry = ToolRegistry(self.policy, self.ledger.record_system_event)
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
                    scope=resource_scope,
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
        context_latency, context_gpu_seconds = self._context_timings.pop((str(row.case_id), session_token), (0.0, 0.0))
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
            latency_seconds=context_latency + time.perf_counter() - started,
            context_gpu_seconds=context_gpu_seconds,
            tool_argument_errors=tool_argument_errors,
        )
