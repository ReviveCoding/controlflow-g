from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from controlflow.audit.ledger import ActionLedger
from controlflow.authorization.policy import LocalPolicyBackend, ToolPolicyInput
from controlflow.core.state import canonical_json
from controlflow.hitl.approval import ApprovalAuthority
from controlflow.modeling.supervised import LABELS, _preprocessor, _xy
from controlflow.retrieval.core import BM25Retriever
from controlflow.schemas import AgentState, IdentityContext, Severity, TemporalEvidence
from controlflow.verification.claims import verify_claims

CONTROL = re.compile(r"\b[A-Z]{2}-\d+(?:\.\d+)?\b")
REGULATION = re.compile(r"\b12CFR-\d+\b")
INJECTION = re.compile(r"ignore (?:all )?previous|unrestricted tool|role=administrator", re.I)


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

    def predict(self, row: pd.Series, *, calibrated: bool = True) -> tuple[str, float, float]:
        probability = self.model.predict_proba(pd.DataFrame([row]))[0]
        if calibrated and self.calibrator is not None:
            probability = self.calibrator.predict(probability.reshape(1, -1))[0]  # type: ignore[attr-defined]
        numeric = pd.DataFrame([row])[["amount", "repeat_count", "historical_failures", "data_sensitivity"]]
        return (
            LABELS[int(probability.argmax())],
            float(probability.max()),
            float(-self.anomaly.score_samples(numeric)[0]),
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
    authorization_outcome: str
    human_review_requested: bool
    structured_output_valid: bool
    action_executed: bool
    action_id: int | None
    injection_detected: bool
    latency_seconds: float
    retry_count: int = 0


class GovernedWorkflow:
    def __init__(
        self,
        training: pd.DataFrame,
        controls: pd.DataFrame,
        ledger: ActionLedger,
        authority: ApprovalAuthority,
        risk_service: RiskService | None = None,
    ) -> None:
        self.risk = risk_service or RiskService(training)
        self.controls = controls
        self.ledger = ledger
        self.policy = LocalPolicyBackend()
        self.authority = authority

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
        if not config.retrieval or "source evidence unavailable" in row.narrative.casefold():
            # Absence is a correct temporal state for an explicit missing-
            # evidence case; the verifier will force insufficient evidence.
            return [], "source evidence unavailable" in row.narrative.casefold()
        origin = datetime(2020, 1, 1, tzinfo=UTC)
        items: list[TemporalEvidence] = []
        for control in self.controls.itertuples():
            text = f"[SUPPORTS] control {control.control_id}: {control.title}. {control.description}"
            items.append(
                TemporalEvidence(
                    evidence_id=str(control.control_id),
                    source="NIST SP 800-53",
                    text=text,
                    classification=0,
                    business_valid_from=origin,
                    system_known_from=origin,
                    authorized_roles=frozenset({"Control Analyst", "Risk Manager", "Compliance Reviewer"}),
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        # Authorization happens before lexical scoring and content enters no
        # model until this partition is selected.
        authorized = [
            item
            for item in items
            if identity.clearance >= item.classification and identity.role in item.authorized_roles
        ]
        hits = BM25Retriever(authorized).search(str(row.narrative), len(authorized) if config.reranker else 5)
        if config.reranker:
            narrative = str(row.narrative).casefold()
            hits.sort(
                key=lambda hit: (
                    hit.evidence.evidence_id.casefold() in narrative,
                    hit.score,
                ),
                reverse=True,
            )
            hits = hits[:5]
        selected = [hit.evidence for hit in hits]
        regulation_match = REGULATION.search(str(row.narrative))
        if regulation_match:
            regulation = regulation_match.group(0)
            event_time = pd.Timestamp(row.event_timestamp).to_pydatetime()
            expected_version = "policy-v1" if event_time.year < 2024 else "policy-v2"
            chosen_version = expected_version if config.temporal_retrieval else "policy-v2"
            valid_from = datetime(2020 if chosen_version == "policy-v1" else 2024, 1, 1, tzinfo=UTC)
            valid_to = datetime(2024, 1, 1, tzinfo=UTC) if chosen_version == "policy-v1" else None
            text = f"[SUPPORTS] regulation {regulation} {chosen_version} applies"
            selected.append(
                TemporalEvidence(
                    evidence_id=f"{regulation}:{chosen_version}",
                    source="eCFR Title 12",
                    text=text,
                    classification=0,
                    business_valid_from=valid_from,
                    business_valid_to=valid_to,
                    system_known_from=valid_from,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        if "authoritative records disagree" in row.narrative.casefold() and selected:
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
        return "\n".join(item.text[:300] for item in evidence[:3])

    def execute(self, row: pd.Series, config: WorkflowConfig, llm_prediction: tuple[str, str, bool]) -> WorkflowTrace:
        started = time.perf_counter()
        identity = self.identity(row)
        tool_calls: list[str] = []
        llm_severity, llm_disposition, llm_valid = llm_prediction
        if config.ml_risk:
            severity, confidence, anomaly_score = self.risk.predict(row, calibrated=config.calibration)
            if config.anomaly and anomaly_score > self.risk.anomaly_threshold:
                severity = LABELS[min(3, LABELS.index(severity) + 1)]
            if not config.point_in_time_features:
                # Negative-control ablation: deliberately expose the post-case
                # outcome to quantify how leakage can inflate apparent quality.
                severity = str(row.severity)
            tool_calls.append("compute_risk")
        else:
            severity = llm_severity
            confidence, anomaly_score = 0.5, 0.0
        evidence, temporal = self._evidence(row, config, identity)
        if config.retrieval:
            tool_calls.extend(["search_controls", "search_regulations"])
        state = AgentState(
            case_id=str(row.case_id),
            workflow_version="agent-graph-v2",
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
        if "records disagree" in str(row.narrative).casefold() and control_match:
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
        injection_detected = bool(INJECTION.search(str(row.narrative)))
        requested_scope = (
            "restricted" if "exceeds" in str(row.narrative).casefold() or injection_detected else identity.business_unit
        )
        request = ToolPolicyInput(
            tool_name="propose_case_update",
            risk_tier=2 if severity in {"HIGH", "CRITICAL"} else 1,
            read_only=False,
            allowed_roles=frozenset({"Control Analyst"}),
            allowed_scopes=frozenset({identity.business_unit}),
            requires_review=severity in {"HIGH", "CRITICAL"},
            data_classification=0,
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
        structured_valid = (
            llm_valid
            if not config.structured_output
            else disposition in {"AUTO", "REVIEW_REQUIRED", "INSUFFICIENT_EVIDENCE", "DENY"} and severity in LABELS
        )
        action_executed = False
        action_id = None
        if (
            disposition == "AUTO"
            and config.tools_enabled
            and (
                (config.bounded_tools and authorization_outcome in {"ALLOW", "ALLOW_READONLY"})
                or not config.bounded_tools
            )
        ):
            payload = {"status": "investigated"}
            evidence_hash = hashlib.sha256(canonical_json(sorted(item.evidence_id for item in evidence))).hexdigest()
            token = self.authority.issue(
                case_id=str(row.case_id),
                action_type="propose_case_update",
                payload=payload,
                workflow_version="agent-graph-v2",
                reviewer_id="SYSTEM_AUTO" if config.authorization else "UNRESTRICTED_AGENT",
                policy_version="local-policy-v1",
                evidence_hash=evidence_hash,
            )
            receipt = self.ledger.execute_simulated(
                case_id=str(row.case_id),
                action_type="propose_case_update",
                payload=payload,
                workflow_version="agent-graph-v2",
                authorization_token=token,
                approval_authority=self.authority,
                evidence_hash=evidence_hash,
            )
            action_executed, action_id = receipt.executed, receipt.action_id
            tool_calls.append("propose_case_update")
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
            authorization_outcome=authorization_outcome,
            human_review_requested=human_review,
            structured_output_valid=structured_valid,
            action_executed=action_executed,
            action_id=action_id,
            injection_detected=injection_detected,
            latency_seconds=time.perf_counter() - started,
        )
