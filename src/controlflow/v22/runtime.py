from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from controlflow.v22.approval import ApprovalVerifier
from controlflow.v22.candidate import CandidateModelBundle, CandidateRunner
from controlflow.v22.executor import TransactionalExecutor
from controlflow.v22.policy import ActionRegistry, AuthorizationState, PolicyDecisionPoint
from controlflow.v22.retrieval import EvidenceRetriever
from controlflow.v22.schemas import EvidenceDocument
from controlflow.v22.temporal import CandidateTemporalRetriever, load_policy_corpus


def load_authoritative_state(path: Path) -> AuthorizationState:
    frame = pd.read_parquet(path)
    required = {"identity", "active", "role", "business_unit", "region", "clearance"}
    if not required <= set(frame.columns) or frame.identity.duplicated().any():
        raise RuntimeError("AUTHORITATIVE_STATE_INVALID")
    return AuthorizationState(
        {
            str(row.identity): {
                "active": bool(row.active),
                "role": str(row.role),
                "business_unit": str(row.business_unit),
                "region": str(row.region),
                "clearance": int(row.clearance),
                "authorization_version": str(row.authorization_version),
                "case_risk_version": str(row.case_risk_version),
            }
            for row in frame.itertuples()
        }
    )


def build_candidate_runtime(
    *,
    root: Path,
    bundle_path: Path,
    evidence_path: Path,
    authorization_path: Path,
    ledger_path: Path,
    explanation_client: Any = None,
    fault: str | None = None,
) -> tuple[CandidateRunner, TransactionalExecutor, CandidateModelBundle]:
    """Build the executable runtime exclusively from verified bundle bindings.

    Evidence and current authorization are run inputs, not candidate configuration;
    their hashes belong in the dataset manifest/checkpoint.
    """
    bundle = CandidateModelBundle(bundle_path, root)
    runtime_config = yaml.safe_load(bundle.artifact_path("retrieval_config").read_text(encoding="utf-8"))
    evidence_frame = pd.read_parquet(evidence_path)
    retriever_config = runtime_config["retrieval"]
    evidence = EvidenceRetriever(
        [EvidenceDocument.model_validate(row) for row in evidence_frame.to_dict(orient="records")],
        top_k=int(retriever_config["top_k"]),
        lexical_weight=float(retriever_config["lexical_weight"]),
        control_metadata_weight=float(retriever_config["control_metadata_weight"]),
        case_metadata_weight=float(retriever_config["case_metadata_weight"]),
        minimum_score=float(retriever_config["minimum_score"]),
    )
    registry = ActionRegistry(bundle.artifact_path("action_registry"))
    pdp = PolicyDecisionPoint(
        policy_path=bundle.artifact_path("policy_config"),
        registry=registry,
        authorization=load_authoritative_state(authorization_path),
    )
    executor = TransactionalExecutor(
        ledger_path,
        pdp=pdp,
        registry=registry,
        approval_verifier=ApprovalVerifier(bundle.artifact_path("approval_public_key")),
        candidate_bundle_hash=bundle.bundle_hash,
        fault=fault,
    )
    runner = CandidateRunner(
        bundle=bundle,
        evidence=evidence,
        temporal=CandidateTemporalRetriever(load_policy_corpus(bundle.artifact_path("temporal_config"))),
        executor=executor,
        workflow_version=str(runtime_config["workflow_version"]),
        explanation_client=explanation_client,
    )
    return runner, executor, bundle
