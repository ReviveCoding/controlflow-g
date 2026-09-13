from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from controlflow.core.state import sha256_file
from controlflow.v22.bundle import verify_bundle, write_bundle
from controlflow.v22.candidate import CandidateExecutionWorkflow, CandidateModelBundle
from controlflow.v22.checkpoint import REQUIRED_FIELDS
from controlflow.v22.dgp import RUNTIME_FORBIDDEN, contamination_report, generate_split
from controlflow.v22.schemas import CandidateResult, PolicyDecision, Severity


def test_latent_truth_is_physically_absent_from_runtime_and_splits_are_independent(tmp_path: Path) -> None:
    paths = {}
    for role, seed in (("TRAIN", 1), ("CALIBRATION", 2), ("VALIDATION", 3)):
        directory = tmp_path / role
        generate_split(directory, role=role, count=30, seed=seed, prefix=f"T{role}")
        paths[role] = directory / "runtime_cases.parquet"
        runtime = pd.read_parquet(paths[role])
        truth = pd.read_parquet(directory / "evaluator_truth.parquet")
        evidence = pd.read_parquet(directory / "evidence_corpus.parquet")
        authorization = pd.read_parquet(directory / "authorization_state.parquet")
        assert not (set(runtime.columns) & RUNTIME_FORBIDDEN)
        assert "latent" in truth.columns
        assert set(runtime.case_id) == set(truth.case_id)
        assert evidence.case_id.isna().all()
        assert set(authorization.identity) == set(runtime.authenticated_identity)
        assert "expected_evidence_ids" not in runtime
    report = contamination_report(paths)
    assert report["leakage_findings"] == 0


def test_evaluator_cardinality_fails_closed_on_missing_case_or_event() -> None:
    from controlflow.v22.evaluation import assert_exact_evaluation_cardinality

    truth = pd.DataFrame({"case_id": ["A", "B"]})
    results = pd.DataFrame({"case_id": ["A"], "execution_event_id": ["E-A"]})
    events = pd.DataFrame({"event_id": ["E-A"]})
    with pytest.raises(RuntimeError, match="EVALUATION_CARDINALITY_MISMATCH"):
        assert_exact_evaluation_cardinality(truth, results, events)
    results = pd.DataFrame({"case_id": ["A", "B"], "execution_event_id": ["E-A", "E-B"]})
    with pytest.raises(RuntimeError, match="EVALUATION_CARDINALITY_MISMATCH"):
        assert_exact_evaluation_cardinality(truth, results, events)


def test_bundle_verification_fails_closed_on_tamper(tmp_path: Path) -> None:
    names = {
        "critical_model",
        "calibrator",
        "noncritical_model",
        "root_model",
        "novelty_model",
        "retrieval_config",
        "temporal_config",
        "policy_config",
        "action_registry",
        "approval_public_key",
        "prompt",
        "schema",
        "serving_config",
    }
    bindings = {}
    for name in names:
        artifact = tmp_path / f"{name}.bin"
        artifact.write_bytes(name.encode())
        bindings[name] = {"path": artifact.name, "sha256": sha256_file(artifact)}
    path = tmp_path / "bundle.json"
    write_bundle(
        path,
        {
            **bindings,
            "critical_threshold": 0.2,
            "embedding_revision": "revision",
            "qwen_model": "Qwen/model",
            "qwen_served_model": "served-model",
            "qwen_revision": "commit",
            "vllm_version": "0.29.0",
            "structured_output_backend": "xgrammar",
            "tabular_inference_device": "cpu",
        },
    )
    verify_bundle(path, tmp_path)
    (tmp_path / "critical_model.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="CANDIDATE_BUNDLE_MISMATCH"):
        verify_bundle(path, tmp_path)


class StubRunner:
    run_dataset = CandidateExecutionWorkflow.run_dataset

    def run_case(self, row, **_):
        return CandidateResult(
            case_id=row["case_id"],
            severity=Severity.LOW,
            critical_probability=0.1,
            root_cause="PROCESS",
            novelty=0.1,
            uncertainty=0.1,
            disposition=PolicyDecision.ALLOW,
            proposed_action="CLOSE_NO_ACTION",
            evidence_ids=(),
            policy_id="policy-2026",
            policy_decision_id="decision",
            execution_event_id="event",
            committed=True,
            rationale="deterministic",
            rationale_claims=(),
            non_llm_latency_seconds=0.01,
            total_latency_seconds=0.01,
        )


def _checkpoint_fields() -> dict[str, object]:
    return {name: 1 for name in REQUIRED_FIELDS}


def test_actual_runner_resume_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    runner = StubRunner()
    partial = tmp_path / "partial.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    output = tmp_path / "output.parquet"
    runner.run_dataset(
        pd.DataFrame([{"case_id": "A"}]),
        output_path=output,
        partial_path=partial,
        checkpoint_path=checkpoint,
        checkpoint_fields=_checkpoint_fields(),
        checkpoint_interval=1,
    )
    changed = _checkpoint_fields()
    changed["seed"] = 2
    with pytest.raises(RuntimeError, match="CHECKPOINT_INCOMPATIBLE"):
        runner.run_dataset(
            pd.DataFrame([{"case_id": "A"}, {"case_id": "B"}]),
            output_path=output,
            partial_path=partial,
            checkpoint_path=checkpoint,
            checkpoint_fields=changed,
        )


def test_hard_critical_route_cannot_be_downgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    class Classifier:
        classes_ = np.asarray(["LOW", "MEDIUM", "HIGH"])

        def predict_proba(self, _frame):
            return np.asarray([[0.0, 0.0, 1.0]])

    class Root:
        def predict(self, _frame):
            return np.asarray(["PROCESS"])

    class Preprocessor:
        def transform(self, _frame):
            return np.asarray([[0.0]])

    class Novelty:
        def score_samples(self, _frame):
            return np.asarray([-0.1])

    bundle = CandidateModelBundle.__new__(CandidateModelBundle)
    bundle.threshold = 0.5
    bundle.critical = object()
    bundle.calibrator = {}
    bundle.noncritical = Classifier()
    bundle.root_model = Root()
    bundle.novelty = {"preprocessor": Preprocessor(), "model": Novelty(), "threshold": 0.5}
    monkeypatch.setattr("controlflow.v22.candidate.calibrated_probability", lambda *_: np.asarray([0.9]))
    decision = bundle.infer(pd.DataFrame([{"observable": 1}]), hard_critical_routing=True)
    assert decision.severity is Severity.CRITICAL
    assert bundle.infer(pd.DataFrame([{"observable": 1}]), hard_critical_routing=False).severity is Severity.HIGH
