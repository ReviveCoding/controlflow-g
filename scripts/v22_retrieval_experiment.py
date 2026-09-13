from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from controlflow.core.state import atomic_write_json, utc_now
from controlflow.v22.retrieval import EvidenceRetriever
from controlflow.v22.schemas import AuthenticatedContext, EvidenceDocument, RuntimeCase

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/v22/runtime.yaml").read_text(encoding="utf-8"))
    namespace = config["development"]["namespace"]
    base = ROOT / f"data/v22/{namespace}/validation"
    runtime = pd.read_parquet(base / "runtime_cases.parquet")
    truth = pd.read_parquet(base / "evaluator_truth.parquet").set_index("case_id")
    documents = [
        EvidenceDocument.model_validate(row)
        for row in pd.read_parquet(base / "evidence_corpus.parquet").to_dict(orient="records")
    ]
    candidates = {
        "LEXICAL": {"lexical_weight": 1.0, "control_metadata_weight": 0.0, "case_metadata_weight": 0.0},
        "HYBRID_METADATA": {
            "lexical_weight": 1.0,
            "control_metadata_weight": 3.0,
            "case_metadata_weight": 8.0,
        },
        "METADATA_ONLY": {
            "lexical_weight": 0.0,
            "control_metadata_weight": 3.0,
            "case_metadata_weight": 8.0,
        },
    }
    results = {}
    for name, weights in candidates.items():
        retriever = EvidenceRetriever(documents, top_k=int(config["retrieval"]["top_k"]), **weights)
        required = retrieved = exact_cases = 0
        for item in runtime.to_dict(orient="records"):
            case = RuntimeCase.model_validate(item)
            context = AuthenticatedContext(
                identity=case.authenticated_identity,
                role=case.role,
                business_unit=case.business_unit,
                region=case.region,
                clearance=case.clearance,
                purpose=case.purpose,
                requested_scope=case.requested_scope,
                data_classification=case.data_classification,
            )
            selected = retriever.retrieve(case, context)
            expected = set(truth.loc[case.case_id, "expected_evidence_ids"])
            required += len(expected)
            retrieved += len(expected & set(selected))
            exact_cases += set(selected) == expected
        results[name] = {
            "required_documents": required,
            "retrieved_required_documents": retrieved,
            "evidence_completeness": None if required == 0 else retrieved / required,
            "exact_case_sets": exact_cases,
            "case_count": len(runtime),
            "weights": weights,
        }
    selected = max(results, key=lambda name: (results[name]["evidence_completeness"], results[name]["exact_case_sets"]))
    atomic_write_json(
        ROOT / f"results/v22/{namespace}_retrieval_experiment.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "selection_split": "VALIDATION",
            "authorization_filter": True,
            "temporal_filter": True,
            "candidates": results,
            "selected": selected,
            "qualification_evidence": False,
        },
    )


if __name__ == "__main__":
    main()
