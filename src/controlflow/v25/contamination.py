"""Fresh holdout overlap checks against all earlier runtime cohorts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from controlflow.v22.dgp import contamination_against_prior, resolve_prior_runtime_paths


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def check_contamination(root: Path, candidate_path: Path) -> dict[str, Any]:
    prior = resolve_prior_runtime_paths(
        root,
        root / "configs/v25/prior_runtime_manifest.yaml",
        excluded_directory=candidate_path.parent,
    )
    report = contamination_against_prior(candidate_path, prior)
    candidate = pd.read_parquet(candidate_path)
    candidate_texts = {
        _normalize(text): str(case_id) for case_id, text in zip(candidate.case_id, candidate.narrative, strict=True)
    }
    exact_text: list[dict[str, str]] = []
    for path in prior:
        frame = pd.read_parquet(path)
        if "narrative" not in frame or "case_id" not in frame:
            continue
        for case_id, narrative in zip(frame.case_id, frame.narrative, strict=True):
            current = candidate_texts.get(_normalize(narrative))
            if current is not None:
                exact_text.append(
                    {"candidate_case_id": current, "prior_case_id": str(case_id), "prior_path": path.as_posix()}
                )
    report["exact_normalized_text_matches"] = exact_text
    report["leakage_findings"] += len(exact_text)
    report["prior_path_count"] = len(prior)
    report["v23qual_use"] = "overlap_protection_only"
    return report
