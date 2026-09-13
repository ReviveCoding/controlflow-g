from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from controlflow.v22.temporal import CandidateTemporalRetriever, load_policy_corpus

ROOT = Path(__file__).resolve().parents[2]


def test_temporal_candidate_handles_history_correction_future_and_invalid_overlap() -> None:
    retriever = CandidateTemporalRetriever(load_policy_corpus(ROOT / "configs/v22/temporal_policies.yaml"))
    system_2024 = datetime(2024, 8, 1, tzinfo=UTC)
    system_2026 = datetime(2026, 9, 1, tzinfo=UTC)
    assert retriever.select(datetime(2021, 5, 1, tzinfo=UTC), system_2026) == "policy-2021"
    assert retriever.select(datetime(2023, 5, 1, tzinfo=UTC), system_2024) == "policy-2023"
    assert retriever.select(datetime(2023, 5, 1, tzinfo=UTC), system_2026) == "policy-2023-correction"
    assert retriever.select(datetime(2025, 5, 1, tzinfo=UTC), system_2026) == "policy-2024"
    assert retriever.select(datetime(2026, 5, 1, tzinfo=UTC), system_2026) == "policy-2026"
    assert retriever.select(datetime(2022, 6, 5, tzinfo=UTC), system_2026) is None


def test_candidate_matches_independently_authored_temporal_truth_fixtures() -> None:
    retriever = CandidateTemporalRetriever(load_policy_corpus(ROOT / "configs/v22/temporal_policies.yaml"))
    fixtures = yaml.safe_load((ROOT / "configs/v22/temporal_truth_fixtures.yaml").read_text())["fixtures"]
    for fixture in fixtures:
        event_time = datetime.fromisoformat(str(fixture["event_time"]).replace("Z", "+00:00"))
        system_time = datetime.fromisoformat(str(fixture["system_time"]).replace("Z", "+00:00"))
        assert retriever.select(event_time, system_time) == fixture.get("expected_policy_id")
