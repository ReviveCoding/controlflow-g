from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Any]: ...


class FeatureStore(Protocol):
    def load(self, entity_ids: list[str], as_of: pd.Timestamp) -> pd.DataFrame: ...


class ExperimentTracker(Protocol):
    def log(self, group: str, record: dict[str, Any], artifacts: list[Path]) -> str: ...


class PolicyBackend(Protocol):
    def authorize(self, identity: Any, request: Any) -> Any: ...


class DataWarehouse(Protocol):
    def select(self, validated_sql: str) -> pd.DataFrame: ...


class ModelEndpoint(Protocol):
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]: ...
