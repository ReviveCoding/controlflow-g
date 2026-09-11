from __future__ import annotations

import pytest

from controlflow.tools.sql_safety import UnsafeQuery, validate_readonly_sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM gold.fact_case",
        "UPDATE gold.fact_case SET severity='LOW'",
        "DROP TABLE gold.fact_case",
        "CREATE TABLE x AS SELECT * FROM gold.fact_case",
        "SELECT * FROM gold.fact_case; SELECT * FROM gold.fact_case_event",
    ],
)
def test_rejects_write_ddl_and_multiple_statements(sql: str) -> None:
    with pytest.raises(UnsafeQuery):
        validate_readonly_sql(sql, frozenset({"fact_case"}))


def test_enforces_allowlist_and_limit() -> None:
    query = validate_readonly_sql("SELECT case_id FROM fact_case", frozenset({"fact_case"}), 25)
    assert query.row_limit == 25
    assert "LIMIT 25" in query.sql
    with pytest.raises(UnsafeQuery, match="unauthorized"):
        validate_readonly_sql("SELECT * FROM secret", frozenset({"fact_case"}))


def test_rejects_allowlisted_basename_in_unauthorized_schema() -> None:
    with pytest.raises(UnsafeQuery):
        validate_readonly_sql("SELECT * FROM secret_catalog.fact_case", frozenset({"gold.fact_case"}))


@pytest.mark.parametrize("function", ["REFLECT(a,b,c)", "JAVA_METHOD(a,b,c)", "INPUT_FILE_NAME()"])
def test_rejects_denied_spark_functions(function: str) -> None:
    with pytest.raises(UnsafeQuery):
        validate_readonly_sql(f"SELECT {function} FROM gold.fact_case", frozenset({"gold.fact_case"}))
