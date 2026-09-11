from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp, parse
from sqlglot.errors import ParseError


class UnsafeQuery(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str
    tables: frozenset[str]
    row_limit: int


def validate_readonly_sql(sql: str, allowed_tables: frozenset[str], max_rows: int = 1000) -> ValidatedQuery:
    try:
        statements = parse(sql, read="spark")
    except ParseError as exc:
        raise UnsafeQuery("SQL parse failed") from exc
    if len(statements) != 1:
        raise UnsafeQuery("exactly one SQL statement is allowed")
    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union, exp.Subquery)):
        raise UnsafeQuery("only SELECT queries are allowed")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Alter, exp.Create)
    if any(statement.find(node) is not None for node in forbidden):
        raise UnsafeQuery("write or DDL expression rejected")
    tables = frozenset(".".join(part.name for part in table.parts).lower() for table in statement.find_all(exp.Table))
    normalized_allowlist = {table.lower() for table in allowed_tables}
    if not tables or not tables.issubset(normalized_allowlist):
        raise UnsafeQuery("query references a missing or unauthorized table")
    denied_functions = {"read_csv", "read_json", "read_parquet", "input_file_name", "java_method", "reflect"}
    functions = {
        (function.name if isinstance(function, exp.Anonymous) else function.key).casefold()
        for function in statement.find_all(exp.Func)
    }
    if functions & denied_functions:
        raise UnsafeQuery("query uses a denied external or reflective function")
    limit = statement.args.get("limit")
    if limit is None:
        statement = statement.limit(max_rows)
        row_limit = max_rows
    else:
        expression = limit.expression
        try:
            requested = int(expression.this)
        except (AttributeError, TypeError, ValueError) as exc:
            raise UnsafeQuery("LIMIT must be a literal integer") from exc
        if requested < 0 or requested > max_rows:
            raise UnsafeQuery(f"LIMIT must be between 0 and {max_rows}")
        row_limit = requested
    return ValidatedQuery(sql=statement.sql(dialect="spark"), tables=tables, row_limit=row_limit)
