"""Second line of defense for the agent's query_sql tool.

The first line is permissions: the Databricks App's service principal is granted SELECT on
the trial_signal schema and nothing else. This guard rejects obviously unsafe SQL before it
is ever sent, and caps result size so the model never receives a huge table.
"""
from __future__ import annotations

import re

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|call|copy|"
    r"optimize|vacuum|restore|refresh|msck|set|use|put|get|remove|attach|install|load)\b",
    re.IGNORECASE,
)
MAX_ROWS = 200


class UnsafeSQL(ValueError):
    pass


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", sql)


def _strip_literals(sql: str) -> str:
    """Blank out string literals so a term like 'Drug: create' doesn't trip the keyword check."""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def guard(sql: str, schema: str = "trial_signal", max_rows: int = MAX_ROWS) -> str:
    """Return a safe, row-capped version of sql or raise UnsafeSQL."""
    s = _strip_comments(sql).strip().rstrip(";").strip()
    if not s:
        raise UnsafeSQL("empty query")
    bare = _strip_literals(s)
    if ";" in bare:
        raise UnsafeSQL("only one statement allowed")
    if not re.match(r"^(select|with)\b", bare, re.IGNORECASE):
        raise UnsafeSQL("only SELECT / WITH queries allowed")
    if m := FORBIDDEN.search(bare):
        raise UnsafeSQL(f"keyword not allowed: {m.group(0)}")
    # Every table reference must live in the project schema.
    for ref in re.findall(r"\b(?:from|join)\s+([`\w.]+)", bare, re.IGNORECASE):
        if "(" in ref:
            continue
        parts = ref.replace("`", "").split(".")
        if len(parts) >= 2 and parts[-2] != schema:
            raise UnsafeSQL(f"table outside schema '{schema}': {ref}")
    if not re.search(r"\blimit\s+\d+\s*$", bare, re.IGNORECASE):
        s = f"{s}\nLIMIT {max_rows}"
    return s
