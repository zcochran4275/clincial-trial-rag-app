"""The four tools the RAG agent can call, plus the backend that implements them.

`Backend` is a small protocol so tests (and local dev) can swap Databricks for fakes.
Docs:
  Vector search query: https://docs.databricks.com/aws/en/vector-search/query-vector-search
  SQL connector:       https://docs.databricks.com/aws/en/dev-tools/python-sql-connector
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol

from trial_signal.rag.sql_guard import UnsafeSQL, guard

CATALOG = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "trial_signal")
INDEX_NAME = os.getenv("TRIAL_INDEX_NAME", f"{CATALOG}.{SCHEMA}.trial_chunks_index")
SEARCH_COLUMNS = ["chunk_id", "nct_id", "drugs", "phase", "section", "criteria_type", "text"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_trial_text",
        "description": (
            "Hybrid keyword + semantic search over trial free text: summaries, descriptions, "
            "eligibility criteria (one chunk per bullet), outcome measure definitions, arm descriptions. "
            "Use for questions about how trials were designed or who could enroll. "
            "Do NOT use for counts or rates; use query_sql for numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "drug": {"type": "string", "description": "generic name, e.g. 'tucatinib'"},
                "section": {"type": "string", "enum": ["summary", "description", "eligibility", "outcome", "arm"]},
                "criteria_type": {"type": "string", "enum": ["inclusion", "exclusion"]},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_sql",
        "description": (
            "Run one read-only SELECT on the trial_signal schema (Databricks SQL). Tables: "
            "silver_trials(nct_id, brief_title, phases, enrollment, start_date, completion_date, interventions), "
            "silver_arm_adverse_events(nct_id, group_id, group_title, seriousness, term, organ_system, num_affected, num_at_risk), "
            "gold_term_trial_rates(term_key, organ_system, n_trials, n_arms, total_affected, total_at_risk, pooled_rate). "
            f"Always fully qualify tables as {CATALOG}.{SCHEMA}.<table>. Results capped at 200 rows."
        ),
        "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    },
    {
        "name": "get_trial",
        "description": "Core structured fields for one trial by NCT ID.",
        "input_schema": {
            "type": "object",
            "properties": {"nct_id": {"type": "string", "pattern": "^NCT\\d{8}$"}},
            "required": ["nct_id"],
        },
    },
    {
        "name": "faers_signal",
        "description": "Post-market FAERS disproportionality (PRR, 95% CI, chi-square, signal flag) for one drug and one MedDRA preferred term.",
        "input_schema": {
            "type": "object",
            "properties": {"drug": {"type": "string"}, "term": {"type": "string"}},
            "required": ["drug", "term"],
        },
    },
]


class Backend(Protocol):
    def vector_search(self, query: str, filters: dict[str, Any], k: int) -> dict[str, Any]: ...
    def sql(self, statement: str) -> list[dict[str, Any]]: ...


def parse_search_result(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Vector search returns {'manifest': {'columns': [{'name':..}, ...]}, 'result': {'data_array': [[...], ...]}}.
    The last value of each row is the relevance score."""
    cols = [c["name"] for c in raw.get("manifest", {}).get("columns", [])]
    rows = raw.get("result", {}).get("data_array", []) or []
    out = []
    for row in rows:
        rec = dict(zip(cols, row))
        if len(row) > len(cols):  # score appended without a manifest entry
            rec["score"] = row[-1]
        out.append(rec)
    return out


def run_tool(name: str, args: dict[str, Any], backend: Backend) -> str:
    """Execute a tool call and return text for the model. Errors go back to the model, not up the stack."""
    try:
        if name == "search_trial_text":
            filters = {k: args[k] for k in ("section", "criteria_type") if args.get(k)}
            if args.get("drug"):
                # LIKE matches whole whitespace tokens; multi-word generics filter on their first token
                filters["drugs LIKE"] = args["drug"].lower().split()[0]
            hits = parse_search_result(backend.vector_search(args["query"], filters, args.get("k", 8)))
            return json.dumps([{k: h.get(k) for k in ("chunk_id", "nct_id", "section", "text", "score")} for h in hits])
        if name == "query_sql":
            return json.dumps(backend.sql(guard(args["sql"], SCHEMA)), default=str)
        if name == "get_trial":
            nct = args["nct_id"]
            if not nct.startswith("NCT") or not nct[3:].isdigit() or len(nct) != 11:
                raise ValueError("bad NCT ID")
            rows = backend.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.silver_trials WHERE nct_id = '{nct}' LIMIT 1")
            return json.dumps(rows[0] if rows else {"error": "not found"}, default=str)
        if name == "faers_signal":
            drug = args["drug"].lower().replace("'", "")
            term = args["term"].upper().replace("'", "")
            rows = backend.sql(
                f"SELECT * FROM {CATALOG}.{SCHEMA}.gold_faers_prr WHERE drug = '{drug}' AND term_key = '{term}' LIMIT 1"
            )
            return json.dumps(rows[0] if rows else {"error": "no FAERS row for that drug/term"}, default=str)
        return json.dumps({"error": f"unknown tool {name}"})
    except UnsafeSQL as e:
        return json.dumps({"error": f"query rejected: {e}"})
    except Exception as e:  # noqa: BLE001 — surface to the model so it can recover
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


class DatabricksBackend:
    """Real backend. Inside a Databricks App, auth comes from the app's service principal
    (DATABRICKS_HOST / DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET are set automatically).
    Locally, the SDK falls back to DATABRICKS_HOST + DATABRICKS_TOKEN from .env."""

    def __init__(self, warehouse_id: str | None = None):
        from databricks.ai_search.client import AISearchClient
        from databricks.sdk.core import Config

        self._cfg = Config()
        self._index = AISearchClient().get_index(index_name=INDEX_NAME)
        self._http_path = f"/sql/1.0/warehouses/{warehouse_id or os.environ['DATABRICKS_WAREHOUSE_ID']}"

    def vector_search(self, query: str, filters: dict[str, Any], k: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(query_text=query, columns=SEARCH_COLUMNS, num_results=k, query_type="hybrid")
        if filters:
            kwargs["filters"] = filters
        return self._index.similarity_search(**kwargs)

    def sql(self, statement: str) -> list[dict[str, Any]]:
        from databricks import sql

        with sql.connect(
            server_hostname=self._cfg.host.replace("https://", ""),
            http_path=self._http_path,
            credentials_provider=lambda: self._cfg.authenticate,
        ) as conn, conn.cursor() as cur:
            cur.execute(statement)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
