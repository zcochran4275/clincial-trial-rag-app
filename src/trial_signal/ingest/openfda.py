"""openFDA drug adverse event (FAERS) client.

Docs: https://open.fda.gov/apis/drug/event/
Read the disclaimer page before interpreting anything: FAERS counts are *reports*, not
patients or incidence. That is why the analysis uses disproportionality (PRR), not rates.

The four numbers PRR needs, all obtainable from count/total queries:
  a = reports mentioning drug AND event      -> count_reactions_for_drug()
  drug_total = reports mentioning drug       -> total_reports(drug_query(drug))
  event_total = reports mentioning event     -> total_reports(event_query(term))
  N = all reports in the database            -> total_reports(None)
"""
from __future__ import annotations

from typing import Any

import requests

from trial_signal.config import OPENFDA_API_KEY, OPENFDA_EVENT_URL, USER_AGENT

REACTION_FIELD = "patient.reaction.reactionmeddrapt.exact"
COUNT_LIMIT_MAX = 1000


def drug_query(generic_name: str, suspect_only: bool = True) -> str:
    """Search clause for one drug. suspect_only keeps reports where the drug was the
    suspected cause (drugcharacterization=1), not just a concomitant medication."""
    clause = f'patient.drug.openfda.generic_name:"{generic_name}"'
    if suspect_only:
        clause = f"({clause}+AND+patient.drug.drugcharacterization:1)"
    return clause


def event_query(meddra_pt: str) -> str:
    return f'{REACTION_FIELD}:"{meddra_pt.upper()}"'


def _get(params: dict[str, Any], session: requests.Session | None) -> dict[str, Any]:
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    if OPENFDA_API_KEY:
        params = {**params, "api_key": OPENFDA_API_KEY}
    # openFDA expects literal '+' in search strings; build the query string by hand.
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    resp = s.get(f"{OPENFDA_EVENT_URL}?{qs}", timeout=30)
    if resp.status_code == 404:  # openFDA returns 404 for "no matches"
        return {"meta": {"results": {"total": 0}}, "results": []}
    resp.raise_for_status()
    return resp.json()


def count_reactions_for_drug(
    generic_name: str, limit: int = COUNT_LIMIT_MAX, session: requests.Session | None = None
) -> list[dict[str, Any]]:
    """[{'term': 'DIARRHOEA', 'count': 1234}, ...] — top reactions for a drug."""
    payload = _get({"search": drug_query(generic_name), "count": REACTION_FIELD, "limit": min(limit, COUNT_LIMIT_MAX)}, session)
    return payload.get("results", [])


def total_reports(search: str | None, session: requests.Session | None = None) -> int:
    params: dict[str, Any] = {"limit": 1}
    if search:
        params["search"] = search
    payload = _get(params, session)
    return int(payload["meta"]["results"]["total"])
