"""ClinicalTrials.gov API v2 client.

Docs: https://clinicaltrials.gov/data-api/api
We pull completed interventional studies that have posted results, one drug at a time,
following `nextPageToken` until the result set is exhausted.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from trial_signal.config import CTGOV_BASE, USER_AGENT

DEFAULT_PAGE_SIZE = 100  # API allows up to 1000; smaller pages = gentler + easier to debug


def build_params(drug: str, page_size: int = DEFAULT_PAGE_SIZE, page_token: str | None = None) -> dict[str, Any]:
    """Query parameters for one page of /studies. Kept pure so it is easy to unit-test."""
    params: dict[str, Any] = {
        "query.intr": drug,
        "filter.overallStatus": "COMPLETED",
        "aggFilters": "results:with",
        "pageSize": page_size,
        "countTotal": "true",
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token
    return params


def iter_studies(
    drug: str,
    session: requests.Session | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw study records (unmodified JSON dicts) for one intervention name."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", USER_AGENT)
    token: str | None = None
    pages = 0
    while True:
        resp = s.get(f"{CTGOV_BASE}/studies", params=build_params(drug, page_size, token), timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        yield from payload.get("studies", [])
        pages += 1
        token = payload.get("nextPageToken")
        if not token or (max_pages is not None and pages >= max_pages):
            break


def nct_id(study: dict[str, Any]) -> str:
    return study["protocolSection"]["identificationModule"]["nctId"]
