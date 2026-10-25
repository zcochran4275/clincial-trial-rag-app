"""Prefect flow: bronze -> silver -> (agent, Phase 3) -> Databricks upload -> dbt.

Run locally:   python flows/pipeline.py
Prefect UI:    prefect server start   (then open http://127.0.0.1:4200)
Docs:          https://docs.prefect.io/v3/get-started/quickstart
"""
from __future__ import annotations

import json

import pandas as pd
from prefect import flow, get_run_logger, task

from trial_signal.bronze import to_bronze_frame, write_bronze
from trial_signal.config import BRONZE_DIR, HER2_DRUGS, SILVER_DIR
from trial_signal.ingest import ctgov, openfda
from trial_signal.silver.flatten import flatten_adverse_events, flatten_trial


@task(retries=3, retry_delay_seconds=[5, 30, 120])
def fetch_ctgov(drug: str, max_pages: int | None) -> str:
    studies = list(ctgov.iter_studies(drug, max_pages=max_pages))
    df = to_bronze_frame(studies, "ctgov", drug, ctgov.nct_id)
    return str(write_bronze(df, BRONZE_DIR / "ctgov", drug.replace(" ", "_")))


@task(retries=3, retry_delay_seconds=[5, 30, 120])
def fetch_faers_counts(drug: str) -> str:
    rows = openfda.count_reactions_for_drug(drug)
    payload = [{"drug": drug, **r} for r in rows]
    df = to_bronze_frame(payload, "openfda", drug, lambda r: f"{r['drug']}|{r['term']}")
    return str(write_bronze(df, BRONZE_DIR / "openfda", drug.replace(" ", "_")))


@task
def build_silver(bronze_paths: list[str]) -> dict[str, int]:
    log = get_run_logger()
    bronze = pd.concat([pd.read_parquet(p) for p in bronze_paths]).drop_duplicates("record_key")
    trials, events, rejects = [], [], []
    for raw in bronze["raw_json"]:
        study = json.loads(raw)
        trials.append(flatten_trial(study).model_dump())
        rows, bad = flatten_adverse_events(study)
        events.extend(r.model_dump() for r in rows)
        rejects.extend(bad)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_parquet(SILVER_DIR / "trials.parquet", index=False)
    pd.DataFrame(events).to_parquet(SILVER_DIR / "arm_adverse_events.parquet", index=False)
    pd.DataFrame(rejects).to_parquet(SILVER_DIR / "rejects.parquet", index=False)
    stats = {"trials": len(trials), "ae_rows": len(events), "rejects": len(rejects)}
    log.info("silver built: %s", stats)
    if events and len(rejects) / (len(events) + len(rejects)) > 0.05:
        log.warning("reject rate above 5%% — check rejects.parquet before trusting gold tables")
    return stats


@flow(name="trial-signal")
def pipeline(drugs: list[str] = HER2_DRUGS, max_pages: int | None = None) -> dict[str, int]:
    ct_paths = [fetch_ctgov.submit(d, max_pages) for d in drugs]
    _faers = [fetch_faers_counts.submit(d) for d in drugs]
    stats = build_silver([p.result() for p in ct_paths])
    # Phase 3: agent_extract(...)   — free-text eligibility + outcome measures
    # Phase 5: upload_to_databricks(...) then run `dbt build`
    [f.result() for f in _faers]
    return stats


if __name__ == "__main__":
    pipeline(max_pages=1)
