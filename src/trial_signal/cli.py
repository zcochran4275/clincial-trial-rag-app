"""Small CLI for poking at each stage by hand.

    python -m trial_signal.cli fetch --drug tucatinib --max-pages 1
    python -m trial_signal.cli silver
    python -m trial_signal.cli chunks
    python -m trial_signal.cli ask "Which tucatinib trials allowed active brain metastases?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trial_signal.bronze import to_bronze_frame, write_bronze
from trial_signal.config import BRONZE_DIR, HER2_DRUGS, SILVER_DIR, ctgov_term
from trial_signal.ingest import ctgov
from trial_signal.silver.flatten import REJECT_DTYPES, flatten_adverse_events, flatten_trial


def cmd_fetch(args: argparse.Namespace) -> None:
    drugs = [args.drug] if args.drug else HER2_DRUGS
    for drug in drugs:
        term = ctgov_term(drug)
        studies = list(ctgov.iter_studies(term, max_pages=args.max_pages))
        df = to_bronze_frame(studies, "ctgov", term, ctgov.nct_id)
        path = write_bronze(df, BRONZE_DIR / "ctgov", drug.replace(" ", "_"))
        print(f"{drug}: {len(df)} studies -> {path}")


def cmd_silver(args: argparse.Namespace) -> None:
    files = sorted((BRONZE_DIR / "ctgov").glob("*.parquet"))
    bronze = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("record_key", keep="last")
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
    rejects_df = pd.DataFrame(rejects, columns=list(REJECT_DTYPES)).astype(REJECT_DTYPES)
    rejects_df.to_parquet(SILVER_DIR / "rejects.parquet", index=False)
    print(f"trials={len(trials)} ae_rows={len(events)} rejects={len(rejects)}")


def cmd_chunks(args: argparse.Namespace) -> None:
    from trial_signal.rag.chunking import chunk_trial

    files = sorted((BRONZE_DIR / "ctgov").glob("*.parquet"))
    bronze = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("record_key", keep="last")
    chunks = [c.model_dump() for raw in bronze["raw_json"] for c in chunk_trial(json.loads(raw))]
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(chunks)
    df.to_parquet(SILVER_DIR / "trial_chunks.parquet", index=False)
    print(f"chunks={len(df)} trials={df['nct_id'].nunique() if len(df) else 0}")
    if len(df):
        print(df["section"].value_counts().to_string())


def cmd_ask(args: argparse.Namespace) -> None:
    from trial_signal.rag.agent import answer
    from trial_signal.rag.tools import DatabricksBackend

    trace = answer(args.question, DatabricksBackend())
    print(trace.answer)
    print(f"\n[tools: {[c['name'] for c in trace.tool_calls]} | turns: {trace.turns} | cited: {trace.cited_nct_ids}]")


def main() -> None:
    p = argparse.ArgumentParser(prog="trial_signal")
    sub = p.add_subparsers(required=True)
    f = sub.add_parser("fetch", help="land raw ClinicalTrials.gov studies in bronze")
    f.add_argument("--drug")
    f.add_argument("--max-pages", type=int, default=None)
    f.set_defaults(func=cmd_fetch)
    s = sub.add_parser("silver", help="flatten bronze into validated silver tables")
    s.set_defaults(func=cmd_silver)
    c = sub.add_parser("chunks", help="build trial_chunks.parquet for vector search")
    c.set_defaults(func=cmd_chunks)
    a = sub.add_parser("ask", help="ask the RAG agent a question (needs Databricks + OpenAI keys)")
    a.add_argument("question")
    a.set_defaults(func=cmd_ask)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
