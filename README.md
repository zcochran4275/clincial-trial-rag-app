# trial-signal

**Do the side effects reported in HER2-targeted cancer drug trials match what patients report after approval?**

An agentic data pipeline that pulls completed HER2-therapy trials from ClinicalTrials.gov, uses
an LLM (OpenAI) to extract structured fields from free text, and compares trial-reported adverse events with
post-market FDA adverse event reports (FAERS) using proportional reporting ratios.

```
ClinicalTrials.gov v2 ─┐                                   ┌─ silver_trials
                       ├─► bronze (raw JSON + provenance) ─┤─ silver_arm_adverse_events ─► dbt gold ─► dashboard
openFDA FAERS ─────────┘        Parquet                    └─ agent-extracted fields
                                                               (OpenAI + Docker sandbox + pytest)
        └──────────────────── orchestrated by Prefect ──────── stored on Databricks (Unity Catalog) ┘
```

## Stack
Python · OpenAI API (function calling) · Pydantic · pytest · Docker · Prefect · Parquet · Databricks · dbt

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add your keys
pytest                          # offline: runs on fixtures only
python -m trial_signal.cli fetch --drug tucatinib --max-pages 1
python -m trial_signal.cli silver
python flows/pipeline.py        # full flow via Prefect
```

## Layout
| Path | What it is | Phase |
|---|---|---|
| `src/trial_signal/ingest/` | ClinicalTrials.gov + openFDA clients | 1 |
| `src/trial_signal/bronze.py` | Raw landing with fetch time, query, content hash | 1 |
| `src/trial_signal/schemas.py` | Pydantic contracts for silver | 2 |
| `src/trial_signal/silver/flatten.py` | Deterministic baseline (structured fields only) | 2 |
| `src/trial_signal/agent/` | Tool definitions, sandbox runner, agent loop (skeleton) | 3 |
| `eval/` | Hand-labeled ground truth + eval runs | 4 |
| `src/trial_signal/analysis/prr.py` | Proportional reporting ratio + signal rule | 5 |
| `flows/pipeline.py` | Prefect flow tying it together | 5 |
| `databricks/`, `dbt/` | Lakehouse setup and gold models | 5 |
| `src/trial_signal/rag/` | Chunker, SQL guard, agent tools, RAG agent | 7 |
| `databricks/02_vector_index.py` | Chunks table (CDF) → AI Search endpoint → Delta Sync index | 7 |
| `app/` | Streamlit chat, deployed as a Databricks App | 7 |
| `eval/run_rag_eval.py` | Retrieval recall, citation precision/recall, decline rate | 7 |

## Roadmap
- [ ] **Phase 1** – Diff one live API record against `tests/fixtures/`; fix any drift. Land all 7 drugs in bronze.
- [ ] **Phase 2** – Build silver; inspect `rejects.parquet`; add tests for every new reject reason you find.
- [ ] **Phase 3** – Finish `agent/loop.py` (error handling, logging each turn). Tasks: eligibility parsing, drug-name normalization.
- [ ] **Phase 4** – Hand-label 20–30 trials in `eval/ground_truth.jsonl`; report field-level precision/recall, pass-without-edit rate, turns and cost per trial.
- [ ] **Phase 5** – FAERS PRR table, Databricks upload, `dbt build`, Streamlit dashboard.
- [ ] **Phase 6** – Write-up: results, limitations, security design.
- [ ] **Phase 7** – RAG agent: `cli chunks` → upload → `02_vector_index.py` → `cli ask "..."` → deploy `app/` → fill `eval/rag_questions.jsonl` to 40–50 questions and run `eval/run_rag_eval.py`.

## Limitations (keep this section honest)
- FAERS counts are **reports**, not patients; reporting is voluntary and biased toward serious and newsworthy events. PRR flags hypotheses, not causation.
- Trial AE tables apply frequency thresholds (often 5%), so rare events are missing from "other" events.
- MedDRA versions differ across trials; the full MedDRA hierarchy is licensed, so this project works at the preferred-term level only.

## Security design for the agent
No network and no credentials in the sandbox · read-only data mounts · resource and time limits ·
trial text treated as untrusted data (prompt-injection aware) · outputs accepted only if Pydantic validation and tests pass.
