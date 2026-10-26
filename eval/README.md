# Evaluation harness (Phase 4)

1. Pick 20–30 trials spanning phases, drugs, and messy eligibility text.
2. Hand-label the agent's target fields into `ground_truth.jsonl`, one object per trial:
   `{"nct_id": "...", "her2_definition": "...", "requires_prior_trastuzumab": true, ...}`
3. Run the agent N times per task (LLM output varies; report mean and spread).
4. Record per run, in `runs/<timestamp>.jsonl`: task, model, turns, tests_passed_first_try,
   pydantic_valid, field-level matches vs ground truth, tokens/cost, wall time.
5. Report: field-level precision/recall, % runs passing without human edits, median turns, cost per trial.

Label before you look at agent output, so the agent can't anchor your labels.

## Commands

```powershell
python -m trial_signal.cli labels --n 30 --seed 0   # eval/runs/label_sheet.md + unlabeled rows in ground_truth.jsonl
# label from the sheet only (don't open data/agent/*/outputs.jsonl first), set "labeled": true per trial
python -m trial_signal.cli agent --task eligibility --samples 30      # repeat 3-5x per task; LLM output varies
python -m trial_signal.cli agent --task drug_normalization --samples 30
python eval/score_agent.py                          # precision/recall/accuracy (mean +- sd over runs), cost, reliability
```

Use the same `--n/--seed` for labels and agent runs so the labeled trials are the ones the agent saw.
Metric definitions are in `src/trial_signal/scoring.py`. Prices default to gpt-5 list prices; set
`OPENAI_PRICE_IN_PER_M` / `OPENAI_PRICE_OUT_PER_M` if you change model.
