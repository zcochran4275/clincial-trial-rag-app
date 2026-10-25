# Evaluation harness (Phase 4)

1. Pick 20–30 trials spanning phases, drugs, and messy eligibility text.
2. Hand-label the agent's target fields into `ground_truth.jsonl`, one object per trial:
   `{"nct_id": "...", "her2_definition": "...", "requires_prior_trastuzumab": true, ...}`
3. Run the agent N times per task (LLM output varies; report mean and spread).
4. Record per run, in `runs/<timestamp>.jsonl`: task, model, turns, tests_passed_first_try,
   pydantic_valid, field-level matches vs ground truth, tokens/cost, wall time.
5. Report: field-level precision/recall, % runs passing without human edits, median turns, cost per trial.

Label before you look at agent output, so the agent can't anchor your labels.
