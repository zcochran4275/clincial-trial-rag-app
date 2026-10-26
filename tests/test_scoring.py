import json

import pytest

from trial_signal import scoring
from trial_signal.scoring import (
    load_ground_truth,
    run_cost_usd,
    score_drugs,
    score_eligibility,
    summarize_task,
    token_f1,
)


def _gold(nct, **kw):
    row = {"nct_id": nct, "labeled": True, "her2_definition": None, "requires_prior_trastuzumab": None,
           "allows_brain_mets": None, "min_age": None, "interventions": []}
    return row | kw


def _write_gold(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_token_f1():
    assert token_f1("IHC 3+ or FISH positive", "ihc 3+ or fish positive") == 1.0
    assert token_f1("a b", "c d") == 0.0
    assert 0.5 < token_f1("HER2 positive by IHC 3+", "HER2 positive IHC 3+ or FISH amplified") < 1.0


def test_eligibility_precision_recall_accuracy_by_hand():
    gold = {
        "NCT00000001": _gold("NCT00000001", allows_brain_mets=True, min_age=18),
        "NCT00000002": _gold("NCT00000002", allows_brain_mets=False, min_age=18),
        "NCT00000003": _gold("NCT00000003", allows_brain_mets=None, min_age=21),
        "NCT00000004": _gold("NCT00000004", allows_brain_mets=True, min_age=18),  # agent output missing
    }
    rows = [
        {"nct_id": "NCT00000001", "allows_brain_mets": True, "min_age": 18},      # right, right
        {"nct_id": "NCT00000002", "allows_brain_mets": True, "min_age": 18.0},    # wrong, right (float from pandas)
        {"nct_id": "NCT00000003", "allows_brain_mets": True, "min_age": None},    # spurious, missed
    ]
    r = score_eligibility(rows, gold)
    assert r["n_trials"] == 4 and r["n_missing"] == 1
    brain = r["fields"]["allows_brain_mets"]
    # predicted: 3 non-null, 1 correct; labeled: 3 non-null; accuracy: 1 of 4 (trial 3 null != True)
    assert (brain["precision"], brain["recall"], brain["accuracy"]) == (pytest.approx(1 / 3), pytest.approx(1 / 3), 0.25)
    age = r["fields"]["min_age"]
    assert (age["precision"], age["recall"], age["accuracy"]) == (1.0, 0.5, 0.5)


def test_free_text_field_uses_token_overlap_and_null_handling():
    gold = {
        "NCT00000001": _gold("NCT00000001", her2_definition="HER2 positive: IHC 3+ or FISH amplified"),
        "NCT00000002": _gold("NCT00000002", her2_definition=None),
    }
    rows = [
        {"nct_id": "NCT00000001", "her2_definition": "IHC 3+ or FISH amplified HER2 positive"},
        {"nct_id": "NCT00000002", "her2_definition": None},
    ]
    f = score_eligibility(rows, gold)["fields"]["her2_definition"]
    assert (f["precision"], f["recall"], f["accuracy"]) == (1.0, 1.0, 1.0)
    rows[0]["her2_definition"] = "completely unrelated text"
    f = score_eligibility(rows, gold)["fields"]["her2_definition"]
    assert (f["precision"], f["recall"], f["accuracy"]) == (0.0, 0.0, 0.5)


def test_drugs_matched_by_name_case_insensitively_and_missing_trial_counts_against_recall():
    gold = {
        "NCT00000001": _gold("NCT00000001", interventions=[
            {"name": "Kadcyla", "generic": "ado-trastuzumab emtansine"}, {"name": "Placebo", "generic": None}]),
        "NCT00000002": _gold("NCT00000002", interventions=[{"name": "Tukysa", "generic": "tucatinib"}]),
    }
    rows = [{"nct_id": "NCT00000001", "interventions": [
        {"name": "kadcyla", "generic": "ado-trastuzumab emtansine"}, {"name": "placebo", "generic": "lapatinib"}]}]
    g = score_drugs(rows, gold)["fields"]["generic"]
    # 3 labeled items, 2 non-null labels; agent gave 2 non-null, 1 correct; Tukysa trial produced nothing
    assert g["n"] == 3 and g["precision"] == 0.5 and g["recall"] == 0.5 and g["accuracy"] == pytest.approx(1 / 3)


def test_ground_truth_skips_unlabeled_and_rejects_invalid_or_incomplete(tmp_path):
    path = tmp_path / "gt.jsonl"
    _write_gold(path, [_gold("NCT00000001", min_age=18), _gold("NCT00000002", labeled=False)])
    assert list(load_ground_truth(path)) == ["NCT00000001"]
    _write_gold(path, [_gold("NCT00000003", min_age=500)])
    with pytest.raises(ValueError, match="NCT00000003"):
        load_ground_truth(path)
    _write_gold(path, [_gold("NCT00000004", interventions=[{"name": "x", "generic": "pembrolizumab"}])])
    with pytest.raises(ValueError, match="NCT00000004"):
        load_ground_truth(path)
    incomplete = _gold("NCT00000005")
    del incomplete["min_age"]
    _write_gold(path, [incomplete])
    with pytest.raises(ValueError, match="missing"):
        load_ground_truth(path)


def test_cost_uses_configurable_prices(monkeypatch):
    monkeypatch.setattr(scoring, "PRICE_IN_PER_M", 2.0)
    monkeypatch.setattr(scoring, "PRICE_OUT_PER_M", 10.0)
    assert run_cost_usd({"input_tokens": 500_000, "output_tokens": 100_000}) == pytest.approx(2.0)


def _run(runs_dir, agent_dir, run_id, outputs, **kw):
    summary = {"run_id": run_id, "task": "eligibility", "n_samples": 2, "submitted": True, "n_rejected": 0,
               "tests_passed_first_try": True, "turns": 6, "wall_s": 60.0, "input_tokens": 100_000, "output_tokens": 10_000} | kw
    (runs_dir / f"agent_{run_id}.json").write_text(json.dumps(summary), encoding="utf-8")
    (agent_dir / run_id).mkdir(parents=True)
    (agent_dir / run_id / "outputs.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outputs), encoding="utf-8")


def test_summarize_task_reports_mean_spread_and_reliability(tmp_path):
    runs_dir, agent_dir = tmp_path / "runs", tmp_path / "agent"
    runs_dir.mkdir()
    gold = {"NCT00000001": _gold("NCT00000001", min_age=18), "NCT00000002": _gold("NCT00000002", min_age=21)}
    ok = [{"nct_id": "NCT00000001", "min_age": 18}, {"nct_id": "NCT00000002", "min_age": 21}]
    half = [{"nct_id": "NCT00000001", "min_age": 18}, {"nct_id": "NCT00000002", "min_age": 99}]
    _run(runs_dir, agent_dir, "a", ok)
    _run(runs_dir, agent_dir, "b", half, tests_passed_first_try=False, n_rejected=2, turns=10)
    s = summarize_task("eligibility", gold, runs_dir, agent_dir)
    age = s["fields"]["min_age"]["recall"]
    assert age["mean"] == 0.75 and age["sd"] == 0.25
    assert s["runs"] == 2 and s["pass_without_edit"] == 0.5 and s["tests_passed_first_try"] == 0.5
    assert s["median_turns"] == 8
    assert s["cost_usd_per_run"] == pytest.approx(0.225) and s["cost_usd_per_trial"] == pytest.approx(0.1125)
    assert summarize_task("drug_normalization", gold, runs_dir, agent_dir) is None
    assert "no labeled trials" in scoring.render(summarize_task("eligibility", {}, runs_dir, agent_dir))
