"""Phase 4: score the extraction agent against hand-labeled ground truth.

    python -m trial_signal.cli labels --n 30 --seed 0     # label sheet + ground_truth.jsonl skeleton
    python eval/score_agent.py                            # report over every agent run in eval/runs/

Ground truth is eval/ground_truth.jsonl, one object per trial, flipped to "labeled": true when done.
Eligibility fields sit at the top level; drug labels are an "interventions" list:

    {"nct_id": "NCT00000001", "labeled": true, "her2_definition": "IHC 3+ or FISH+",
     "requires_prior_trastuzumab": false, "allows_brain_mets": null, "min_age": 18,
     "interventions": [{"name": "Tukysa", "generic": "tucatinib"}, {"name": "Placebo", "generic": null}]}

Per field: precision = correct values / values the agent gave; recall = correct values / values in
the labels; accuracy counts null == null as correct. A trial the agent produced nothing for (its
output was rejected) scores as all-null predictions, so failures cost recall rather than vanishing.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trial_signal.agent.loop import pick_studies
from trial_signal.config import DATA_DIR, HER2_DRUGS, ROOT
from trial_signal.schemas import DrugNormalization, EligibilityExtraction

GROUND_TRUTH = ROOT / "eval" / "ground_truth.jsonl"
RUNS_DIR = ROOT / "eval" / "runs"
ELIGIBILITY_FIELDS = tuple(EligibilityExtraction.model_fields)
TEXT_MATCH_F1 = 0.6  # her2_definition is free text: count it correct at this token overlap with the label

# USD per 1M tokens for the default model; override if you change OPENAI_MODEL or prices move.
PRICE_IN_PER_M = float(os.getenv("OPENAI_PRICE_IN_PER_M", "1.25"))
PRICE_OUT_PER_M = float(os.getenv("OPENAI_PRICE_OUT_PER_M", "10.0"))


# --- ground truth ---------------------------------------------------------------------------


def load_ground_truth(path: Path = GROUND_TRUTH) -> dict[str, dict[str, Any]]:
    """Labeled rows by NCT ID. Rows are validated against the same contracts the agent is held to,
    and every field must be present so an unlabeled null can't pass as "the record says nothing"."""
    gold: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return gold
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("labeled"):
            continue
        nct = row["nct_id"]
        missing = [f for f in (*ELIGIBILITY_FIELDS, "interventions") if f not in row]
        if missing:
            raise ValueError(f"{nct}: labeled row is missing {missing}")
        try:
            EligibilityExtraction.model_validate({f: row[f] for f in ELIGIBILITY_FIELDS})
            DrugNormalization.model_validate({"interventions": row["interventions"]})
        except ValueError as e:
            raise ValueError(f"{nct}: invalid label: {e}") from e
        gold[nct] = row
    return gold


def build_label_sheet(n: int = 30, seed: int = 0, out_dir: Path = RUNS_DIR, gt_path: Path = GROUND_TRUTH) -> tuple[Path, int]:
    """Write a readable sheet of the trials to label (no agent output, so labels aren't anchored) and
    append unlabeled skeleton rows to ground_truth.jsonl. Existing rows are never touched."""
    existing = set()
    if gt_path.exists():
        existing = {json.loads(l)["nct_id"] for l in gt_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    sheet = [
        "# Label sheet\n",
        "Edit eval/ground_truth.jsonl. For each trial below set the fields from the text alone, then `\"labeled\": true`.",
        "- her2_definition: how the trial defines HER2-positive (short quote/paraphrase), else null",
        "- requires_prior_trastuzumab / allows_brain_mets: true / false, or null if the record doesn't say",
        "- min_age: whole years, else null",
        f"- interventions[].generic: one of {HER2_DRUGS}, or null (placebo, chemo, other drugs)\n",
    ]
    skeleton = []
    for study in pick_studies(n, seed):
        ps = study["protocolSection"]
        nct = ps["identificationModule"]["nctId"]
        elig = ps.get("eligibilityModule", {})
        names = [i.get("name", "") for i in ps.get("armsInterventionsModule", {}).get("interventions", [])]
        sheet += [
            f"\n---\n## {nct}: {ps['identificationModule'].get('briefTitle', '')}",
            f"**minimumAge:** {elig.get('minimumAge')}",
            f"**Interventions:** {names}",
            f"\n```\n{elig.get('eligibilityCriteria', '(no eligibility text)')}\n```",
        ]
        if nct not in existing:
            skeleton.append({
                "nct_id": nct, "labeled": False,
                **{f: None for f in ELIGIBILITY_FIELDS},
                "interventions": [{"name": name, "generic": None} for name in names],
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "label_sheet.md"
    sheet_path.write_text("\n".join(sheet), encoding="utf-8")
    if skeleton:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with gt_path.open("a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in skeleton)
    return sheet_path, len(skeleton)


# --- matching and per-run scores ------------------------------------------------------------


def _norm(v: Any) -> Any:
    """pandas-written runs hold ints as 18.0 and nulls as NaN; compare on plain values."""
    if isinstance(v, float):
        return int(v) if v == v and v.is_integer() else (None if v != v else v)
    return v


def token_f1(a: str, b: str) -> float:
    ta, tb = Counter(re.findall(r"[a-z0-9+]+", a.lower())), Counter(re.findall(r"[a-z0-9+]+", b.lower()))
    overlap = sum((ta & tb).values())
    if not overlap:
        return 0.0
    p, r = overlap / sum(ta.values()), overlap / sum(tb.values())
    return 2 * p * r / (p + r)


def _text_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return pred is gold
    return token_f1(pred, gold) >= TEXT_MATCH_F1


def _exact(pred: Any, gold: Any) -> bool:
    return pred == gold


def _tally(pairs: list[tuple[Any, Any]], match: Callable[[Any, Any], bool]) -> dict[str, Any]:
    pred_n = gold_n = tp = correct = 0
    for p, g in pairs:
        ok = match(p, g)
        correct += ok
        pred_n += p is not None
        gold_n += g is not None
        tp += ok and p is not None and g is not None
    return {
        "n": len(pairs),
        "accuracy": correct / len(pairs) if pairs else None,
        "precision": tp / pred_n if pred_n else None,
        "recall": tp / gold_n if gold_n else None,
    }


def score_eligibility(rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["nct_id"]: r for r in rows}
    fields = {}
    for f in ELIGIBILITY_FIELDS:
        pairs = [(_norm(by_id.get(nct, {}).get(f)), g[f]) for nct, g in gold.items()]
        fields[f] = _tally(pairs, _text_match if f == "her2_definition" else _exact)
    return {"fields": fields, "n_trials": len(gold), "n_missing": sum(nct not in by_id for nct in gold)}


def score_drugs(rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["nct_id"]: r for r in rows}
    pairs = []
    for nct, g in gold.items():
        pred = {m["name"].casefold(): m["generic"] for m in by_id.get(nct, {}).get("interventions", [])}
        pairs += [(pred.get(m["name"].casefold()), m["generic"]) for m in g["interventions"]]
    return {"fields": {"generic": _tally(pairs, _exact)}, "n_trials": len(gold), "n_missing": sum(nct not in by_id for nct in gold)}


def run_cost_usd(run: dict[str, Any]) -> float:
    return (run["input_tokens"] * PRICE_IN_PER_M + run["output_tokens"] * PRICE_OUT_PER_M) / 1e6


# --- report over many runs ------------------------------------------------------------------


def _load_runs(task: str, runs_dir: Path) -> list[dict[str, Any]]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("agent_*.json"))]
    return [r for r in runs if r["task"] == task]


def _spread(values: list[float | None]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sd": None}
    return {"mean": statistics.fmean(vals), "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0}


def summarize_task(
    task: str, gold: dict[str, dict[str, Any]], runs_dir: Path = RUNS_DIR, agent_dir: Path = DATA_DIR / "agent"
) -> dict[str, Any] | None:
    runs = _load_runs(task, runs_dir)
    if not runs:
        return None
    scorer = score_eligibility if task == "eligibility" else score_drugs
    per_run = []
    for run in runs:
        out = agent_dir / run["run_id"] / "outputs.jsonl"
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()] if out.exists() else []
        per_run.append(scorer(rows, gold) if gold else None)
    scored = [s for s in per_run if s]
    fields = {}
    for f in (scored[0]["fields"] if scored else {}):
        fields[f] = {m: _spread([s["fields"][f][m] for s in scored]) for m in ("precision", "recall", "accuracy")}
    n = len(runs)
    return {
        "task": task,
        "runs": n,
        "labeled_trials": len(gold),
        "fields": fields,
        "pass_without_edit": sum(bool(r.get("submitted")) and r.get("n_rejected", 1) == 0 for r in runs) / n,
        "tests_passed_first_try": sum(bool(r.get("tests_passed_first_try")) for r in runs) / n,
        "median_turns": statistics.median(r["turns"] for r in runs),
        "cost_usd_per_run": statistics.fmean(run_cost_usd(r) for r in runs),
        "cost_usd_per_trial": statistics.fmean(run_cost_usd(r) / r["n_samples"] for r in runs),
        "median_wall_s": statistics.median(r["wall_s"] for r in runs),
    }


def _pct(m: dict[str, float | None]) -> str:
    return "n/a" if m["mean"] is None else f"{m['mean']:.0%}" + (f" ±{m['sd']:.0%}" if m["sd"] else "")


def render(summary: dict[str, Any]) -> str:
    s = summary
    lines = [
        f"## {s['task']}: {s['runs']} run(s), {s['labeled_trials']} labeled trials",
        f"pass without edit: {s['pass_without_edit']:.0%} | tests green first try: {s['tests_passed_first_try']:.0%} | "
        f"median turns: {s['median_turns']:g} | median wall: {s['median_wall_s']:g}s",
        f"cost: ${s['cost_usd_per_run']:.2f}/run, ${s['cost_usd_per_trial']:.3f}/trial "
        f"(at ${PRICE_IN_PER_M:g}/${PRICE_OUT_PER_M:g} per 1M tokens in/out)",
    ]
    if s["fields"]:
        lines += ["", f"{'field':<30}{'precision':>14}{'recall':>14}{'accuracy':>14}"]
        lines += [f"{f:<30}{_pct(m['precision']):>14}{_pct(m['recall']):>14}{_pct(m['accuracy']):>14}" for f, m in s["fields"].items()]
    else:
        lines.append("\n(no labeled trials yet: set \"labeled\": true in eval/ground_truth.jsonl)")
    return "\n".join(lines)
