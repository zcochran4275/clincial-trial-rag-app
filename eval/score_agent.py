"""Score the Phase 3 extraction agent against eval/ground_truth.jsonl.

    python eval/score_agent.py                       # every task with runs in eval/runs/
    python eval/score_agent.py --task eligibility

Run the agent several times first (LLM output varies): `python -m trial_signal.cli agent --task ...`.
See src/trial_signal/scoring.py for the metric definitions.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from trial_signal.agent.loop import TASKS
from trial_signal.scoring import RUNS_DIR, load_ground_truth, render, summarize_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(TASKS))
    args = p.parse_args()
    gold = load_ground_truth()
    summaries = [s for t in ([args.task] if args.task else sorted(TASKS)) if (s := summarize_task(t, gold))]
    if not summaries:
        print("no agent runs found in eval/runs/")
        return
    print("\n\n".join(render(s) for s in summaries))
    out = RUNS_DIR / f"score_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
