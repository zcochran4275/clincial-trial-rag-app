"""Score the RAG agent against eval/rag_questions.jsonl.

    python eval/run_rag_eval.py            # writes eval/runs/rag_<timestamp>.jsonl + prints a summary

Metrics per question:
  retrieval_recall  share of gold chunk IDs that appeared in any search result
  citation_recall   share of gold NCT IDs the answer cited
  citation_precision share of cited NCT IDs that are in the gold set (catches made-up citations)
  declined          for type=unanswerable, did the agent say the data doesn't cover it?
Answer correctness is graded by hand (or by a second model) in the output file: add "correct": true/false.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from trial_signal.rag.agent import answer
from trial_signal.rag.tools import DatabricksBackend

HERE = Path(__file__).parent
DECLINE = re.compile(r"\b(not (in|available|reported)|doesn't (say|include|cover)|does not (say|include|cover)|no data|cannot (determine|answer))\b", re.I)


def _ratio(hit: set, gold: set) -> float | None:
    return len(hit & gold) / len(gold) if gold else None


def main() -> None:
    questions = [json.loads(line) for line in (HERE / "rag_questions.jsonl").read_text().splitlines() if line.strip()]
    backend = DatabricksBackend()
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"rag_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl"
    rows = []
    for q in questions:
        t = answer(q["question"], backend)
        cited, gold_ids = set(t.cited_nct_ids), set(q["gold_nct_ids"])
        row = {
            **q,
            "answer": t.answer,
            "tools": [c["name"] for c in t.tool_calls],
            "turns": t.turns,
            "tokens": t.input_tokens + t.output_tokens,
            "retrieval_recall": _ratio(set(t.retrieved_chunks), set(q["gold_chunk_ids"])),
            "citation_recall": _ratio(cited, gold_ids),
            "citation_precision": (len(cited & gold_ids) / len(cited)) if cited and gold_ids else None,
            "declined": bool(DECLINE.search(t.answer)) if q["type"] == "unanswerable" else None,
        }
        rows.append(row)
        print(f"{q['id']} {q['type']:<12} tools={row['tools']} recall={row['retrieval_recall']}")
    out.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
