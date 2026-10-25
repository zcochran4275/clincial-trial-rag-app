"""Turn a raw ClinicalTrials.gov study into section-aware text chunks for vector search.

Rules:
  * Free text only. Numbers people will aggregate (AE counts, enrollment) stay in SQL.
  * Eligibility criteria are split per bullet, so "No prior lapatinib" is one retrievable fact.
  * Every chunk gets a short header (NCT ID, drug, phase, section) prepended, because a bare
    bullet means nothing once it is separated from its trial.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from trial_signal.config import HER2_DRUGS

MAX_CHARS = 1200

# Brand names and common abbreviations -> generic name used everywhere else in the project.
_ALIASES = {
    "tukysa": "tucatinib",
    "nerlynx": "neratinib",
    "tykerb": "lapatinib",
    "tyverb": "lapatinib",
    "herceptin": "trastuzumab",
    "perjeta": "pertuzumab",
    "kadcyla": "ado-trastuzumab emtansine",
    "t-dm1": "ado-trastuzumab emtansine",
    "trastuzumab emtansine": "ado-trastuzumab emtansine",
    "enhertu": "fam-trastuzumab deruxtecan-nxki",
    "t-dxd": "fam-trastuzumab deruxtecan-nxki",
    "trastuzumab deruxtecan": "fam-trastuzumab deruxtecan-nxki",
}


class Chunk(BaseModel):
    chunk_id: str
    nct_id: str
    drugs: str  # " | "-separated generic names: whitespace-delimited so a LIKE filter matches whole tokens
    phase: str | None
    section: str  # summary | description | eligibility | outcome | arm
    criteria_type: str | None = None  # inclusion | exclusion for eligibility chunks
    text: str  # header + body: this column is embedded


def normalize_drugs(names: list[str]) -> list[str]:
    """Deterministic baseline. The Phase 3 agent improves on this for messy names."""
    found: set[str] = set()
    for raw in names:
        n = raw.lower()
        # longest aliases first so "trastuzumab deruxtecan" beats "trastuzumab"
        hit = next((g for a, g in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])) if a in n), None)
        if hit is None:
            hit = next((g for g in sorted(HER2_DRUGS, key=len, reverse=True) if g in n), None)
        if hit:
            found.add(hit)
    return sorted(found)


def split_eligibility(text: str) -> list[tuple[str | None, str]]:
    """[(criteria_type, bullet_text), ...]. Lines that aren't bullets join the previous bullet."""
    out: list[tuple[str | None, str]] = []
    current: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower().rstrip(":")
        if low.startswith("inclusion criteria"):
            current = "inclusion"
            continue
        if low.startswith("exclusion criteria"):
            current = "exclusion"
            continue
        m = re.match(r"^(?:[*\-•]|\d+[.)])\s+(.*)", s)
        if m:
            out.append((current, m.group(1).strip()))
        elif out:
            ctype, prev = out[-1]
            out[-1] = (ctype, f"{prev} {s}")
        else:
            out.append((current, s))
    return out


def _windows(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Pack paragraphs into windows of at most max_chars; hard-split a paragraph only if needed."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        while len(p) > max_chars:
            out.append(p[:max_chars])
            p = p[max_chars:]
        if buf and len(buf) + len(p) + 2 > max_chars:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def chunk_trial(study: dict[str, Any]) -> list[Chunk]:
    p = study.get("protocolSection", {})
    nct = p["identificationModule"]["nctId"]
    title = p["identificationModule"].get("briefTitle", "")
    phases = (p.get("designModule") or {}).get("phases") or []
    phase = phases[0] if phases else None
    interventions = [i.get("name", "") for i in (p.get("armsInterventionsModule") or {}).get("interventions", [])]
    drugs = normalize_drugs(interventions + [title])
    drug_str = " | ".join(drugs)

    def make(section: str, i: int, body: str, label: str, ctype: str | None = None) -> Chunk:
        header = f"{nct} | {drug_str or 'unknown drug'} | {phase or 'phase n/a'} | {label}"
        return Chunk(
            chunk_id=f"{nct}:{section}:{i}",
            nct_id=nct,
            drugs=drug_str,
            phase=phase,
            section=section,
            criteria_type=ctype,
            text=f"{header}\n{title}\n\n{body}",
        )

    chunks: list[Chunk] = []
    desc = p.get("descriptionModule") or {}
    if desc.get("briefSummary"):
        chunks.append(make("summary", 0, desc["briefSummary"].strip(), "Brief summary"))
    for i, w in enumerate(_windows(desc.get("detailedDescription") or "")):
        chunks.append(make("description", i, w, "Detailed description"))

    crit = (p.get("eligibilityModule") or {}).get("eligibilityCriteria") or ""
    for i, (ctype, bullet) in enumerate(split_eligibility(crit)):
        label = f"{(ctype or 'other').capitalize()} criterion"
        chunks.append(make("eligibility", i, bullet, label, ctype))

    outcomes = p.get("outcomesModule") or {}
    i = 0
    for kind in ("primaryOutcomes", "secondaryOutcomes", "otherOutcomes"):
        for o in outcomes.get(kind, []) or []:
            body = " | ".join(x for x in (o.get("measure"), o.get("description"), o.get("timeFrame")) if x)
            chunks.append(make("outcome", i, body, f"{kind.replace('Outcomes', '').capitalize()} outcome"))
            i += 1

    for i, arm in enumerate((p.get("armsInterventionsModule") or {}).get("armGroups", []) or []):
        body = " | ".join(
            x for x in (arm.get("label"), arm.get("type"), arm.get("description"), ", ".join(arm.get("interventionNames", []))) if x
        )
        chunks.append(make("arm", i, body, "Arm"))
    return chunks
