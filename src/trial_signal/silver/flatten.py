"""Deterministic silver transforms — the hand-written baseline.

This covers the fields ClinicalTrials.gov already structures. The agent (Phase 3) takes
over the free-text fields this code deliberately leaves alone (eligibility criteria,
outcome-measure descriptions, drug-name normalization). Its output is compared against
this baseline where the two overlap.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from trial_signal.schemas import ArmAdverseEvent, Trial


def _get(d: dict[str, Any], *path: str, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def flatten_trial(study: dict[str, Any]) -> Trial:
    p = study.get("protocolSection", {})
    return Trial(
        nct_id=_get(p, "identificationModule", "nctId"),
        brief_title=_get(p, "identificationModule", "briefTitle", default=""),
        overall_status=_get(p, "statusModule", "overallStatus", default="UNKNOWN"),
        phases=_get(p, "designModule", "phases", default=[]) or [],
        enrollment=_get(p, "designModule", "enrollmentInfo", "count"),
        start_date=_get(p, "statusModule", "startDateStruct", "date"),
        completion_date=_get(p, "statusModule", "completionDateStruct", "date"),
        conditions=_get(p, "conditionsModule", "conditions", default=[]) or [],
        interventions=[i.get("name", "") for i in _get(p, "armsInterventionsModule", "interventions", default=[]) or []],
        eligibility_criteria=_get(p, "eligibilityModule", "eligibilityCriteria"),
    )


def flatten_adverse_events(study: dict[str, Any]) -> tuple[list[ArmAdverseEvent], list[dict[str, Any]]]:
    """Return (valid rows, rejects). Rejects carry the reason so the QC report can count them."""
    nct = _get(study, "protocolSection", "identificationModule", "nctId")
    ae = _get(study, "resultsSection", "adverseEventsModule", default={}) or {}
    groups = {g["id"]: g.get("title", g["id"]) for g in ae.get("eventGroups", [])}

    rows: list[ArmAdverseEvent] = []
    rejects: list[dict[str, Any]] = []
    for seriousness, key in (("serious", "seriousEvents"), ("other", "otherEvents")):
        for event in ae.get(key, []) or []:
            for stat in event.get("stats", []) or []:
                gid = stat.get("groupId")
                candidate = {
                    "nct_id": nct,
                    "group_id": gid,
                    "group_title": groups.get(gid, gid or ""),
                    "seriousness": seriousness,
                    "term": event.get("term", ""),
                    "organ_system": event.get("organSystem"),
                    "source_vocabulary": event.get("sourceVocabulary"),
                    "num_affected": stat.get("numAffected", 0),
                    "num_at_risk": stat.get("numAtRisk", 0),
                    "num_events": stat.get("numEvents"),
                }
                try:
                    rows.append(ArmAdverseEvent(**candidate))
                except ValidationError as exc:
                    rejects.append({**candidate, "reason": exc.errors()[0]["msg"]})
    return rows, rejects
