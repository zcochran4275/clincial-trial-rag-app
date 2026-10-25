import pandas as pd
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from trial_signal.schemas import ArmAdverseEvent, Trial
from trial_signal.silver.flatten import REJECT_DTYPES, flatten_adverse_events, flatten_trial


def test_flatten_trial_core_fields(tucatinib_study):
    t = flatten_trial(tucatinib_study)
    assert t.nct_id == "NCT00000001"
    assert t.phases == ["PHASE2"]
    assert t.enrollment == 612
    assert "Tucatinib" in t.interventions
    assert t.eligibility_criteria.startswith("Inclusion Criteria")


def test_flatten_trial_tolerates_missing_modules(ctgov_page2):
    t = flatten_trial(ctgov_page2["studies"][0])
    assert t.interventions == [] and t.eligibility_criteria is None


def test_adverse_events_one_row_per_term_per_arm(tucatinib_study):
    rows, rejects = flatten_adverse_events(tucatinib_study)
    # 1 serious term x 2 arms + 2 other terms x 2 arms = 6, minus 1 invalid row
    assert len(rows) == 5
    assert len(rejects) == 1
    assert "affected > at risk" in rejects[0]["reason"]


def test_reject_columns_match_declared_schema(tucatinib_study):
    _, rejects = flatten_adverse_events(tucatinib_study)
    assert list(rejects[0]) == list(REJECT_DTYPES)


def test_empty_rejects_still_write_typed_columns(tmp_path):
    path = tmp_path / "rejects.parquet"
    pd.DataFrame([], columns=list(REJECT_DTYPES)).astype(REJECT_DTYPES).to_parquet(path, index=False)
    schema = pq.read_schema(path)
    assert schema.names == list(REJECT_DTYPES)
    assert str(schema.field("num_affected").type) == "int64" and "string" in str(schema.field("term").type)


def test_adverse_event_rates_are_proportions(tucatinib_study):
    rows, _ = flatten_adverse_events(tucatinib_study)
    assert all(0 <= r.rate <= 1 for r in rows)


def test_blank_term_is_rejected_not_dropped_silently(ctgov_page1):
    rows, rejects = flatten_adverse_events(ctgov_page1["studies"][1])
    assert [r.term for r in rows] == ["Nausea"]
    assert len(rejects) == 1


def test_study_without_results_yields_nothing(ctgov_page2):
    assert flatten_adverse_events(ctgov_page2["studies"][0]) == ([], [])


@pytest.mark.parametrize("bad_id", ["NCT123", "nct00000001", "", "NCT000000011"])
def test_trial_rejects_malformed_nct_id(bad_id):
    with pytest.raises(ValidationError):
        Trial(nct_id=bad_id, brief_title="x", overall_status="COMPLETED")


@pytest.mark.parametrize(
    "affected, at_risk, ok",
    [(0, 0, True), (5, 10, True), (10, 10, True), (11, 10, False), (-1, 10, False)],
)
def test_arm_event_count_rules(affected, at_risk, ok):
    kwargs = dict(nct_id="NCT00000001", group_id="EG000", group_title="A", seriousness="other",
                  term="Nausea", num_affected=affected, num_at_risk=at_risk)
    if ok:
        ArmAdverseEvent(**kwargs)
    else:
        with pytest.raises(ValidationError):
            ArmAdverseEvent(**kwargs)
