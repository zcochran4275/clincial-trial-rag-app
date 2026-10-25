import pytest

from trial_signal.bronze import to_bronze_frame
from trial_signal.ingest import ctgov, openfda


def test_build_params_filters_to_completed_with_results():
    p = ctgov.build_params("tucatinib")
    assert p["query.intr"] == "tucatinib"
    assert p["filter.overallStatus"] == "COMPLETED"
    assert p["aggFilters"] == "results:with"
    assert "pageToken" not in p


def test_iter_studies_follows_page_token(ctgov_page1, ctgov_page2, fake_session):
    session = fake_session([ctgov_page1, ctgov_page2])
    ids = [ctgov.nct_id(s) for s in ctgov.iter_studies("tucatinib", session=session)]
    assert ids == ["NCT00000001", "NCT00000002", "NCT00000003"]
    assert session.calls[1][1]["pageToken"] == "PAGE2TOKEN"


def test_iter_studies_respects_max_pages(ctgov_page1, fake_session):
    session = fake_session([ctgov_page1])
    studies = list(ctgov.iter_studies("tucatinib", session=session, max_pages=1))
    assert len(studies) == 2 and len(session.calls) == 1


@pytest.mark.parametrize(
    "suspect_only, expected_fragment",
    [(True, "drugcharacterization:1"), (False, 'generic_name:"neratinib"')],
)
def test_drug_query(suspect_only, expected_fragment):
    assert expected_fragment in openfda.drug_query("neratinib", suspect_only=suspect_only)


def test_count_reactions_parses_terms(openfda_count, fake_session):
    rows = openfda.count_reactions_for_drug("tucatinib", session=fake_session([openfda_count]))
    assert rows[0] == {"term": "DIARRHOEA", "count": 2100}


def test_openfda_404_means_zero(fake_session):
    assert openfda.total_reports("x", session=fake_session([{}], status_code=404)) == 0


def test_bronze_dedupes_same_study_from_two_queries(tucatinib_study):
    df = to_bronze_frame([tucatinib_study, tucatinib_study], "ctgov", "tucatinib", ctgov.nct_id)
    assert len(df) == 1
    assert df.loc[0, "record_key"] == "NCT00000001"
    assert len(df.loc[0, "content_sha256"]) == 64


def test_ctgov_search_uses_registry_name_not_fda_suffix():
    from trial_signal.config import ctgov_term

    assert ctgov_term("fam-trastuzumab deruxtecan-nxki") == "trastuzumab deruxtecan"
    assert ctgov_term("tucatinib") == "tucatinib"
