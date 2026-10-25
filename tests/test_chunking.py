import pytest

from trial_signal.rag.chunking import chunk_trial, normalize_drugs, split_eligibility


def test_eligibility_split_per_bullet_with_type():
    text = "Inclusion Criteria:\n\n* HER2+ by central lab\n* Age >= 18\n\nExclusion Criteria:\n\n1. Prior lapatinib\n   within 12 months"
    assert split_eligibility(text) == [
        ("inclusion", "HER2+ by central lab"),
        ("inclusion", "Age >= 18"),
        ("exclusion", "Prior lapatinib within 12 months"),  # continuation line joined
    ]


@pytest.mark.parametrize(
    "names, expected",
    [
        (["Tucatinib", "Placebo"], ["tucatinib"]),
        (["Enhertu"], ["fam-trastuzumab deruxtecan-nxki"]),
        (["Trastuzumab deruxtecan (T-DXd)"], ["fam-trastuzumab deruxtecan-nxki"]),  # not plain trastuzumab
        (["T-DM1"], ["ado-trastuzumab emtansine"]),
        (["Capecitabine"], []),
    ],
)
def test_normalize_drugs(names, expected):
    assert normalize_drugs(names) == expected


def test_chunk_trial_sections_and_ids(tucatinib_study):
    chunks = chunk_trial(tucatinib_study)
    sections = [c.section for c in chunks]
    assert sections.count("summary") == 1
    assert sections.count("eligibility") == 3
    assert sections.count("outcome") == 1
    assert sections.count("arm") == 1
    assert "description" in sections
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id must be unique: it is the index primary key"


def test_every_chunk_carries_context_header(tucatinib_study):
    for c in chunk_trial(tucatinib_study):
        first_line = c.text.splitlines()[0]
        assert first_line.startswith("NCT00000001 | ")
        assert "tucatinib" in first_line


def test_drugs_column_is_token_filterable(tucatinib_study):
    c = chunk_trial(tucatinib_study)[0]
    assert c.drugs == "trastuzumab | tucatinib"
    assert "tucatinib" in c.drugs.split()


def test_exclusion_chunk_is_tagged(tucatinib_study):
    excl = [c for c in chunk_trial(tucatinib_study) if c.criteria_type == "exclusion"]
    assert len(excl) == 1 and "lapatinib" in excl[0].text


def test_trial_without_text_yields_no_chunks(ctgov_page2):
    assert chunk_trial(ctgov_page2["studies"][0]) == []
