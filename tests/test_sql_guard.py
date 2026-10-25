import pytest

from trial_signal.rag.sql_guard import UnsafeSQL, guard


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM workspace.trial_signal.silver_trials",
        "with t as (select nct_id from workspace.trial_signal.silver_trials) select * from t",
        "SELECT term FROM workspace.trial_signal.silver_arm_adverse_events WHERE term = 'Drug: create; drop'",
        "SELECT * FROM workspace.trial_signal.gold_term_trial_rates LIMIT 5;",
    ],
)
def test_allowed(sql):
    guard(sql)


@pytest.mark.parametrize(
    "sql, reason",
    [
        ("DROP TABLE workspace.trial_signal.silver_trials", "only SELECT"),
        ("SELECT 1; DELETE FROM workspace.trial_signal.silver_trials", "one statement"),
        ("SELECT * FROM workspace.trial_signal.silver_trials /* */ ; GRANT ALL", "one statement"),
        ("SELECT * FROM workspace.other_schema.secrets", "outside schema"),
        ("WITH x AS (DELETE FROM workspace.trial_signal.silver_trials) SELECT 1", "not allowed"),
        ("", "empty"),
        ("-- just a comment", "empty"),
    ],
)
def test_rejected(sql, reason):
    with pytest.raises(UnsafeSQL) as e:
        guard(sql)
    assert reason in str(e.value)


def test_limit_added_when_missing():
    assert guard("SELECT * FROM workspace.trial_signal.silver_trials").endswith("LIMIT 200")


def test_existing_limit_kept():
    out = guard("SELECT * FROM workspace.trial_signal.silver_trials LIMIT 5")
    assert out.endswith("LIMIT 5") and "LIMIT 200" not in out
