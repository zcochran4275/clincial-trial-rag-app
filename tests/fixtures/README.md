# Test fixtures

These files mirror the documented response shapes of the ClinicalTrials.gov v2 `/studies`
endpoint and the openFDA `drug/event` count endpoint. NCT IDs and counts are **synthetic**.
The NCT00000002 record contains a deliberately blank term and the NCT00000001 placebo arm has
an `affected > at risk` row, so the reject path gets exercised.

**First task on day 1:** run one real query (`python -m trial_signal.cli fetch --drug tucatinib --max-pages 1`)
and diff a real record against these. If the live API shape differs, update the fixtures
first, then the code — that's the contract-testing habit interviewers like to hear about.
