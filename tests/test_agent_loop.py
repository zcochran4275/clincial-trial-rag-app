import json
from types import SimpleNamespace

import pytest

from trial_signal.agent import loop, sandbox
from trial_signal.schemas import DrugNormalization, EligibilityExtraction

# --- scripted fake OpenAI client (same shape as tests/test_rag_tools.py) -----------------


def _msg(content=None, *calls):
    message = SimpleNamespace(content=content, tool_calls=list(calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def _call(name, args, id="c"):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=raw))


class FakeClient:
    def __init__(self, script):
        self.sent = []

        def create(**kw):
            self.sent.append(kw["messages"][:])
            return script.pop(0)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.fixture
def dirs(tmp_path):
    ws, samples = tmp_path / "ws", tmp_path / "samples"
    ws.mkdir()
    samples.mkdir()
    (samples / "NCT00000001.json").write_text(json.dumps({"protocolSection": {"eligibilityModule": {"minimumAge": "18 Years"}}}))
    return ws, samples


@pytest.fixture
def fake_pytest(monkeypatch):
    """Sandbox stand-in: results are popped in order, so tests script pass/fail sequences."""
    results = []
    monkeypatch.setattr(sandbox, "run_pytest", lambda ws, s: sandbox.SandboxResult(results.pop(0), "1 passed"))
    return results


WRITE_BOTH = [_call("write_file", {"path": "extract.py", "content": "def extract(s): return {}"}, "w1"),
              _call("write_file", {"path": "test_extract.py", "content": "def test(): pass"}, "w2")]


def _tool_outputs(client, call):
    """Tool results the model saw for the first time on its `call`-th request (0-based)."""
    new = client.sent[call][len(client.sent[call - 1]):]
    return [m["content"] for m in new if m["role"] == "tool"]


def test_happy_path_submits_and_logs_each_turn(dirs, fake_pytest, tmp_path):
    ws, samples = dirs
    fake_pytest.append(True)
    log = tmp_path / "turns.jsonl"
    client = FakeClient([
        _msg(None, _call("read_sample_records", {"n": 5})),
        _msg(None, *WRITE_BOTH),
        _msg(None, _call("run_tests", {})),
        _msg(None, _call("submit", {"notes": "all good"})),
    ])
    run = loop.run_agent("eligibility", ws, samples, client=client, log_path=log)
    assert (run.submitted, run.turns, run.notes) == (True, 4, "all good")
    assert run.tests_passed_first_try is True and run.test_runs == 1
    assert run.input_tokens == 400 and run.tool_errors == 0
    turns = [json.loads(line) for line in log.read_text().splitlines()]
    assert [t["turn"] for t in turns] == [1, 2, 3, 4]
    assert turns[1]["calls"][0]["args"]["content"] == "<25 chars>"  # code isn't dumped into the log
    assert json.loads(_tool_outputs(client, 1)[0])["returned"] == 1


def test_path_traversal_is_refused_and_reported_to_model(dirs, fake_pytest):
    ws, samples = dirs
    client = FakeClient([
        _msg(None, _call("write_file", {"path": "../evil.py", "content": "x"})),
        _msg("ok"), _msg("ok"),
    ])
    run = loop.run_agent("eligibility", ws, samples, client=client)
    assert not (ws.parent / "evil.py").exists()
    assert "path must be one of" in _tool_outputs(client, 1)[0]
    assert run.tool_errors == 1


def test_submit_refused_until_tests_pass_after_last_write(dirs, fake_pytest):
    ws, samples = dirs
    fake_pytest += [False, True, True]
    client = FakeClient([
        _msg(None, *WRITE_BOTH),
        _msg(None, _call("run_tests", {})),                # fails
        _msg(None, _call("submit", {"notes": "early"})),   # refused: tests red
        _msg(None, _call("run_tests", {})),                # passes
        _msg(None, _call("write_file", {"path": "extract.py", "content": "changed"})),
        _msg(None, _call("submit", {"notes": "stale"})),   # refused: wrote after green
        _msg(None, _call("run_tests", {})),                # passes
        _msg(None, _call("submit", {"notes": "done"})),
    ])
    run = loop.run_agent("eligibility", ws, samples, client=client)
    assert run.submitted and run.notes == "done" and run.turns == 8
    assert "must pass" in _tool_outputs(client, 3)[0] and "must pass" in _tool_outputs(client, 6)[0]
    assert run.tests_passed_first_try is False and run.test_runs == 3


def test_run_tests_before_files_exist_is_a_tool_error(dirs, fake_pytest):
    ws, samples = dirs
    client = FakeClient([_msg(None, _call("run_tests", {})), _msg("hm"), _msg("hm")])
    run = loop.run_agent("eligibility", ws, samples, client=client)
    assert "write extract.py, test_extract.py first" in _tool_outputs(client, 1)[0]
    assert run.test_runs == 0


def test_malformed_arguments_and_unknown_tool_go_back_as_errors(dirs, fake_pytest):
    ws, samples = dirs
    client = FakeClient([
        _msg(None, _call("write_file", "{not json", "a"), _call("rm_rf", {}, "b")),
        _msg("done?"), _msg("done."),
    ])
    run = loop.run_agent("eligibility", ws, samples, client=client)
    outs = _tool_outputs(client, 1)
    assert "JSONDecodeError" in outs[0] and "unknown tool rm_rf" in outs[1]
    assert run.tool_errors == 2


def test_one_nudge_then_stop_when_model_quits_talking(dirs, fake_pytest):
    ws, samples = dirs
    client = FakeClient([_msg("I think I'm done."), _msg("Really done.")])
    run = loop.run_agent("eligibility", ws, samples, client=client)
    assert (run.submitted, run.turns, run.stop_reason) == (False, 2, "stopped without submitting")
    assert client.sent[1][-1] == {"role": "user", "content": "Keep going with the tools: run_tests until they pass, then submit."}


def test_turn_limit(dirs, fake_pytest, monkeypatch):
    ws, samples = dirs
    monkeypatch.setattr(loop, "MAX_TURNS", 3)
    client = FakeClient([_msg(None, _call("read_sample_records", {"n": 1})) for _ in range(3)])
    run = loop.run_agent("eligibility", ws, samples, client=client)
    assert (run.turns, run.stop_reason) == (3, "hit turn limit")


def test_validate_outputs_holds_agent_to_the_contract():
    results = [
        {"file": "NCT00000001.json", "output": {"her2_definition": "IHC 3+", "requires_prior_trastuzumab": True,
                                                "allows_brain_mets": None, "min_age": 18}},
        {"file": "NCT00000002.json", "output": {"min_age": 18, "surprise": 1}},  # extra key
        {"file": "NCT00000003.json", "output": {"min_age": -5}},
        {"file": "NCT00000004.json", "error": "KeyError: 'protocolSection'"},
    ]
    valid, rejects = loop.validate_outputs("eligibility", results)
    assert [v["nct_id"] for v in valid] == ["NCT00000001"] and valid[0]["min_age"] == 18
    assert [(r["nct_id"], r["reason"]) for r in rejects] == [
        ("NCT00000002", "contract violation"), ("NCT00000003", "contract violation"), ("NCT00000004", "extract raised")]


def test_drug_contract_only_allows_study_generics():
    DrugNormalization(interventions=[{"name": "Kadcyla", "generic": "ado-trastuzumab emtansine"}, {"name": "Placebo", "generic": None}])
    with pytest.raises(ValueError):
        DrugNormalization(interventions=[{"name": "Keytruda", "generic": "pembrolizumab"}])
    with pytest.raises(ValueError):
        EligibilityExtraction(min_age=200)


def test_every_task_names_a_contract():
    assert set(loop.TASKS) == {"eligibility", "drug_normalization"}
    assert "fam-trastuzumab deruxtecan-nxki" in loop.TASKS["drug_normalization"].goal
