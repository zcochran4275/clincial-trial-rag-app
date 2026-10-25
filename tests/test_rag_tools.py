import json
from types import SimpleNamespace

from trial_signal.rag.agent import answer
from trial_signal.rag.tools import parse_search_result, run_tool

SEARCH_RAW = {
    "manifest": {"columns": [{"name": "chunk_id"}, {"name": "nct_id"}, {"name": "section"}, {"name": "text"}, {"name": "score"}]},
    "result": {"data_array": [["NCT00000001:eligibility:2", "NCT00000001", "eligibility", "Exclusion: prior lapatinib", 0.91]]},
}


class FakeBackend:
    def __init__(self):
        self.searches, self.statements = [], []

    def vector_search(self, query, filters, k):
        self.searches.append((query, filters, k))
        return SEARCH_RAW

    def sql(self, statement):
        self.statements.append(statement)
        return [{"term_key": "DIARRHOEA", "pooled_rate": 0.62}]


def test_parse_search_result():
    hits = parse_search_result(SEARCH_RAW)
    assert hits[0]["chunk_id"] == "NCT00000001:eligibility:2" and hits[0]["score"] == 0.91


def test_search_builds_filters():
    b = FakeBackend()
    run_tool("search_trial_text", {"query": "brain mets", "drug": "Ado-Trastuzumab Emtansine", "section": "eligibility"}, b)
    _, filters, k = b.searches[0]
    assert filters == {"section": "eligibility", "drugs LIKE": "ado-trastuzumab"}
    assert k == 8


def test_unsafe_sql_returns_error_to_model_not_exception():
    b = FakeBackend()
    out = json.loads(run_tool("query_sql", {"sql": "DROP TABLE x"}, b))
    assert "rejected" in out["error"] and b.statements == []


def test_get_trial_rejects_bad_id():
    out = json.loads(run_tool("get_trial", {"nct_id": "NCT1' OR '1'='1"}, FakeBackend()))
    assert "error" in out


# --- agent loop with a scripted fake OpenAI client ---------------------------------------

def _msg(content, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def _call(id, name, args):
    return SimpleNamespace(id=id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: script.pop(0)))


def test_agent_loop_traces_tools_and_citations():
    script = [
        _msg(None, [_call("t1", "search_trial_text", {"query": "lapatinib exclusion"})]),
        _msg("It excluded recent lapatinib use [NCT00000001]."),
    ]
    t = answer("Did it exclude lapatinib?", FakeBackend(), client=FakeClient(script))
    assert t.turns == 2
    assert t.retrieved_chunks == ["NCT00000001:eligibility:2"]
    assert t.cited_nct_ids == ["NCT00000001"]
    assert t.input_tokens == 20


def test_agent_loop_returns_bad_json_args_to_model():
    bad = SimpleNamespace(id="t1", function=SimpleNamespace(name="query_sql", arguments="{not json"))
    script = [_msg(None, [bad]), _msg("No data.")]
    client = FakeClient(script)
    t = answer("q", FakeBackend(), client=client)
    assert t.answer == "No data." and t.tool_calls == [{"name": "query_sql", "args": {}}]
