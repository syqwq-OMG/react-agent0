import json
import pytest
from adapters import parse_action


def test_tool_call_round_trip():
    original = {"type": "tool_call", "tool": "search", "args": {"query": "Ada Lovelace", "k": 3}}
    output = parse_action(json.dumps(original))
    assert output["type"] == "tool_call"
    assert output["tool"] == "search"
    assert output["args"]["query"] == "Ada Lovelace"


def test_final_action():
    output = parse_action('{"type":"final","answer":"Lord Byron","evidence":[["Ada Lovelace",1]]}')
    assert output["type"] == "final"
    assert output["answer"] == "Lord Byron"
    assert output["evidence"] == [["Ada Lovelace", 1]]


@pytest.mark.parametrize("raw", [
    "not json",
    "[]",
    '{}',
    '{"type":"tool_call","tool":"search"}',
    '{"type":"tool_call","tool":"search","args":[]}',
    '{"type":"tool_call","tool":"","args":{}}',
    '{"type":"final"}',
    '{"type":"final","answer":10}',
    '{"type":"final","answer":"x","evidence":[["Title","one"]]}',
    '{"type":"secret","payload":"unknown"}',
    '{"type":"final","answer":"a"} trailing garbage',
])
def test_invalid_action_raises_valueerror(raw):
    with pytest.raises(ValueError):
        parse_action(raw)
