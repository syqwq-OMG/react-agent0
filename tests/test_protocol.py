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


@pytest.mark.parametrize("action", [
    {"type": "final", "answer": "", "note": "No evidence yet"},
    {"type": "final", "answer": "中文", "evidence": []},
    {"type": "tool_call", "tool": "missing", "args": {}, "note": "Try it"},
    {"type": "tool_call", "tool": "echo", "args": {"nested": {"values": [1, True, None]}}},
])
def test_optional_note_whitespace_and_runtime_independent_parsing(action):
    output = parse_action(" \n" + json.dumps(action) + "\t ")
    for key, value in action.items():
        assert output[key] == value
    if action["type"] == "final":
        assert output["evidence"] == action.get("evidence", [])


@pytest.mark.parametrize("raw", [
    "", "  ", "null", "true", "42", '"text"',
    '```json\n{"type":"final","answer":"x"}\n```',
    '{"type":"final","answer":"a"}{"type":"final","answer":"b"}',
    '{"type":7}', '{"type":[]}',
    '{"type":"tool_call","tool":"  ","args":{}}',
    '{"type":"tool_call","tool":1,"args":{}}',
    '{"type":"tool_call","tool":"echo","args":null}',
    '{"type":"final","answer":"x","note":null}',
    '{"type":"tool_call","tool":"echo","args":{},"note":1}',
    '{"type":"final","answer":"x","evidence":{}}',
    '{"type":"final","answer":"x","evidence":null}',
    '{"type":"final","answer":"x","evidence":[["Title",-1]]}',
    '{"type":"final","answer":"x","evidence":[["Title",true]]}',
    '{"type":"final","answer":"x","evidence":[["Title",false]]}',
    '{"type":"final","answer":"x","evidence":[["Title",1.0]]}',
    '{"type":"final","answer":"x","evidence":[["",0]]}',
    '{"type":"final","answer":"x","evidence":[[7,0]]}',
    '{"type":"final","answer":"x","evidence":[["Title"]]}',
    '{"type":"final","answer":"x","evidence":[["Title",0,1]]}',
    '{"type":"final","answer":"x","evidence":["Title"]}',
    None, 1, {}, [], b'{"type":"final","answer":"x"}',
])
def test_extended_invalid_protocol(raw):
    with pytest.raises(ValueError):
        parse_action(raw)


def test_unknown_evidence_title_is_not_parser_error():
    action = parse_action('{"type":"final","answer":"x","evidence":[["Not in context",999]]}')
    assert action["evidence"] == [["Not in context", 999]]
