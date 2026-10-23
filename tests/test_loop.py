import json
from adapters import run_agent


def _tool(tool, **args):
    return json.dumps({"type": "tool_call", "tool": tool, "args": args})


def _final(answer):
    return json.dumps({"type": "final", "answer": answer, "evidence": []})


def test_final_stops_without_tool_calls(scripted_model):
    model = scripted_model([_final("Paris")])
    result = run_agent("Capital of France?", model, {}, max_steps=3, max_tool_calls=2)
    assert result["status"] == "final"
    assert result["answer"] == "Paris"
    assert result["num_tool_calls"] == 0
    assert len(model.seen_messages) == 1
    assert isinstance(result["trace"], list)


def test_observation_reaches_next_model_call(scripted_model):
    model = scripted_model([_tool("echo", message="step 1"), _final("Done")])
    result = run_agent("Do it", model, {"echo": lambda message: "FOUND_123 " + message},
                       max_steps=3, max_tool_calls=2)
    assert result["status"] == "final"
    assert result["num_tool_calls"] == 1
    assert len(model.seen_messages) == 2
    assert any("FOUND_123" in str(item.get("content", "")) for item in model.seen_messages[1])


def test_two_step_tool_chain(scripted_model):
    model = scripted_model([_tool("echo", message="first"), _tool("echo", message="second"), _final("ok")])
    output = []
    result = run_agent("Go", model, {"echo": lambda message: output.append(message) or message},
                       max_steps=3, max_tool_calls=2)
    assert output == ["first", "second"]
    assert result["status"] == "final"
    assert result["num_tool_calls"] == 2
    assert len(model.seen_messages) == 3


def test_decision_budget_stops(scripted_model):
    model = scripted_model([_tool("echo", message="1"), _tool("echo", message="2")])
    result = run_agent("Go", model, {"echo": lambda message: message}, max_steps=2, max_tool_calls=3)
    assert result["status"] == "max_steps"
    assert len(model.seen_messages) == 2


def test_tool_budget_stops(scripted_model):
    model = scripted_model([_tool("echo", message="1"), _tool("echo", message="2")])
    result = run_agent("Go", model, {"echo": lambda message: message}, max_steps=4, max_tool_calls=1)
    assert result["status"] == "tool_limit"
    assert result["num_tool_calls"] <= 1


def test_invalid_protocol_is_not_silent(scripted_model):
    model = scripted_model(["not valid json"])
    result = run_agent("Go", model, {}, max_steps=1, max_tool_calls=1)
    assert result["status"] == "error"
    assert result.get("answer", "") != "not valid json"


import pytest
from adapters import make_tools


def _assert_result(result, model, status, tool_calls):
    assert result["status"] == status
    assert result["num_llm_calls"] == len(model.seen_messages)
    assert result["num_tool_calls"] == tool_calls
    assert isinstance(result["trace"], list)
    if status != "final":
        assert result["answer"] == ""
        assert result["evidence"] == []


def test_actual_two_hop_evidence_and_both_observations(toy_context, scripted_model):
    evidence = [["Ada Lovelace", 1], ["Lord Byron", 1]]
    raw = [_tool("lookup", title="Ada Lovelace"), _tool("lookup", title="Lord Byron"),
           json.dumps({"type": "final", "answer": "London", "evidence": evidence})]
    model = scripted_model(raw)
    result = run_agent("Where was Ada Lovelace's father born?", model, make_tools(toy_context), 3, 2)
    _assert_result(result, model, "final", 2)
    assert result["answer"] == "London"
    assert result["evidence"] == evidence
    assert "Lord Byron" in json.dumps(model.seen_messages[1])
    assert "London" in json.dumps(model.seen_messages[2])
    assert "Ada Lovelace" in json.dumps(model.seen_messages[2])
    assert all(any(m["content"] == "Where was Ada Lovelace's father born?" for m in seen)
               for seen in model.seen_messages)


@pytest.mark.parametrize("steps,tools,outputs,status,attempts", [
    (0, 6, [], "max_steps", 0),
    (1, 0, [_final("direct")], "final", 0),
    (2, 0, [_tool("echo", message="blocked")], "tool_limit", 0),
    (1, 2, [_tool("echo", message="last")], "max_steps", 1),
    (2, 2, [_tool("echo", message="1"), _tool("echo", message="2")], "max_steps", 2),
    (4, 2, [_tool("echo", message="1"), _tool("echo", message="2"), _tool("echo", message="3")], "tool_limit", 2),
])
def test_budget_states_and_actual_side_effects(scripted_model, steps, tools, outputs, status, attempts):
    model = scripted_model(outputs)
    called = []
    result = run_agent("q", model, {"echo": lambda message: called.append(message) or message}, steps, tools)
    _assert_result(result, model, status, attempts)
    assert len(called) == attempts


@pytest.mark.parametrize("steps,tools", [(-1, 1), (1, -1), (True, 1), (1, False), (1.2, 1), (1, None)])
def test_invalid_budget_before_model_or_tool(scripted_model, steps, tools):
    model = scripted_model([])
    calls = []
    with pytest.raises(ValueError):
        run_agent("q", model, {"echo": lambda: calls.append(1)}, steps, tools)
    assert model.seen_messages == []
    assert calls == []


@pytest.mark.parametrize("first", [_tool("missing"), _tool("echo"), _tool("broken")])
def test_failed_attempt_consumes_budget_and_can_recover(scripted_model, first):
    calls = []
    def broken():
        raise RuntimeError("CONTROLLED_FAILURE")
    model = scripted_model([first, _tool("echo", message="fixed"), _final("ok")])
    result = run_agent("q", model, {"echo": lambda message: calls.append(message) or message, "broken": broken}, 3, 2)
    _assert_result(result, model, "final", 2)
    assert calls == ["fixed"]
    assert "ERROR:" in json.dumps(model.seen_messages[1])


def test_failed_tool_attempt_also_exhausts_allowance(scripted_model):
    model = scripted_model([_tool("missing"), _tool("echo", message="blocked")])
    calls = []
    result = run_agent("q", model, {"echo": lambda message: calls.append(message)}, 3, 1)
    _assert_result(result, model, "tool_limit", 1)
    assert calls == []


def test_protocol_error_is_error_with_raw_trace(scripted_model):
    raw = "BAD_JSON_SENTINEL"
    model = scripted_model([raw])
    result = run_agent("q", model, {}, 2, 1)
    _assert_result(result, model, "error", 0)
    assert raw in json.dumps(result["trace"])
    assert result["trace"][-1]["error"]


def test_generate_exception_is_controlled(scripted_model):
    model = scripted_model([RuntimeError("BACKEND_FAILURE_SENTINEL")])
    result = run_agent("q", model, {}, 2, 1)
    _assert_result(result, model, "error", 0)
    assert "BACKEND_FAILURE_SENTINEL" in json.dumps(result["trace"])
    assert result["trace"][-1]["error"]


def test_trace_schema_matches_actual_events(scripted_model):
    raw = [_tool("echo", message="OBS"), _final("ok")]
    model = scripted_model(raw)
    result = run_agent("q", model, {"echo": lambda message: message}, 2, 1)
    assert result["num_llm_calls"] == len(result["trace"]) == 2
    required = {"step_index", "raw_output", "action", "tool_result", "error", "elapsed_seconds"}
    for i, step in enumerate(result["trace"]):
        assert required <= step.keys()
        assert step["step_index"] == i
        assert step["raw_output"] == raw[i]
        assert step["action"] == json.loads(raw[i])
        assert step["error"] is None
        assert isinstance(step["elapsed_seconds"], (int, float))
        assert step["elapsed_seconds"] >= 0
    assert result["trace"][0]["tool_result"] == {"ok": True, "observation": "OBS"}
    assert result["trace"][1]["tool_result"] is None


def test_initial_prompt_does_not_dump_context_or_gold(scripted_model):
    context = [["PRIVATE_TITLE_91", ["PRIVATE_SENTENCE_92"]]]
    model = scripted_model([_final("unknown")])
    run_agent("PUBLIC_QUESTION", model, make_tools(context))
    initial = json.dumps(model.seen_messages[0])
    assert "PUBLIC_QUESTION" in initial
    assert "PRIVATE_TITLE_91" not in initial
    assert "PRIVATE_SENTENCE_92" not in initial


@pytest.mark.parametrize("outputs,steps,tools,status,attempts", [
    ([_tool("missing"), _final("recovered")], 2, 1, "final", 1),
    ([_tool("echo", message="last")], 1, 1, "max_steps", 1),
    ([_tool("echo", message="blocked")], 1, 0, "tool_limit", 0),
])
def test_failure_and_budget_traces_are_recountable(scripted_model, outputs, steps, tools, status, attempts):
    model = scripted_model(outputs)
    result = run_agent("q", model, {"echo": lambda message: message}, steps, tools)
    _assert_result(result, model, status, attempts)
    assert len(result["trace"]) == result["num_llm_calls"]
    assert [step["step_index"] for step in result["trace"]] == list(range(len(outputs)))
    assert sum(step["tool_result"] is not None for step in result["trace"]) == attempts
    for step in result["trace"]:
        assert {"raw_output", "action", "tool_result", "error", "elapsed_seconds"} <= step.keys()


def test_agent_preserves_invalid_citations_for_offline_scoring(scripted_model, toy_context):
    evidence = [["Unknown title", 999]]
    model = scripted_model([json.dumps({"type": "final", "answer": "x", "evidence": evidence})])
    result = run_agent("q", model, make_tools(toy_context))
    assert result["status"] == "final"
    assert result["evidence"] == evidence
