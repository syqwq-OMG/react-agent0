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
    assert result["status"] in ("error", "max_steps")
    assert result.get("answer", "") != "not valid json"
