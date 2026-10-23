from adapters import execute_tool, run_search, run_lookup


def test_execute_known_tool():
    calls = []
    def echo(message):
        calls.append(message)
        return "received: " + message
    result = execute_tool({"type": "tool_call", "tool": "echo", "args": {"message": "hi"}}, {"echo": echo})
    assert result["ok"] is True
    assert "received: hi" in result["observation"]
    assert calls == ["hi"]


def test_unknown_tool_does_not_run_code():
    result = execute_tool({"type": "tool_call", "tool": "__import__", "args": {"module": "os"}}, {})
    assert result["ok"] is False
    assert isinstance(result["observation"], str)


def test_tool_exception_becomes_error_observation():
    def broken():
        raise RuntimeError("controlled failure")
    result = execute_tool({"type": "tool_call", "tool": "broken", "args": {}}, {"broken": broken})
    assert result["ok"] is False
    assert "failure" in result["observation"].lower() or "error" in result["observation"].lower()


def test_search_retrieves_expected_title(toy_context):
    titles = run_search(toy_context, "Analytical Engine", k=2)
    assert isinstance(titles, list)
    assert 0 < len(titles) <= 2
    assert titles[0] == "Ada Lovelace"
    assert all(isinstance(title, str) for title in titles)


def test_search_is_deterministic_and_bounded(toy_context):
    assert run_search(toy_context, "Lord Byron", 1) == run_search(toy_context, "Lord Byron", 1)
    assert len(run_search(toy_context, "Lord Byron", 1)) <= 1


def test_lookup_preserves_sentence_index(toy_context):
    doc = run_lookup(toy_context, "Ada Lovelace")
    assert doc["title"] == "Ada Lovelace"
    assert doc["sentences"] == toy_context[0][1]
    assert "Lord Byron" in doc["sentences"][1]


import json
import pytest
from adapters import make_tools


def _action(name, **args):
    return {"type": "tool_call", "tool": name, "args": args}


@pytest.mark.parametrize("args", [{}, {"message": "hi", "extra": True}])
def test_argument_binding_errors_are_observations(args):
    calls = []
    def echo(message):
        calls.append(message)
        return message
    result = execute_tool(_action("echo", **args), {"echo": echo})
    assert result["ok"] is False
    assert result["observation"].startswith("ERROR:")
    assert calls == []


@pytest.mark.parametrize("value", [{"b": ["中文", 1], "a": True}, [1, {"a": 2}]])
def test_structured_results_are_deterministic_json(value):
    tools = {"structured": lambda: value}
    first = execute_tool(_action("structured"), tools)
    second = execute_tool(_action("structured"), tools)
    assert first["ok"] is True
    assert first["observation"] == second["observation"]
    assert json.loads(first["observation"]) == value


def test_string_observation_preserved_exactly():
    value = "  中文\nnot JSON\t"
    assert execute_tool(_action("echo"), {"echo": lambda: value}) == {
        "ok": True, "observation": value,
    }


@pytest.mark.parametrize("limit", [64, 8000])
def test_long_output_is_error_with_bounded_visible_preview(limit):
    value = "VISIBLE_START " + "x" * (limit + 100)
    result = execute_tool(_action("long"), {"long": lambda: value}, max_observation_chars=limit)
    observation = result["observation"]
    assert result["ok"] is False
    assert observation.startswith("ERROR:")
    assert len(observation) <= limit
    assert "VISIBLE_START" in observation
    assert any(word in observation.lower() for word in ("truncat", "截断", "省略"))


def test_output_at_limit_is_not_truncated():
    result = execute_tool(_action("echo"), {"echo": lambda: "x" * 64}, 64)
    assert result == {"ok": True, "observation": "x" * 64}


def test_unknown_tool_has_no_other_side_effects():
    calls = []
    result = execute_tool(_action("missing"), {"echo": lambda: calls.append(1)})
    assert result["ok"] is False
    assert result["observation"].startswith("ERROR:")
    assert calls == []


def test_keyboard_interrupt_is_not_swallowed():
    def interrupt():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        execute_tool(_action("interrupt"), {"interrupt": interrupt})


def test_registered_contexts_remain_isolated():
    # Use markers with no shared tokens under the required [a-z0-9]+ tokenizer.
    # SENTINEL_A and SENTINEL_B both contain the token "sentinel", so they are
    # not suitable for checking whether per-question contexts stay isolated.
    first = make_tools([["Only A", ["ALPHAUNIQUE"]]])
    second = make_tools([["Only B", ["BETAUNIQUE"]]])
    assert callable(first["search"]) and callable(first["lookup"])
    assert first["lookup"](title="Only A")["sentences"] == ["ALPHAUNIQUE"]
    assert second["lookup"](title="Only B")["sentences"] == ["BETAUNIQUE"]
    assert first["search"](query="ALPHAUNIQUE", k=1) == ["Only A"]
    assert first["search"](query="BETAUNIQUE", k=1) == []
    with pytest.raises(ValueError):
        first["lookup"](title="Only B")
    with pytest.raises(ValueError):
        second["lookup"](title="Only A")
