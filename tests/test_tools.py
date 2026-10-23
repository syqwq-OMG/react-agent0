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
