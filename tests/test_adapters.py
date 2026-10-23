"""Harness-only smoke checks: the adapters forward, without solving exercises."""
import sys
from types import ModuleType

import pytest

import adapters


@pytest.mark.parametrize("name,module,args,kwargs,forwarded", [
    ("parse_action", "protocol", ("raw",), {}, {}),
    ("tokenize_text", "retrieval", ("text",), {}, {}),
    ("run_search", "retrieval", ([], "q"), {}, {"k": 3}),
    ("run_lookup", "retrieval", ([], "T"), {}, {}),
    ("execute_tool", "tools", ({}, {}), {}, {"max_observation_chars": 8000}),
    ("make_tools", "tools", ([],), {}, {}),
    ("build_initial_messages", "prompting", ("q", "tools"), {"max_steps": 3, "max_tool_calls": 2},
     {"max_steps": 3, "max_tool_calls": 2}),
    ("append_observation", "prompting", ([], "raw", "obs"), {}, {}),
    ("run_agent", "agent", ("q", object(), {}), {}, {"max_steps": 8, "max_tool_calls": 6}),
    ("make_model", "providers", ("ollama", "id", "http://localhost/v1"), {}, {"api_key": None, "timeout": 60.0}),
    ("write_jsonl_record", "tracing", ("trace.jsonl", {}), {}, {}),
    ("load_completed_ids", "tracing", ("trace.jsonl", "run"), {}, {}),
    ("load_hotpot", "dataset", ("data.json",), {}, {}),
    ("split_example", "dataset", ({},), {}, {}),
    ("normalize_answer", "metrics", ("text",), {}, {}),
    ("score_answer", "metrics", ("prediction", "gold"), {}, {}),
    ("score_support", "metrics", ([], []), {}, {}),
    ("make_fixed_splits", "evaluation", ([],), {}, {"seed": 42}),
    ("make_predictions_json", "evaluation", ([], []), {}, {}),
    ("run_batch", "evaluation", ("data", "ids", object(), "react", {}, "out"), {}, {}),
])
def test_public_adapter_forwards_without_transforming(monkeypatch, name, module, args, kwargs, forwarded):
    implementation = ModuleType(f"react_agent.{module}")
    calls = []
    sentinel = object()
    def function(*actual_args, **actual_kwargs):
        calls.append((actual_args, actual_kwargs))
        return sentinel
    setattr(implementation, name, function)
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    assert getattr(adapters, name)(*args, **kwargs) is sentinel
    assert calls == [(args, forwarded)]


def test_custom_limits_are_forwarded(monkeypatch):
    implementation = ModuleType("react_agent.tools")
    calls = []
    implementation.execute_tool = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    action, tools = {}, {}
    adapters.execute_tool(action, tools, 64)
    assert calls == [((action, tools), {"max_observation_chars": 64})]
