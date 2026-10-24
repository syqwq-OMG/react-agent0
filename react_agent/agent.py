import inspect
import time
from typing import Any

from .prompting import append_observation, build_initial_messages
from .protocol import parse_action
from .tools import execute_tool


def _validate_budget(name: str, value: int) -> None:
    """Validate a call budget without accepting bool as an integer."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _describe_tools(tools: dict[str, Any]) -> str:
    """Build a deterministic prompt description without exposing tool state."""
    descriptions = []
    for name in sorted(tools, key=str):
        function = tools[name]
        try:
            signature = str(inspect.signature(function))
        except (TypeError, ValueError):
            signature = "(...)"
        descriptions.append(f"{name}{signature}")
    return "; ".join(descriptions) if descriptions else "No tools available"


def _result(
    status: str,
    trace: list[dict[str, Any]],
    num_llm_calls: int,
    num_tool_calls: int,
    *,
    answer: str = "",
    evidence: list | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "answer": answer,
        "evidence": [] if evidence is None else evidence,
        "trace": trace,
        "num_llm_calls": num_llm_calls,
        "num_tool_calls": num_tool_calls,
    }


def run_agent(
    question: str,
    model: Any,
    tools: dict[str, Any],
    max_steps: int = 8,
    max_tool_calls: int = 6,
) -> dict[str, Any]:
    """Run the ReAct decision loop until final output or a budget is reached."""
    _validate_budget("max_steps", max_steps)
    _validate_budget("max_tool_calls", max_tool_calls)

    messages = build_initial_messages(
        question,
        _describe_tools(tools),
        max_steps,
        max_tool_calls,
    )
    trace: list[dict[str, Any]] = []
    num_llm_calls = 0
    num_tool_calls = 0

    for step_index in range(max_steps):
        step: dict[str, Any] = {
            "step_index": step_index,
            "raw_output": None,
            "action": None,
            "tool_result": None,
            "error": None,
            "elapsed_seconds": 0.0,
        }
        started_at = time.monotonic()

        try:
            # Increment when the logical model call is actually attempted.
            num_llm_calls += 1
            raw_output = model.generate(messages)
            step["raw_output"] = raw_output
            action = parse_action(raw_output)
            step["action"] = action
        except Exception as exc:
            step["error"] = str(exc)
            step["elapsed_seconds"] = time.monotonic() - started_at
            trace.append(step)
            return _result("error", trace, num_llm_calls, num_tool_calls)

        if action["type"] == "final":
            step["elapsed_seconds"] = time.monotonic() - started_at
            trace.append(step)
            return _result(
                "final",
                trace,
                num_llm_calls,
                num_tool_calls,
                answer=action["answer"],
                evidence=action["evidence"],
            )

        # Reaching the limit does not stop the agent until the model asks for
        # another tool. This still permits a later decision step to return final.
        if num_tool_calls >= max_tool_calls:
            step["elapsed_seconds"] = time.monotonic() - started_at
            trace.append(step)
            return _result("tool_limit", trace, num_llm_calls, num_tool_calls)

        # Failed and unknown tools are attempts too, so count before execution.
        num_tool_calls += 1
        tool_result = execute_tool(action, tools)
        step["tool_result"] = tool_result
        messages = append_observation(
            messages,
            raw_output,
            tool_result["observation"],
        )

        step["elapsed_seconds"] = time.monotonic() - started_at
        trace.append(step)

    return _result("max_steps", trace, num_llm_calls, num_tool_calls)
