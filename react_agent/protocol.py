import json
from typing import Any


def parse_action(text: str) -> dict[str, Any]:
    """
    Parse the action from the llm output
    """
    if not isinstance(text, str):
        raise ValueError(f"text should be a string, got {type(text)}")
    try:
        action: dict = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json: {e}")

    if not isinstance(action, dict):
        raise ValueError(f"action{action} should be a dict")

    action_type = action.get("type")

    if action_type not in ["tool_call", "final"]:
        raise ValueError(f"invalid action type: {action_type}")

    if action_type == "tool_call":
        tool = action.get("tool")
        args = action.get("args")
        if not isinstance(tool, str) or tool.strip() == "":
            raise ValueError(f"invalid tool: {tool}")
        if not isinstance(args, dict):
            raise ValueError(f"invalid args: {args}")
    else:
        answer = action.get("answer")
        evidence = action.get("evidence", [])
        if not isinstance(answer, str):
            raise ValueError(f"invalid answer: {answer}")
        if not isinstance(evidence, list):
            raise ValueError(f"invalid evidence: {evidence}")
        for e in evidence:
            if not isinstance(e, list) or len(e) != 2:
                raise ValueError(f"invalid evidence item: {e}")
            title, sentence_id = e
            if not isinstance(title, str) or not title:
                raise ValueError(f"invalid evidence title: {title}")
            if type(sentence_id) != int or sentence_id < 0:
                raise ValueError(f"invalid evidence sentence_id: {sentence_id}")
        action["evidence"] = evidence

    note = action.get("note")

    if "note" in action and not isinstance(note, str):
        raise ValueError(f"invalid note: {note}")

    return action
