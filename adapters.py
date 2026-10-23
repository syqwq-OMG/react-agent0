"""CS336-style test adapters.

Every function below is a contract, NOT an implementation.
Write the actual code inside react_agent/ and forward arguments here.
Do not put substantive business logic into this file.
"""
from typing import Any


def parse_action(text: str) -> dict[str, Any]:
    """Parse one STRICT JSON action: tool_call or final. Raise ValueError on invalid input."""
    raise NotImplementedError


def execute_tool(action: dict[str, Any], tools: dict[str, Any]) -> dict[str, Any]:
    """Return {'ok': bool, 'observation': str}; never execute unregistered tools."""
    raise NotImplementedError


def run_search(context: list, query: str, k: int = 3) -> list[str]:
    """Search one question's [title, sentence-list] candidate paragraphs; return titles."""
    raise NotImplementedError


def run_lookup(context: list, title: str) -> dict[str, Any]:
    """Return {'title': str, 'sentences': list[str]}, preserving sentence order."""
    raise NotImplementedError


def run_agent(
    question: str,
    model: Any,
    tools: dict[str, Any],
    max_steps: int = 8,
    max_tool_calls: int = 6,
) -> dict[str, Any]:
    """Model must expose generate(messages)->str. Result has status/answer/evidence/trace/num_tool_calls."""
    raise NotImplementedError


def normalize_answer(answer: str) -> str:
    """Normalize as in the official English HotpotQA scorer."""
    raise NotImplementedError


def score_answer(prediction: str, gold: str) -> dict[str, float]:
    """Return {'em':float, 'f1':float, 'precision':float, 'recall':float}."""
    raise NotImplementedError


def score_support(predicted: list, gold: list) -> dict[str, float]:
    """Return {'em':float, 'f1':float, 'precision':float, 'recall':float}."""
    raise NotImplementedError
