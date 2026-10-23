"""Thin public test API for react-agent-assignment.pdf (v4.0).

Implement the functions in the indicated react_agent modules. These adapters only
forward arguments; they deliberately contain no assignment solutions. Imports are
lazy so each milestone can be tested before later modules exist. If you choose a
different internal layout, change the corresponding forwarding target here.
"""
from importlib import import_module
from typing import Any


def _implementation(module: str, name: str):
    target = f"react_agent.{module}"
    try:
        implementation = import_module(target)
    except ModuleNotFoundError as exc:
        if exc.name != target:
            raise  # A missing dependency inside student code is a real error.
        raise NotImplementedError(f"Implement {target}.{name} first") from exc
    function = getattr(implementation, name, None)
    if function is None:
        raise NotImplementedError(f"Implement {target}.{name} first")
    return function


def parse_action(text: str) -> dict[str, Any]:
    """§3: strict single JSON action; invalid text/schema raises ValueError."""
    return _implementation("protocol", "parse_action")(text)


def tokenize_text(text: str) -> set[str]:
    """§4: lowercase [a-z0-9]+ token set; non-string raises ValueError."""
    return _implementation("retrieval", "tokenize_text")(text)


def run_search(context: list, query: str, k: int = 3) -> list[str]:
    """§4: positive overlap, descending score then ascending original title."""
    return _implementation("retrieval", "run_search")(context, query, k=k)


def run_lookup(context: list, title: str) -> dict[str, Any]:
    """§4: exact title match; preserve original sentences and their indices."""
    return _implementation("retrieval", "run_lookup")(context, title)


def execute_tool(
    action: dict[str, Any], tools: dict[str, Any], max_observation_chars: int = 8000,
) -> dict[str, Any]:
    """§5: registered calls only; failures yield bounded ERROR: observations."""
    return _implementation("tools", "execute_tool")(
        action, tools, max_observation_chars=max_observation_chars,
    )


def make_tools(context: list) -> dict[str, Any]:
    """§5: bind search/lookup to this question's context."""
    return _implementation("tools", "make_tools")(context)


def build_initial_messages(
    question: str, tool_descriptions: str, max_steps: int, max_tool_calls: int,
) -> list[dict]:
    """§6: system protocol/tools/budgets followed by the user question."""
    return _implementation("prompting", "build_initial_messages")(
        question, tool_descriptions, max_steps=max_steps, max_tool_calls=max_tool_calls,
    )


def append_observation(
    messages: list[dict], raw_action: str, observation: str,
) -> list[dict]:
    """§6: return a new history with assistant action then user Observation:."""
    return _implementation("prompting", "append_observation")(messages, raw_action, observation)


def run_agent(
    question: str, model: Any, tools: dict[str, Any],
    max_steps: int = 8, max_tool_calls: int = 6,
) -> dict[str, Any]:
    """§6: return status/answer/evidence/trace and both actual call counts."""
    return _implementation("agent", "run_agent")(
        question, model, tools, max_steps=max_steps, max_tool_calls=max_tool_calls,
    )


def make_model(
    backend: str, model_id: str, base_url: str,
    api_key: str | None = None, timeout: float = 60.0,
) -> Any:
    """§7: deepseek/ollama transport exposing generate(messages) -> str."""
    return _implementation("providers", "make_model")(
        backend, model_id, base_url, api_key=api_key, timeout=timeout,
    )


def write_jsonl_record(path: Any, record: dict) -> None:
    """§8: append one complete UTF-8 sample record as one JSON line."""
    return _implementation("tracing", "write_jsonl_record")(path, record)


def load_completed_ids(path: Any, run_id: str) -> set[str]:
    """§8: complete sample IDs for this run; diagnose corrupt/duplicate rows."""
    return _implementation("tracing", "load_completed_ids")(path, run_id)


def load_hotpot(path: Any) -> list[dict]:
    """§9: validate samples, unique IDs, per-sample titles and sentence types."""
    return _implementation("dataset", "load_hotpot")(path)


def split_example(example: dict) -> tuple[dict, dict]:
    """§9: separate runtime _id/question/context from evaluator gold/metadata."""
    return _implementation("dataset", "split_example")(example)


def normalize_answer(answer: str) -> str:
    """§10: official HotpotQA English normalization; reject non-strings."""
    return _implementation("metrics", "normalize_answer")(answer)


def score_answer(prediction: str, gold: str) -> dict[str, float]:
    """§10: answer em/f1/precision/recall in [0, 1], including special answers."""
    return _implementation("metrics", "score_answer")(prediction, gold)


def score_support(predicted: list, gold: list) -> dict[str, float]:
    """§10: exact-title sentence sets; duplicate/order invariant SP scores."""
    return _implementation("metrics", "score_support")(predicted, gold)


def make_fixed_splits(ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    """§10: SHA256(f'{seed}:{id}') order, slices 0:20 / 20:70 / 70:270."""
    return _implementation("evaluation", "make_fixed_splits")(ids, seed=seed)


def make_predictions_json(records: list[dict], expected_ids: list[str]) -> dict:
    """§10: official answer/sp mappings; non-final samples get empty predictions."""
    return _implementation("evaluation", "make_predictions_json")(records, expected_ids)


def run_batch(
    data_path: Any, ids_path: Any, model_factory: Any, method: str,
    config: dict, output_dir: Any,
) -> dict:
    """§10: isolated per-question runs, complete artifacts, deduplicated resume."""
    return _implementation("evaluation", "run_batch")(
        data_path, ids_path, model_factory, method, config, output_dir,
    )
