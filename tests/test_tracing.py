"""Assignment §8: append-only records and explicit resume diagnostics."""
from copy import deepcopy
import json
import warnings

import pytest

from adapters import load_completed_ids, write_jsonl_record


def record(sample_id="a", run_id="run-a"):
    return {"sample_id": sample_id, "run_id": run_id, "status": "final", "answer": "中文\nanswer",
            "evidence": [], "num_llm_calls": 1, "num_tool_calls": 0, "elapsed_seconds": .01,
            "config": {"backend": "scripted", "prompt_version": "v1"},
            "trace": [{"step_index": 0, "raw_output": '{"type":"final","answer":"中文\\nanswer","evidence":[]}',
                       "action": {"type": "final", "answer": "中文\nanswer", "evidence": []},
                       "tool_result": None, "error": None, "elapsed_seconds": .01}]}


def test_utf8_append_one_json_line_per_record(tmp_path):
    path = tmp_path / "trace.jsonl"
    first, second = record(), record("b")
    snapshot = deepcopy(first)
    write_jsonl_record(path, first)
    write_jsonl_record(path, second)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == [first, second]
    assert first == snapshot
    assert load_completed_ids(path, "run-a") == {"a", "b"}


def test_completed_ids_only_for_requested_run(tmp_path):
    path = tmp_path / "trace.jsonl"
    for row in [record("a"), record("b"), record("c", "other-run")]:
        write_jsonl_record(path, row)
    assert load_completed_ids(path, "run-a") == {"a", "b"}
    assert load_completed_ids(path, "other-run") == {"c"}
    assert load_completed_ids(path, "not-present") == set()


def _diagnosed_load(path, expected, caplog, capsys):
    # PDF allows explicit failure OR an explicit repair/skip diagnostic. Neither
    # silent corruption nor silently treating an incomplete sample as done is OK.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = load_completed_ids(path, "run-a")
        except ValueError as exc:
            assert str(exc)
            return
    captured = capsys.readouterr()
    assert caught or caplog.records or captured.err or captured.out, (
        "Corrupt/incomplete/duplicate data must be explicitly diagnosed"
    )
    assert result == expected


@pytest.mark.parametrize("tail", ['{"sample_id":', '{"sample_id":"partial","run_id":"run-a"}\n'])
def test_partial_record_diagnosed_and_not_completed(tmp_path, tail, caplog, capsys):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(record()) + "\n" + tail, encoding="utf-8")
    _diagnosed_load(path, {"a"}, caplog, capsys)


def test_duplicate_sample_run_pair_is_diagnosed(tmp_path, caplog, capsys):
    path = tmp_path / "trace.jsonl"
    row = json.dumps(record())
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    _diagnosed_load(path, {"a"}, caplog, capsys)


def test_failed_terminal_sample_can_be_resumed_without_reexecuting(tmp_path):
    row = record()
    row.update(status="error", answer="", evidence=[])
    row["trace"][0].update(raw_output="bad JSON", action=None, error="invalid action")
    path = tmp_path / "trace.jsonl"
    write_jsonl_record(path, row)
    assert load_completed_ids(path, "run-a") == {"a"}
