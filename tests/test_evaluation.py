"""Assignment §10: fixed splits, official exports, and tiny resumable runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from adapters import make_fixed_splits, make_predictions_json, run_batch


IDS = [f"sample-{index:03d}" for index in range(300)]


def test_sha256_split_order_sizes_and_no_overlap():
    result = make_fixed_splits(list(reversed(IDS)))
    ordered = sorted(IDS, key=lambda value: (hashlib.sha256(f"42:{value}".encode()).hexdigest(), value))
    assert result == {"smoke": ordered[:20], "dev": ordered[20:70], "eval": ordered[70:270]}
    assert len(set(result["smoke"] + result["dev"] + result["eval"])) == 270
    assert make_fixed_splits(IDS) == result
    assert make_fixed_splits(IDS[::2] + IDS[1::2]) == result


def test_seed_is_part_of_digest():
    ordered = sorted(IDS, key=lambda value: (hashlib.sha256(f"7:{value}".encode()).hexdigest(), value))
    assert make_fixed_splits(IDS, seed=7) == {
        "smoke": ordered[:20], "dev": ordered[20:70], "eval": ordered[70:270],
    }


def test_splits_do_not_depend_on_python_hash_seed():
    expected = make_fixed_splits(IDS)
    program = ('import json; from adapters import make_fixed_splits; '
               f'print(json.dumps(make_fixed_splits({IDS!r}), sort_keys=True))')
    for seed in ("1", "123"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        completed = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
                                   env=env, text=True, capture_output=True, check=True, timeout=20)
        assert json.loads(completed.stdout) == expected


@pytest.mark.parametrize("ids", [IDS[:269], IDS[:269] + [IDS[0]], IDS[:269] + [""],
                                  IDS[:269] + [None], IDS[:269] + [42]])
def test_bad_split_input_rejected(ids):
    with pytest.raises(ValueError):
        make_fixed_splits(ids)


def prediction_record(sample_id, status="final"):
    return {"sample_id": sample_id, "status": status, "answer": "London",
            "evidence": [["Lord Byron", 1]], "num_llm_calls": 1, "num_tool_calls": 0, "trace": []}


def test_official_export_includes_failures_and_all_planned_ids():
    records = [prediction_record("ok")]
    records += [prediction_record(status, status) for status in ("error", "max_steps", "tool_limit")]
    ids = ["ok", "error", "max_steps", "tool_limit"]
    result = make_predictions_json(records, ids)
    assert result == {"answer": {"ok": "London", "error": "", "max_steps": "", "tool_limit": ""},
                      "sp": {"ok": [["Lord Byron", 1]], "error": [], "max_steps": [], "tool_limit": []}}
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("records,ids", [
    ([prediction_record("a"), prediction_record("a")], ["a"]),
    ([prediction_record("a")], ["a", "b"]),
    ([prediction_record("a"), prediction_record("b")], ["a"]),
    ([prediction_record("a")], ["a", "a"]),
])
def test_prediction_export_rejects_missing_duplicate_or_foreign_ids(records, ids):
    with pytest.raises(ValueError):
        make_predictions_json(records, ids)


def _batch_inputs(tmp_path):
    data = [{"_id": key, "question": f"Question {key}?",
             "context": [[f"Private {key}", [f"Secret document {key}"]]],
             "answer": f"GOLD_SENTINEL_{key}", "supporting_facts": [[f"Private {key}", 0]],
             "type": "bridge", "level": "easy"} for key in ("a", "b")]
    data_path, ids_path = tmp_path / "data.json", tmp_path / "ids.txt"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    ids_path.write_text("a\nb\n", encoding="utf-8")
    return data_path, ids_path


def test_small_batch_artifacts_failure_denominator_and_resume(tmp_path):
    data_path, ids_path = _batch_inputs(tmp_path)
    calls = []
    class Model:
        def generate(self, messages):
            calls.append(messages)
            text = json.dumps(messages)
            assert "GOLD_SENTINEL_" not in text
            assert "Secret document" not in text
            if any(m["content"] == "Question b?" for m in messages):
                raise RuntimeError("CONTROLLED_BACKEND_FAILURE")
            return '{"type":"final","answer":"London","evidence":[]}'
    def factory(*args, **kwargs):
        return Model()
    output_dir = tmp_path / "run"
    config = {"run_id": "test-run", "backend": "scripted", "prompt_version": "test-v1",
              "max_steps": 2, "max_tool_calls": 1}
    aggregate = run_batch(data_path, ids_path, factory, "react", config, output_dir)
    assert isinstance(aggregate, dict)
    assert len(calls) == 2
    for name in ("config.json", "ids.txt", "predictions.json", "metrics.json", "traces.jsonl"):
        assert (output_dir / name).is_file(), f"Missing required artifact {name}"
    assert (output_dir / "ids.txt").read_text().splitlines() == ["a", "b"]
    predictions = json.loads((output_dir / "predictions.json").read_text())
    assert predictions == {"answer": {"a": "London", "b": ""}, "sp": {"a": [], "b": []}}
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert isinstance(metrics, dict) and metrics
    rows = [json.loads(line) for line in (output_dir / "traces.jsonl").read_text().splitlines()]
    assert {row["sample_id"] for row in rows} == {"a", "b"}
    assert len(rows) == 2
    for row in rows:
        assert row["run_id"] == "test-run"
        assert row["num_llm_calls"] == 1
        assert row["num_tool_calls"] == 0
    assert next(row for row in rows if row["sample_id"] == "b")["status"] == "error"
    # Fully completed rerun must not pay for either success OR failed terminal sample.
    run_batch(data_path, ids_path, factory, "react", config, output_dir)
    assert len(calls) == 2
    resumed = [json.loads(line) for line in (output_dir / "traces.jsonl").read_text().splitlines()]
    assert len(resumed) == 2
    # Simulate interruption after only the first record; only b should run again.
    first = next(row for row in rows if row["sample_id"] == "a")
    (output_dir / "traces.jsonl").write_text(json.dumps(first) + "\n", encoding="utf-8")
    run_batch(data_path, ids_path, factory, "react", config, output_dir)
    assert len(calls) == 3
    resumed = [json.loads(line) for line in (output_dir / "traces.jsonl").read_text().splitlines()]
    assert len(resumed) == 2 and {row["sample_id"] for row in resumed} == {"a", "b"}
