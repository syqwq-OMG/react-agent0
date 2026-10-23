"""Assignment §9: tiny synthetic data only; no benchmark download needed."""
from copy import deepcopy
import json

import pytest

from adapters import load_hotpot, make_tools, split_example


def example(sample_id="sample-a"):
    return {"_id": sample_id, "question": "PUBLIC_QUESTION",
            "context": [["PRIVATE_TITLE", ["FIRST_SENTENCE", "SECOND_SENTENCE"]]],
            "answer": "GOLD_ANSWER_SENTINEL", "supporting_facts": [["PRIVATE_TITLE", 1]],
            "type": "bridge", "level": "easy"}


def _write(tmp_path, value):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_preserves_ids_titles_sentences_and_cross_question_titles(tmp_path):
    records = [example("a"), example("b")]
    assert load_hotpot(_write(tmp_path, records)) == records


@pytest.mark.parametrize("value", [{}, None, "text", [None], [7]])
def test_invalid_dataset_container(tmp_path, value):
    with pytest.raises(ValueError):
        load_hotpot(_write(tmp_path, value))


@pytest.mark.parametrize("field", ["_id", "question", "context", "answer", "supporting_facts"])
def test_missing_required_field(tmp_path, field):
    record = example()
    del record[field]
    with pytest.raises(ValueError):
        load_hotpot(_write(tmp_path, [record]))


@pytest.mark.parametrize("field,value", [
    ("_id", ""), ("_id", None), ("_id", 1),
    ("question", None), ("question", []), ("context", {}),
    ("context", [["T"]]), ("context", [[1, ["sentence"]]]),
    ("context", [["T", "not a sentence list"]]),
    ("context", [["T", [1]]]),
    ("context", [["T", ["one"]], ["T", ["two"]]]),
])
def test_invalid_sample_fields(tmp_path, field, value):
    record = example()
    record[field] = value
    with pytest.raises(ValueError):
        load_hotpot(_write(tmp_path, [record]))


def test_duplicate_sample_id_is_not_silently_dropped(tmp_path):
    with pytest.raises(ValueError):
        load_hotpot(_write(tmp_path, [example(), example()]))


def test_split_physically_separates_runtime_and_gold():
    original = example()
    before = deepcopy(original)
    runtime, gold = split_example(original)
    assert set(runtime) == {"_id", "question", "context"}
    assert runtime == {key: original[key] for key in runtime}
    assert gold["_id"] == original["_id"]
    assert gold["answer"] == original["answer"]
    assert gold["supporting_facts"] == original["supporting_facts"]
    assert gold["type"] == "bridge" and gold["level"] == "easy"
    assert set(gold) <= {"_id", "answer", "supporting_facts", "type", "level"}
    assert "GOLD_ANSWER_SENTINEL" not in json.dumps(runtime)
    assert original == before


def test_tools_created_from_runtime_cannot_reveal_gold():
    runtime, _ = split_example(example())
    tools = make_tools(runtime["context"])
    observation = tools["lookup"](title="PRIVATE_TITLE")
    assert observation["sentences"][1] == "SECOND_SENTENCE"
    assert "GOLD_ANSWER_SENTINEL" not in json.dumps(observation)
