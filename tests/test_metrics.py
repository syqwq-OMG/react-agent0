import pytest
from adapters import normalize_answer, score_answer, score_support


def test_official_normalization_style():
    assert normalize_answer("The QUICK, Brown Fox!") == "quick brown fox"
    assert normalize_answer("An apple") == "apple"


def test_exact_match_ignores_articles_and_punctuation():
    result = score_answer("The Lord Byron!", "lord byron")
    assert result["em"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_answer_partial_overlap():
    result = score_answer("Ada Byron", "Ada Lovelace Byron")
    assert result["em"] == pytest.approx(0.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(2.0 / 3.0)
    assert result["f1"] == pytest.approx(0.8)


def test_yes_no_mismatch_zero():
    result = score_answer("yes", "no")
    assert result["em"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_supporting_fact_exact():
    facts = [["Ada Lovelace", 1], ["Lord Byron", 0]]
    result = score_support(facts, list(reversed(facts)))
    assert result["em"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_supporting_fact_partial():
    result = score_support([["Ada Lovelace", 1]], [["Ada Lovelace", 1], ["Lord Byron", 0]])
    assert result["em"] == pytest.approx(0.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("text,expected", [
    ("  THE\tAn a  apple\n", "apple"),
    ("theater and another", "theater and another"),
    ("foo-bar", "foobar"),  # Official scorer deletes punctuation, not replaces it.
    ("Ada's", "adas"),
    ("", ""), ("A!", ""),
])
def test_normalization_order_and_word_boundaries(text, expected):
    assert normalize_answer(text) == expected


@pytest.mark.parametrize("value", [None, 10, True, [], {}])
def test_normalize_rejects_non_string(value):
    with pytest.raises(ValueError):
        normalize_answer(value)


@pytest.mark.parametrize("prediction,gold,expected", [
    ("red red blue", "red blue blue", {"em": 0, "precision": 2/3, "recall": 2/3, "f1": 2/3}),
    ("", "", {"em": 1, "precision": 0, "recall": 0, "f1": 0}),
    ("", "London", {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ("London", "", {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ("yes", "yes indeed", {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ("yes indeed", "yes", {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ("noanswer", "noanswer maybe", {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ("no", "no way", {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ("Yes!", "yes", {"em": 1, "precision": 1, "recall": 1, "f1": 1}),
])
def test_official_answer_edge_cases(prediction, gold, expected):
    result = score_answer(prediction, gold)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
        assert 0 <= result[key] <= 1


@pytest.mark.parametrize("prediction,gold,expected", [
    ([], [], {"em": 1, "precision": 0, "recall": 0, "f1": 0}),
    ([], [["T", 0]], {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ([["T", 0]], [], {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ([["T", 0], ["T", 0]], [["T", 0]], {"em": 1, "precision": 1, "recall": 1, "f1": 1}),
    ([["T", 0]], [["T", 0], ["T", 0]], {"em": 1, "precision": 1, "recall": 1, "f1": 1}),
    ([["the Title", 0]], [["The Title", 0]], {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ([["T", 0]], [["T", 1]], {"em": 0, "precision": 0, "recall": 0, "f1": 0}),
    ([["T", 0], ["X", 0]], [["T", 0]], {"em": 0, "precision": .5, "recall": 1, "f1": 2/3}),
])
def test_support_set_edge_cases(prediction, gold, expected):
    result = score_support(prediction, gold)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
