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
