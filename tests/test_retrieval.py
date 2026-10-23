"""Assignment §4: pure retrieval contracts, independent of any model."""
from copy import deepcopy

import pytest

from adapters import run_lookup, run_search, tokenize_text


@pytest.mark.parametrize("text, expected", [
    ("British poet", {"british", "poet"}),
    ("Ada ADA ada 123 12.3", {"ada", "123", "12", "3"}),
    ("a_b-Café 中文", {"a", "b", "caf"}),
    ("", set()), ("!? 中文", set()),
])
def test_tokenize_exact_rule(text, expected):
    assert tokenize_text(text) == expected
    assert isinstance(tokenize_text(text), set)


@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_tokenize_rejects_non_string(value):
    with pytest.raises(ValueError):
        tokenize_text(value)


def test_ranking_uses_sets_body_and_title():
    context = [["Z", ["poet poet poet"]], ["A", ["poet"]],
               ["British", ["poet"]], ["No hit", ["plants"]]]
    assert run_search(context, "BRITISH poet poet", 10) == ["British", "A", "Z"]


@pytest.mark.parametrize("query,k", [("", 3), ("!!!", 3), ("absentword", 3), ("Ada", 0)])
def test_empty_search_result(toy_context, query, k):
    assert run_search(toy_context, query, k) == []


def test_search_empty_context_and_topk(toy_context):
    assert run_search([], "Ada", 3) == []
    assert run_search(toy_context, "Ada Lord Byron", 1) == ["Ada Lovelace"]


@pytest.mark.parametrize("k", [-1, True, False, 1.5, "3", None])
def test_invalid_k(toy_context, k):
    with pytest.raises(ValueError):
        run_search(toy_context, "Ada", k)


@pytest.mark.parametrize("query", [None, 7, [], {}])
def test_invalid_query(toy_context, query):
    with pytest.raises(ValueError):
        run_search(toy_context, query)


@pytest.mark.parametrize("title", ["missing", "ada lovelace", "Ada Lovelace ", None, 1, []])
def test_lookup_requires_exact_string_title(toy_context, title):
    with pytest.raises(ValueError):
        run_lookup(toy_context, title)


def test_retrieval_does_not_mutate_original_sentences(toy_context):
    before = deepcopy(toy_context)
    run_search(toy_context, "Byron", 3)
    found = run_lookup(toy_context, "Ada Lovelace")
    assert found == {"title": "Ada Lovelace", "sentences": before[0][1]}
    assert toy_context == before
