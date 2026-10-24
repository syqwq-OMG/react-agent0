"""Local retrieval tools used by the agent."""

import re


def tokenize_text(text: str) -> set[str]:
    if not isinstance(text, str):
        raise ValueError(f"text should be a string, got {type(text)}")
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def run_search(context: list, query: str, k: int = 3) -> list[str]:
    if type(k) is not int or k < 0:
        raise ValueError(f"k should be a positive integer, got {k}")

    scores = []
    query_tokens = tokenize_text(query)
    for title, sentences in context:
        document = title + " " + " ".join(sentences)
        document_tokens = tokenize_text(document)
        score = len(query_tokens & document_tokens)
        if score > 0:
            scores.append((score, title))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [title for _, title in scores[:k]]


def run_lookup(context: list, title: str) -> dict:
    for t, s in context:
        if t == title:
            return {"title": t, "sentences": s}
    raise ValueError(f"title {title} not found in context")
