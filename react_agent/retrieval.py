"""
concrete tools
"""

import re


def tokenize_text(text: str) -> set[str]:
    if not isinstance(text, str):
        raise ValueError(f"text should be a string, got {type(text)}")
    t = text.lower()
    t = re.findall(r"[a-z0-9]+", t)
    return set(t)


def run_search(context: list, query: str, k: int = 3) -> list[str]:
    if type(k) != int or k < 0:
        raise ValueError(f"k should be a positive integer, got {k}")
    
    score = []
    query_tokens = tokenize_text(query)
    for title, sentences in context:
        document = title + " " + " ".join(sentences)
        document_tokens = tokenize_text(document)
        s = len(query_tokens & document_tokens)
        if s > 0:
            score.append((s, title))
    score.sort(key=lambda p: (-p[0], p[1]))
    return [t for _, t in score[:k]]


def run_lookup(context: list, title: str) -> dict:
    if not isinstance(title, str):
        raise ValueError(f"title should be a string, got {type(title)}")
    if not isinstance(context, list):
        raise ValueError(f"context should be a list, got {type(context)}")

    for t, s in context:
        if t == title:
            return {"title": t, "sentences": s}
    raise ValueError(f"title {title} not found in context")
