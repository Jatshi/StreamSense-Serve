from __future__ import annotations

import re


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalize_words(reference)
    hypothesis_words = normalize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for reference_word in reference_words:
        current = [previous[0] + 1]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous[column - 1] + (reference_word != hypothesis_word)
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference_words)
