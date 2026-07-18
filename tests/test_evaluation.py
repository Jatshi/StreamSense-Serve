from streamsense.evaluation import normalize_words, word_error_rate


def test_word_error_rate_exact_match_ignores_case_and_punctuation() -> None:
    assert word_error_rate("Hello, WORLD!", "hello world") == 0.0


def test_word_error_rate_counts_substitution_insertion_and_deletion() -> None:
    assert word_error_rate("a b c", "a x c") == 1 / 3
    assert word_error_rate("a b", "a b c") == 1 / 2
    assert word_error_rate("a b c", "a c") == 1 / 3


def test_empty_reference_behavior() -> None:
    assert normalize_words("...") == []
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "unexpected") == 1.0
