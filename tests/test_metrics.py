import pytest

from spamdetector.metrics import confusion_counts, report

ACTUAL = ["spam", "spam", "spam", "ham", "ham", "ham", "ham", "ham"]
PREDICTED = ["spam", "spam", "ham", "spam", "ham", "ham", "ham", "ham"]


def test_confusion_counts():
    assert confusion_counts(ACTUAL, PREDICTED) == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}


def test_report_values():
    r = report(ACTUAL, PREDICTED)
    assert r["accuracy"] == pytest.approx(6 / 8)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_perfect_predictions_score_one():
    r = report(ACTUAL, ACTUAL)
    assert r["accuracy"] == r["precision"] == r["recall"] == r["f1"] == 1.0


def test_never_predicting_spam_gives_zero_not_an_error():
    r = report(ACTUAL, ["ham"] * len(ACTUAL))
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0


def test_different_lengths_are_rejected():
    with pytest.raises(ValueError):
        confusion_counts(["spam"], ["spam", "ham"])
