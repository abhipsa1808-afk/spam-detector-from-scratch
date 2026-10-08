"""Classification metrics, written from scratch."""

from __future__ import annotations

SPAM = "spam"


def confusion_counts(actual: list[str], predicted: list[str]) -> dict[str, int]:
    """Count the four possible outcomes, treating spam as the positive class.

    - tp (true positive):  spam correctly caught
    - fp (false positive): normal message wrongly flagged as spam
    - fn (false negative): spam that slipped through
    - tn (true negative):  normal message correctly let through
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for truth, guess in zip(actual, predicted):
        if guess == SPAM:
            counts["tp" if truth == SPAM else "fp"] += 1
        else:
            counts["fn" if truth == SPAM else "tn"] += 1
    return counts


def _divide(top: float, bottom: float) -> float:
    return top / bottom if bottom else 0.0


def report(actual: list[str], predicted: list[str]) -> dict[str, float]:
    """Return accuracy, precision, recall and F1 along with the raw counts."""
    c = confusion_counts(actual, predicted)
    precision = _divide(c["tp"], c["tp"] + c["fp"])
    recall = _divide(c["tp"], c["tp"] + c["fn"])
    return {
        "accuracy": _divide(c["tp"] + c["tn"], len(actual)),
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
        **c,
    }
