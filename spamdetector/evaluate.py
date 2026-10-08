"""Evaluate the from-scratch model and compare it with scikit-learn.

Run with:  python -m spamdetector.evaluate
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from spamdetector.data import load, train_test_split
from spamdetector.metrics import report
from spamdetector.model import HAM, SPAM, NaiveBayes
from spamdetector.tokenizer import tokenize


def run() -> dict[str, dict[str, float]]:
    """Train every model, print a comparison table and return the numbers."""
    train, test = train_test_split(load())
    train_texts = [text for _, text in train]
    train_labels = [label for label, _ in train]
    test_texts = [text for _, text in test]
    test_labels = [label for label, _ in test]

    # 1. A "model" that always says ham. Anything useful has to beat this.
    baseline = [HAM] * len(test_texts)

    # 2. Our own Naive Bayes.
    ours = NaiveBayes().fit(train_texts, train_labels)
    our_probabilities = [ours.predict_proba(text) for text in test_texts]
    our_predictions = [SPAM if p >= 0.5 else HAM for p in our_probabilities]

    # 3. scikit-learn's Naive Bayes, given the same tokenizer and settings.
    vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    reference = MultinomialNB(alpha=1.0)
    reference.fit(vectorizer.fit_transform(train_texts), train_labels)
    test_matrix = vectorizer.transform(test_texts)
    spam_column = list(reference.classes_).index(SPAM)
    reference_probabilities = reference.predict_proba(test_matrix)[:, spam_column]
    reference_predictions = list(reference.predict(test_matrix))

    results = {
        "Always 'ham' (baseline)": report(test_labels, baseline),
        "Naive Bayes (ours)": report(test_labels, our_predictions),
        "scikit-learn MultinomialNB": report(test_labels, reference_predictions),
    }

    print(f"Trained on {len(train)} messages, tested on {len(test)} unseen messages\n")
    print(f"{'Model':28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    print("-" * 66)
    for name, r in results.items():
        print(
            f"{name:28}{r['accuracy']:>10.2%}{r['precision']:>11.2%}"
            f"{r['recall']:>9.2%}{r['f1']:>8.2%}"
        )

    r = results["Naive Bayes (ours)"]
    print("\nWhat our model did with the test messages:")
    print(f"  spam caught               : {r['tp']}")
    print(f"  spam missed               : {r['fn']}")
    print(f"  normal let through        : {r['tn']}")
    print(f"  normal wrongly flagged    : {r['fp']}")

    agree = sum(a == b for a, b in zip(our_predictions, reference_predictions))
    gap = max(abs(a - b) for a, b in zip(our_probabilities, reference_probabilities))
    print("\nOurs vs scikit-learn:")
    print(f"  same answer on            : {agree}/{len(test)} messages")
    print(f"  largest probability gap   : {gap:.1e}")
    return results


if __name__ == "__main__":
    run()
