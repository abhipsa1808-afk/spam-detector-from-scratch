"""Multinomial Naive Bayes, written from scratch in plain Python."""

from __future__ import annotations

import math
from collections import Counter

from spamdetector.tokenizer import tokenize

SPAM = "spam"
HAM = "ham"


class NaiveBayes:
    """A spam classifier that learns by counting words.

    Training counts how often every word appears in spam and in normal
    messages. To classify a new message, each word votes for spam or ham
    according to those counts, and the votes are added up.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        self.alpha = alpha
        self.vocabulary: set[str] = set()
        self.log_prior: dict[str, float] = {}
        self.log_likelihood: dict[str, dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> NaiveBayes:
        """Learn word statistics from labelled messages."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if set(labels) != {SPAM, HAM}:
            raise ValueError("training data must contain both 'spam' and 'ham'")

        word_counts = {SPAM: Counter(), HAM: Counter()}
        message_counts = Counter(labels)
        for text, label in zip(texts, labels):
            word_counts[label].update(tokenize(text))

        self.vocabulary = set(word_counts[SPAM]) | set(word_counts[HAM])
        vocabulary_size = len(self.vocabulary)

        for label in (SPAM, HAM):
            # Prior: how common is this class overall?
            self.log_prior[label] = math.log(message_counts[label] / len(labels))
            # Likelihood: how likely is each word inside this class?
            # Adding alpha (Laplace smoothing) stops unseen words from
            # getting a probability of exactly zero.
            total = sum(word_counts[label].values()) + self.alpha * vocabulary_size
            self.log_likelihood[label] = {
                word: math.log((word_counts[label][word] + self.alpha) / total)
                for word in self.vocabulary
            }

        self.is_fitted = True
        return self

    def _scores(self, text: str) -> dict[str, float]:
        """Return the log score of each class for one message."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before predicting")
        scores = dict(self.log_prior)
        for word in tokenize(text):
            if word in self.vocabulary:  # words never seen in training are skipped
                for label in (SPAM, HAM):
                    scores[label] += self.log_likelihood[label][word]
        return scores

    def predict_proba(self, text: str) -> float:
        """Return the probability (0 to 1) that a message is spam."""
        scores = self._scores(text)
        difference = scores[HAM] - scores[SPAM]
        if difference > 700:  # avoids overflow in math.exp
            return 0.0
        return 1.0 / (1.0 + math.exp(difference))

    def predict(self, text: str, threshold: float = 0.5) -> str:
        """Return 'spam' or 'ham' for one message."""
        return SPAM if self.predict_proba(text) >= threshold else HAM

    def explain(self, text: str) -> list[tuple[str, float]]:
        """Return each known word with its push towards spam (+) or ham (-).

        The list is sorted with the strongest evidence first.
        """
        if not self.is_fitted:
            raise RuntimeError("call fit() before explaining")
        evidence: dict[str, float] = {}
        for word in tokenize(text):
            if word in self.vocabulary:
                push = self.log_likelihood[SPAM][word] - self.log_likelihood[HAM][word]
                evidence[word] = evidence.get(word, 0.0) + push
        return sorted(evidence.items(), key=lambda item: abs(item[1]), reverse=True)
