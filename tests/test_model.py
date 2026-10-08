import math

import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from spamdetector.model import NaiveBayes
from spamdetector.tokenizer import tokenize

TEXTS = [
    "win a free prize now",
    "free cash call now",
    "claim your free prize",
    "are we meeting for lunch",
    "see you at home later",
    "call me when you are free",
    "lunch tomorrow at noon",
]
LABELS = ["spam", "spam", "spam", "ham", "ham", "ham", "ham"]


@pytest.fixture
def model():
    return NaiveBayes().fit(TEXTS, LABELS)


def test_classifies_obvious_messages(model):
    assert model.predict("free prize, claim now") == "spam"
    assert model.predict("see you at lunch") == "ham"


def test_probability_is_between_zero_and_one(model):
    for text in TEXTS + ["", "completely unknown words", "free " * 500, "lunch " * 500]:
        assert 0.0 <= model.predict_proba(text) <= 1.0


def test_unknown_words_fall_back_to_the_prior(model):
    # 3 of the 7 training messages are spam, so with no evidence the answer is 3/7.
    assert model.predict_proba("zzz qqq") == pytest.approx(3 / 7)
    assert model.predict_proba("") == pytest.approx(3 / 7)


def test_matches_a_calculation_done_by_hand():
    model = NaiveBayes().fit(["free prize", "free lunch"], ["spam", "ham"])
    # Vocabulary: free, prize, lunch (3 words). Each class has 2 words.
    # P(free | spam) = (1 + 1) / (2 + 3), P(prize | spam) = (1 + 1) / (2 + 3)
    # P(free | ham)  = (1 + 1) / (2 + 3), P(prize | ham)  = (0 + 1) / (2 + 3)
    spam = 0.5 * (2 / 5) * (2 / 5)
    ham = 0.5 * (2 / 5) * (1 / 5)
    assert model.predict_proba("free prize") == pytest.approx(spam / (spam + ham))


def test_smoothing_stops_one_unseen_word_from_deciding_everything(model):
    # "lunch" never appears in spam. Without smoothing its spam probability
    # would be zero and this message could never be classified as spam.
    assert model.predict("free prize free cash claim now lunch") == "spam"


def test_threshold_changes_the_decision(model):
    probability = model.predict_proba("call me")
    assert model.predict("call me", threshold=probability + 0.01) == "ham"
    assert model.predict("call me", threshold=probability - 0.01) == "spam"


def test_explain_points_the_right_way(model):
    evidence = dict(model.explain("free prize for lunch"))
    assert evidence["prize"] > 0
    assert evidence["lunch"] < 0
    strengths = [abs(score) for _, score in model.explain("free prize for lunch")]
    assert strengths == sorted(strengths, reverse=True)


def test_explanation_adds_up_to_the_prediction(model):
    text = "claim your free lunch now"
    log_odds = math.log(3 / 4) + sum(score for _, score in model.explain(text))
    assert model.predict_proba(text) == pytest.approx(1 / (1 + math.exp(-log_odds)))


def test_matches_scikit_learn(model):
    vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    reference = MultinomialNB(alpha=1.0).fit(vectorizer.fit_transform(TEXTS), LABELS)
    spam_column = list(reference.classes_).index("spam")
    messages = TEXTS + ["free lunch", "call now", "hello there", "prize prize prize"]
    expected = reference.predict_proba(vectorizer.transform(messages))[:, spam_column]
    for message, probability in zip(messages, expected):
        assert model.predict_proba(message) == pytest.approx(probability, abs=1e-12)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        NaiveBayes(alpha=0)
    with pytest.raises(ValueError):
        NaiveBayes().fit(["a", "b"], ["spam"])
    with pytest.raises(ValueError):
        NaiveBayes().fit(["a", "b"], ["ham", "ham"])
    with pytest.raises(RuntimeError):
        NaiveBayes().predict("not trained yet")
