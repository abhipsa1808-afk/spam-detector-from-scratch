"""Streamlit demo: type a message and see whether the model thinks it is spam.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import html

import streamlit as st

from spamdetector.data import load, train_test_split
from spamdetector.metrics import report
from spamdetector.model import HAM, SPAM, NaiveBayes
from spamdetector.tokenizer import tokenize

REPO_URL = "https://github.com/abhipsa1808-afk/spam-detector-from-scratch"
SPAM_RGB = "235, 104, 52"  # orange-red
HAM_RGB = "42, 120, 214"  # blue

EXAMPLES = {
    "Prize scam": (
        "WINNER!! You have been selected to receive a £900 prize reward! "
        "To claim call 09061701461. Claim code KL341. Valid 12 hours only."
    ),
    "Message from a friend": (
        "Hey, are we still meeting for lunch at 1 tomorrow? I can book the table."
    ),
    "Fake bank alert": (
        "URGENT: your account has been suspended. Verify your details now at "
        "http://secure-login.example.com to avoid charges."
    ),
    "Tricky one": "Free tonight? Call me when you finish work.",
}


@st.cache_resource
def train_model():
    """Train once and keep the result in memory between page reloads."""
    train_rows, test_rows = train_test_split(load())
    model = NaiveBayes().fit([text for _, text in train_rows], [label for label, _ in train_rows])
    test_labels = [label for label, _ in test_rows]
    test_probabilities = [model.predict_proba(text) for _, text in test_rows]
    return model, len(train_rows), test_labels, test_probabilities


def highlight(model: NaiveBayes, message: str) -> str:
    """Return the message as HTML, with each word shaded by how it voted."""
    pieces = []
    for chunk in message.split():
        push = sum(score for _, score in model.explain(chunk))
        safe = html.escape(chunk)
        if abs(push) < 0.5:
            pieces.append(safe)
            continue
        strength = min(abs(push) / 6, 1.0)
        color = SPAM_RGB if push > 0 else HAM_RGB
        side = "spam" if push > 0 else "not spam"
        pieces.append(
            f'<span title="pushes towards {side} ({push:+.1f})" style="background: '
            f"rgba({color}, {0.15 + 0.55 * strength:.2f}); border-radius: 4px; "
            f'padding: 1px 4px;">{safe}</span>'
        )
    return '<div style="line-height: 2; font-size: 1.05rem;">' + " ".join(pieces) + "</div>"


st.set_page_config(page_title="Spam Detector From Scratch", page_icon="📨")
model, train_size, test_labels, test_probabilities = train_model()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Spam threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="A message is called spam when its spam probability reaches this value.",
    )
    st.header("Scores on unseen messages")
    predictions = [SPAM if p >= threshold else HAM for p in test_probabilities]
    scores = report(test_labels, predictions)
    left, right = st.columns(2)
    left.metric("Accuracy", f"{scores['accuracy']:.1%}")
    right.metric("F1", f"{scores['f1']:.1%}")
    left.metric("Precision", f"{scores['precision']:.1%}")
    right.metric("Recall", f"{scores['recall']:.1%}")
    st.caption(
        f"Trained on {train_size:,} messages and tested on {len(test_labels):,} it had "
        f"never seen. It knows {len(model.vocabulary):,} words. Move the threshold to "
        "watch precision and recall trade off against each other."
    )

st.title("Spam Detector From Scratch")
st.write(
    "A Naive Bayes classifier written in plain Python, with no machine learning "
    "library doing the work. Type a text message, or pick an example."
)

if "message" not in st.session_state:
    st.session_state.message = EXAMPLES["Prize scam"]

for column, (name, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
    if column.button(name):
        st.session_state.message = text

message = st.text_area("Message", key="message", height=120)

if not message.strip():
    st.info("Type a message above to see the prediction.")
else:
    probability = model.predict_proba(message)
    if probability >= threshold:
        st.error(f"**Spam.** The model gives this a {probability:.1%} chance of being spam.")
    else:
        st.success(f"**Not spam.** The model gives this a {probability:.1%} chance of being spam.")
    st.progress(probability, text="Spam probability")

    st.subheader("Why the model decided this")
    st.write(
        "Each word votes. Orange words push towards spam and blue words push towards "
        "not spam. A stronger colour means a stronger vote."
    )
    st.markdown(highlight(model, message), unsafe_allow_html=True)

    evidence = model.explain(message)
    if evidence:
        st.dataframe(
            [
                {
                    "Word": word,
                    "Pushes towards": "spam" if score > 0 else "not spam",
                    "Strength": round(abs(score), 2),
                }
                for word, score in evidence[:8]
            ],
            hide_index=True,
        )

    unknown = sorted({token for token in tokenize(message) if token not in model.vocabulary})
    if unknown:
        st.caption(
            "Words the model never saw in training, so it ignores them: " + ", ".join(unknown)
        )

with st.expander("How it works"):
    st.markdown(
        """
1. **Training.** The model counts how often every word appears in spam and in
   normal messages from the SMS Spam Collection dataset.
2. **Voting.** For a new message, each word votes for spam or not spam according
   to those counts. The votes are added up and turned into a probability.
3. **Smoothing.** One is added to every count, so a word that never appeared in
   spam does not get a probability of zero and overrule everything else.
4. **Checking.** The same data is given to scikit-learn's Naive Bayes, and the two
   models agree on every test message.
"""
    )

st.caption(f"Built by Abhipsa Rout. Code, tests and explanation: [GitHub]({REPO_URL})")
