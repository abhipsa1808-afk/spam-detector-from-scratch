"""Draw the charts used in the README.

Run with:  python -m spamdetector.plots
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from spamdetector.data import load, train_test_split
from spamdetector.metrics import report
from spamdetector.model import HAM, SPAM, NaiveBayes
from spamdetector.tokenizer import tokenize

ASSETS = Path(__file__).resolve().parent.parent / "assets"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # normal messages, precision
ORANGE = "#eb6834"  # spam, recall

READABLE = {
    "<longnum>": "(long number)",
    "<num>": "(short number)",
    "<money>": "(currency symbol)",
    "<url>": "(web link)",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def _clean(ax) -> None:
    """Remove the box around a chart and keep only a quiet baseline."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def threshold_chart(labels: list[str], probabilities: list[float]) -> Path:
    """Precision and recall for every possible spam threshold."""
    thresholds = [t / 100 for t in range(1, 100)]
    precision, recall = [], []
    for threshold in thresholds:
        predicted = [SPAM if p >= threshold else HAM for p in probabilities]
        scores = report(labels, predicted)
        precision.append(scores["precision"] * 100)
        recall.append(scores["recall"] * 100)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    _clean(ax)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.axvline(0.5, color=AXIS, linewidth=1)
    ax.plot(thresholds, precision, color=BLUE, linewidth=2, label="Precision")
    ax.plot(thresholds, recall, color=ORANGE, linewidth=2, label="Recall")

    at_default = report(labels, [SPAM if p >= 0.5 else HAM for p in probabilities])
    lowest = min(precision + recall)
    ax.set_ylim(max(0, lowest - 8), 101)
    ax.set_xlim(0, 1)
    ax.annotate(
        f"default threshold 0.5\nprecision {at_default['precision']:.1%}, "
        f"recall {at_default['recall']:.1%}",
        (0.5, ax.get_ylim()[0]),
        xytext=(8, 8),
        textcoords="offset points",
        color=SECONDARY,
        fontsize=10,
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.set_xlabel("Spam threshold (probability needed to call a message spam)")
    fig.suptitle(
        "A higher threshold flags fewer normal messages but lets more spam through",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.0),
        ncol=2,
        frameon=False,
        labelcolor=SECONDARY,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14)
    path = ASSETS / "threshold_tradeoff.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def words_chart(model: NaiveBayes, counts: Counter, minimum_count: int = 25) -> Path:
    """The words that most strongly point to spam and to normal messages."""
    pushes = {
        word: model.log_likelihood[SPAM][word] - model.log_likelihood[HAM][word]
        for word in model.vocabulary
        if counts[word] >= minimum_count
    }
    ranked = sorted(pushes, key=pushes.get)
    panels = (
        ("Strongest signs of spam", ranked[-10:][::-1], ORANGE),
        ("Strongest signs of a normal message", ranked[:10], BLUE),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (title, words, color) in zip(axes, panels):
        sizes = [abs(pushes[word]) for word in words]
        ax.barh(range(len(words)), sizes, height=0.55, color=color)
        for position, word in enumerate(words):
            ax.annotate(
                f"{math.exp(abs(pushes[word])):,.0f}x",
                (sizes[position], position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=SECONDARY,
                fontsize=10,
            )
        ax.set_yticks(range(len(words)), [READABLE.get(word, word) for word in words])
        ax.tick_params(axis="y", labelcolor=INK, length=0)
        ax.set_xticks([])
        ax.set_xlim(0, max(sizes) * 1.18)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.invert_yaxis()  # strongest word at the top
    fig.suptitle(
        "What the model learned from counting words",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.885,
        "Each number shows how many times more likely the word is in that kind of message. "
        f"Words seen fewer than {minimum_count} times are left out.",
        color=SECONDARY,
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.16, right=0.97, top=0.76, bottom=0.04, wspace=0.5)
    path = ASSETS / "top_words.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run() -> None:
    ASSETS.mkdir(exist_ok=True)
    train, test = train_test_split(load())
    train_texts = [text for _, text in train]
    model = NaiveBayes().fit(train_texts, [label for label, _ in train])
    counts = Counter(token for text in train_texts for token in tokenize(text))
    probabilities = [model.predict_proba(text) for _, text in test]

    for path in (
        threshold_chart([label for label, _ in test], probabilities),
        words_chart(model, counts),
    ):
        print("saved", path.relative_to(ASSETS.parent))


if __name__ == "__main__":
    run()
