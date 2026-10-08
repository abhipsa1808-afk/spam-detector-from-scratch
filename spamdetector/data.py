"""Download and load the SMS Spam Collection dataset (UCI, CC BY 4.0)."""

from __future__ import annotations

import io
import random
import urllib.request
import zipfile
from pathlib import Path

DATA_URLS = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"

Row = tuple[str, str]  # (label, text) where label is "spam" or "ham"


def download(force: bool = False) -> Path:
    """Download the dataset once and return the path to the raw file."""
    if DATA_FILE.exists() and not force:
        return DATA_FILE
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in DATA_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            DATA_FILE.write_bytes(archive.read("SMSSpamCollection"))
            return DATA_FILE
        except Exception as error:  # try the next address
            last_error = error
    raise RuntimeError(f"Could not download the dataset: {last_error}")


def load() -> list[Row]:
    """Return every message as a (label, text) pair."""
    rows: list[Row] = []
    with download().open(encoding="utf-8", errors="replace") as f:
        for line in f:
            label, _, text = line.rstrip("\n").partition("\t")
            if label in ("spam", "ham") and text:
                rows.append((label, text))
    return rows


def train_test_split(
    rows: list[Row], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[Row], list[Row]]:
    """Shuffle and split the rows, keeping the spam/ham ratio equal in both parts."""
    rng = random.Random(seed)
    train: list[Row] = []
    test: list[Row] = []
    for label in ("ham", "spam"):
        group = [row for row in rows if row[0] == label]
        rng.shuffle(group)
        cut = int(len(group) * (1 - test_fraction))
        train += group[:cut]
        test += group[cut:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
