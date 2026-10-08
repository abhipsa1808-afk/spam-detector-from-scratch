"""Turn a raw text message into a list of simple word tokens."""

from __future__ import annotations

import re

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_TOKEN = re.compile(r"<url>|[a-z]+(?:'[a-z]+)?|\d+|[£$€]")


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into tokens.

    Three kinds of things are replaced by a placeholder, because the exact
    value does not matter but the fact that it is there does:

    - web links become ``<url>``
    - currency symbols become ``<money>``
    - numbers become ``<num>`` (1-3 digits) or ``<longnum>`` (4+ digits,
      which is usually a phone number or a prize amount)
    """
    text = _URL.sub(" <url> ", text.lower())
    tokens: list[str] = []
    for token in _TOKEN.findall(text):
        if token.isdigit():
            tokens.append("<longnum>" if len(token) >= 4 else "<num>")
        elif token in "£$€":
            tokens.append("<money>")
        else:
            tokens.append(token)
    return tokens
