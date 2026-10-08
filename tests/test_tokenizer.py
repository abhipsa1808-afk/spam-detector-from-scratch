from spamdetector.tokenizer import tokenize


def test_lowercases_and_splits_words():
    assert tokenize("Hello THERE friend") == ["hello", "there", "friend"]


def test_keeps_apostrophes_inside_words():
    assert tokenize("Don't stop") == ["don't", "stop"]


def test_replaces_links():
    assert tokenize("visit http://win.example.com/now or www.prize.com") == [
        "visit",
        "<url>",
        "or",
        "<url>",
    ]


def test_replaces_numbers_by_length():
    assert tokenize("call 09061701461 in 2 days") == ["call", "<longnum>", "in", "<num>", "days"]


def test_replaces_currency_symbols():
    assert tokenize("win £500 or $9") == ["win", "<money>", "<num>", "or", "<money>", "<num>"]


def test_ignores_punctuation_and_empty_text():
    assert tokenize("!!! ... ???") == []
    assert tokenize("") == []
