from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def run_with(message: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    if message is not None:
        app.text_area[0].set_value(message).run()
    return app


def test_app_starts_without_errors():
    app = run_with()
    assert not app.exception


def test_app_flags_an_obvious_spam_message():
    app = run_with("WINNER! Claim your FREE prize now, call 09061701461")
    assert not app.exception
    assert len(app.error) == 1
    assert len(app.success) == 0


def test_app_passes_a_normal_message():
    app = run_with("Are we still meeting for lunch tomorrow?")
    assert not app.exception
    assert len(app.success) == 1
    assert len(app.error) == 0


def test_app_handles_an_empty_message():
    app = run_with("   ")
    assert not app.exception
    assert len(app.info) == 1
