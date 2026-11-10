"""Streamlit script behavior for loading an example and rejecting an empty one."""

from streamlit.testing.v1 import AppTest


def _button(at: AppTest, label: str):
    matches = [button for button in at.button if button.label == label]
    assert matches, f"no button labeled {label!r}"
    return matches[0]


def test_demo_then_empty_message_warns_and_clears_verdict():
    app = AppTest.from_file("src/app.py", default_timeout=40)
    app.run()
    assert not app.exception

    _button(app, "Class announcement").click().run()
    assert not app.exception
    verdict = " ".join(block.value for block in list(app.success) + list(app.error))
    assert "legitimate" in verdict.lower()

    app.text_input[2].set_value("")
    app.text_area[0].set_value("   ")
    _button(app, "Classify email").click().run()
    assert not app.exception
    warnings = " ".join(block.value for block in app.warning).lower()
    assert "subject" in warnings or "body" in warnings
    assert not app.success
    assert not app.error

    _button(app, "Clear").click().run()
    assert not app.exception
    assert app.text_input[0].value == ""
    assert app.text_area[0].value == ""
    assert not app.success
    assert not app.error


def test_sidebar_disclaimer_and_quoted_homework_mismatch():
    app = AppTest.from_file("src/app.py", default_timeout=40)
    app.run()
    assert not app.exception
    captions = " ".join(block.value for block in app.caption).lower()
    assert "not an estimate of accuracy on real mail" in captions

    _button(app, "Quoted in class").click().run()
    assert not app.exception
    verdict = " ".join(block.value for block in app.error).lower()
    assert "phishing" in verdict
    warnings = " ".join(block.value for block in app.warning)
    assert "Classroom label: legitimate" in warnings
    assert "The model said phishing" in warnings
    assert not app.success
