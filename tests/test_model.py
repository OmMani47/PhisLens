"""The saved pipeline separates obvious teaching examples and explains them."""

import json
import subprocess
import sys

from src.paths import METRICS_PATH, MODEL_PATH
from src.pipeline import load_model
from src.predict import classify_email, format_report, load_demos


def test_saved_model_and_metrics_exist():
    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["rows"] >= 400
    assert metrics["f1_phishing"] >= 0.9
    assert metrics["precision_phishing"] >= 0.9
    assert metrics["recall_phishing"] >= 0.9
    matrix = metrics["confusion_matrix"]["matrix"]
    assert len(matrix) == 2 and len(matrix[0]) == 2
    model = load_model()
    assert list(model.named_steps) == ["features", "clf"]
    assert list(model.named_steps["clf"].classes_) == [0, 1]


def test_demo_expectations_and_explanations():
    model = load_model()
    for demo in load_demos():
        result = classify_email(demo, pipeline=model)
        assert result["label"] == demo["model_expectation"], demo["id"]
        assert 0.0 <= result["phishing_probability"] <= 1.0
        assert abs(result["phishing_probability"] + result["legitimate_probability"] - 1) < 1e-6
        assert "summary" in result and result["summary"]
        assert isinstance(result["flags"], list)
        assert isinstance(result["top_phishing_cues"], list)
        report = format_report(result)
        assert result["label"].upper() in report
        if demo["group"] == "phish":
            assert result["phishing_probability"] >= 0.7
            assert result["flags"]
            assert result["top_phishing_cues"]
        if demo["group"] == "legit":
            assert result["legitimate_probability"] >= 0.7
        if demo["id"] == "quoted-homework":
            flag_ids = {flag["id"] for flag in result["flags"]}
            assert "credential_request" in flag_ids
            assert "urgency_hits" in flag_ids


def test_empty_email_is_rejected():
    try:
        classify_email({"sender": "Ada <a@northwind.edu>", "subject": "", "body": ""})
    except ValueError as exc:
        assert "subject" in str(exc).lower() or "body" in str(exc).lower()
    else:
        raise AssertionError("empty email should be rejected")


def test_cli_demo_json():
    completed = subprocess.run(
        [sys.executable, "-m", "src.predict", "--demo", "class-note", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["label"] == "legitimate"
