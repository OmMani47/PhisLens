"""The perfect holdout score stays an explained property of the teaching set.

A passing run records 1.000 next to the caveat. The number alone is not the test.
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import STRONG_FLAG_IDS, analyze
from src.paths import DATA_PATH, METRICS_PATH, ROOT
from src.predict import classify_email, load_demos


def _metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _evaluation_section() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pieces = readme.split("## Evaluation", 1)
    assert len(pieces) == 2, "README needs a dedicated Evaluation section"
    return pieces[1].split("\n## ", 1)[0]


def test_metrics_record_the_perfect_score_beside_the_caveat():
    metrics = _metrics()
    disclaimer = metrics["disclaimer"].lower()
    assert "synthetic teaching set" in disclaimer
    assert "not an estimate of accuracy on real mail" in disclaimer
    assert metrics["seed"] == 42
    assert metrics["test_size"] == 0.25
    assert metrics["rows"] == 560
    assert metrics["train_rows"] == 420
    assert metrics["test_rows"] == 140
    assert metrics["positive_class"] == "phishing"
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_phishing"] == 1.0
    assert metrics["recall_phishing"] == 1.0
    assert metrics["f1_phishing"] == 1.0
    assert metrics["precision_legitimate"] == 1.0
    assert metrics["recall_legitimate"] == 1.0
    assert metrics["f1_legitimate"] == 1.0
    assert metrics["confusion_matrix"]["matrix"] == [[70, 0], [0, 70]]
    assert metrics["misclassified_test_rows"] == []


def test_readme_evaluation_says_what_the_score_is_and_is_not():
    section = _evaluation_section()
    lowered = section.lower()
    assert "75/25" in section
    assert "seed 42" in lowered
    assert "stratified" in lowered
    assert "210" in section and "70" in section
    assert "separable" in lowered
    assert "what a perfect score means" in lowered
    assert "what a perfect score does not mean" in lowered
    assert "not an estimate of accuracy on real mail" in lowered
    assert "quoted in class" in lowered
    assert "classroom label" in lowered
    assert "legitimate" in lowered and "phishing" in lowered
    assert "credential_request" in section
    assert "urgency_hits" in section


def test_holdout_matches_the_stratified_seed_42_split():
    frame = pd.read_csv(DATA_PATH, keep_default_na=False)
    labels = frame["label"].astype(int)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    metrics = _metrics()
    assert len(train_frame) == metrics["train_rows"] == 420
    assert len(test_frame) == metrics["test_rows"] == 140
    assert int((train_frame["label"].astype(int) == 1).sum()) == 210
    assert int((train_frame["label"].astype(int) == 0).sum()) == 210
    assert int((test_frame["label"].astype(int) == 1).sum()) == 70
    assert int((test_frame["label"].astype(int) == 0).sum()) == 70

    # The empty error list is available because the fold was built that way.
    for row in test_frame.to_dict(orient="records"):
        flag_ids = {flag["id"] for flag in analyze(row).flags}
        if int(row["label"]) == 1:
            assert flag_ids & STRONG_FLAG_IDS, row["email_id"]
        elif row["category"] == "facilities_note":
            assert flag_ids == {"urgency_hits"}, row["email_id"]
        else:
            assert flag_ids == set(), row["email_id"]


def test_quoted_homework_disagrees_outside_the_training_file():
    demo = next(item for item in load_demos() if item["id"] == "quoted-homework")
    assert demo["title"] == "Quoted in class"
    assert demo["group"] == "edge"
    assert demo["classroom_label"] == "legitimate"
    assert demo["model_expectation"] == "phishing"
    assert demo["classroom_label"] != demo["model_expectation"]

    frame = pd.read_csv(DATA_PATH, keep_default_na=False)
    assert demo["body"] not in set(frame["body"])
    assert "northwind.edu" in demo["sender"]

    result = classify_email(demo)
    assert result["label"] == "phishing"
    assert result["label"] != demo["classroom_label"]
    flag_ids = {flag["id"] for flag in result["flags"]}
    assert {"credential_request", "urgency_hits"} <= flag_ids
    assert result["phishing_probability"] >= 0.7
