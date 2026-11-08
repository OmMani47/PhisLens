"""The committed mailbox matches the generator and stays a teaching set."""

import csv

from src.dataset import PLAN, audit_rows, generate_emails
from src.paths import DATA_PATH
from src.predict import load_demos


def _disk_rows() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_plan_is_balanced_and_large_enough():
    legit = sum(count for _name, _fn, count in PLAN if _name in {
        "class_announcement", "coworker_note", "receipt", "security_notice",
        "newsletter", "meeting", "facilities_note", "tuition_notice",
    })
    phishing = sum(count for _name, _fn, count in PLAN) - legit
    assert legit == 280
    assert phishing == 280
    assert legit + phishing >= 400


def test_csv_matches_generator_and_passes_audit():
    fresh = generate_emails(seed=42)
    disk = _disk_rows()
    assert len(disk) == len(fresh) == 560
    for saved, rebuilt in zip(disk, fresh):
        assert saved["email_id"] == rebuilt["email_id"]
        assert saved["sender"] == rebuilt["sender"]
        assert saved["subject"] == rebuilt["subject"]
        assert saved["body"] == rebuilt["body"]
        assert int(saved["label"]) == rebuilt["label"]
        assert saved["category"] == rebuilt["category"]
    assert audit_rows(fresh) == []
    labels = [int(row["label"]) for row in disk]
    assert labels.count(0) == 280
    assert labels.count(1) == 280


def test_demos_are_not_copies_of_training_rows():
    bodies = {row["body"] for row in _disk_rows()}
    demos = load_demos()
    assert len(demos) >= 8
    for demo in demos:
        assert demo["body"] not in bodies
        assert demo["classroom_label"] in {"phishing", "legitimate"}
