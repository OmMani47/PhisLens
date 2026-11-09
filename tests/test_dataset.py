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
        assert saved["reply_to"] == rebuilt["reply_to"]
        assert saved["subject"] == rebuilt["subject"]
        assert saved["body"] == rebuilt["body"]
        assert int(saved["label"]) == rebuilt["label"]
        assert saved["category"] == rebuilt["category"]
    assert audit_rows(fresh) == []
    labels = [int(row["label"]) for row in disk]
    assert labels.count(0) == 280
    assert labels.count(1) == 280


def test_generator_is_repeatable_at_seed_42():
    first = generate_emails(seed=42)
    second = generate_emails(seed=42)
    assert [row["email_id"] for row in first] == [row["email_id"] for row in second]
    assert [row["body"] for row in first] == [row["body"] for row in second]
    assert [row["label"] for row in first] == [row["label"] for row in second]


def _note(**overrides) -> dict:
    row = {
        "email_id": "probe",
        "sender": "Prof. Ada Okonkwo <aokonkwo@northwind.edu>",
        "reply_to": "aokonkwo@northwind.edu",
        "subject": "Notes",
        "body": "See you Thursday in Room 210.",
        "label": 0,
        "category": "class_announcement",
    }
    row.update(overrides)
    return row


def test_audit_rejects_a_phishing_row_with_no_strong_signal():
    problems = audit_rows([_note(label=1, category="account_suspension")])
    assert any("phishing without a strong signal" in problem for problem in problems)


def test_audit_rejects_a_flag_on_ordinary_legitimate_mail():
    problems = audit_rows([
        _note(body="Enter your password and verify your account within 2 hours."),
    ])
    assert any("unexpected flag on legit class_announcement" in problem for problem in problems)


def test_audit_allows_facilities_urgency_and_rejects_anything_else():
    quiet = audit_rows([
        _note(category="facilities_note", body="The lab will reopen in the morning."),
    ])
    assert any("facilities flags" in problem for problem in quiet)

    urgent = audit_rows([
        _note(
            category="facilities_note",
            body="There is an urgent water shutoff tonight. No action is needed.",
        ),
    ])
    assert not any("facilities flags" in problem for problem in urgent)

    extra = audit_rows([
        _note(
            category="facilities_note",
            body="There is an urgent water shutoff tonight. Enter your password.",
        ),
    ])
    assert any("facilities flags" in problem for problem in extra)


def test_audit_rejects_a_repeated_phishing_body():
    lure = _note(
        label=1,
        category="password_reset",
        body="Enter your password and verify your account.",
    )
    problems = audit_rows([lure, dict(lure)])
    assert any("duplicate body is not a repeated legitimate note" in problem for problem in problems)


def test_demos_are_not_copies_of_training_rows():
    bodies = {row["body"] for row in _disk_rows()}
    demos = load_demos()
    assert len(demos) >= 8
    for demo in demos:
        assert demo["body"] not in bodies
        assert demo["classroom_label"] in {"phishing", "legitimate"}
