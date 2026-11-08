"""Unit tests for the handcrafted security checks."""

from src.features import FEATURE_NAMES, analyze
from src.predict import parse_message_text


def _flags(record: dict) -> set[str]:
    return {flag["id"] for flag in analyze(record).flags}


def test_feature_vector_matches_names():
    analysis = analyze(
        {"sender": "A <a@northwind.edu>", "reply_to": "", "subject": "Hi", "body": "Hello there."}
    )
    assert list(analysis.values) == FEATURE_NAMES
    assert len(analysis.vector()) == len(FEATURE_NAMES)


def test_ip_shortener_userinfo_and_suspicious_tld():
    flags = _flags(
        {
            "sender": "Notice <n@mail.example>",
            "subject": "Links",
            "body": " ".join(
                [
                    "http://203.0.113.10/training-demo",
                    "http://bit.ly/class-demo",
                    "https://bitly.example/demo-7",
                    "http://account-update.xyz/class-demo",
                    "http://accounts.northwind.edu@198.51.100.8/training-demo",
                ]
            ),
        }
    )
    assert "ip_url_count" in flags
    assert "shortener_count" in flags
    assert "suspicious_tld_count" in flags
    assert "url_userinfo" in flags


def test_lookalike_and_display_mismatch():
    flags = _flags(
        {
            "sender": "PayPal Security <alert@paypa1-secure.example>",
            "subject": "Hello",
            "body": "A short note.",
        }
    )
    assert "lookalike_domain" in flags
    assert "brand_impersonation" in flags


def test_official_harbor_domain_is_quiet():
    flags = _flags(
        {
            "sender": "Harbor Credit Union <statements@harborcu.example>",
            "reply_to": "statements@harborcu.example",
            "subject": "Statement ready",
            "body": "View it at https://online.harborcu.example/statements when you want.",
        }
    )
    assert flags == set()


def test_pineapple_is_not_an_apple_lookalike():
    flags = _flags(
        {
            "sender": "Pineapple Stand <hello@pineapple.example>",
            "subject": "Saturday fruit",
            "body": "The stand is open on Saturday morning.",
        }
    )
    assert "lookalike_domain" not in flags
    assert "brand_impersonation" not in flags


def test_same_school_reply_to_is_not_a_mismatch():
    flags = _flags(
        {
            "sender": "Prof. Ada Okonkwo <aokonkwo@cs.northwind.edu>",
            "reply_to": "aokonkwo@northwind.edu",
            "subject": "Notes",
            "body": "See you Thursday in Room 210.",
        }
    )
    assert "reply_to_mismatch" not in flags


def test_reply_to_mismatch_and_freemail_org_claim():
    flags = _flags(
        {
            "sender": "Northwind IT Helpdesk <helpdesk@gmail.com>",
            "reply_to": "tickets@helpdesk-queue.example",
            "subject": "Directory",
            "body": "Please read this note.",
        }
    )
    assert "freemail_org_claim" in flags
    assert "brand_impersonation" in flags
    assert "reply_to_mismatch" in flags


def test_credential_request_does_not_fire_on_a_denial():
    quiet = _flags(
        {
            "sender": "Northwind IT <it-news@northwind.edu>",
            "subject": "Reminder",
            "body": "Staff will not ask you to send a password by email. Review your account at https://accounts.northwind.edu yourself.",
        }
    )
    loud = _flags(
        {
            "sender": "Northwind IT <it-news@northwind.edu>",
            "subject": "Reminder",
            "body": "Enter your password and verify your account.",
        }
    )
    assert "credential_request" not in quiet
    assert "credential_request" in loud


def test_urgency_is_not_the_word_deadline():
    flags = _flags(
        {
            "sender": "Prof. Mei Chen <mchen@northwind.edu>",
            "subject": "Drafts",
            "body": "The assignment deadline is Friday. Bring the draft to class.",
        }
    )
    assert "urgency_hits" not in flags


def test_subject_caps_and_punctuation():
    flags = _flags(
        {
            "sender": "Rewards <r@claim-reward.xyz>",
            "subject": "CONGRATULATIONS!!! YOU WON A PRIZE TODAY!!!",
            "body": "Hello.",
        }
    )
    assert "subject_excess_caps" in flags
    assert "subject_excess_punct" in flags
    assert "suspicious_tld_count" in flags


def test_html_form_script_and_mismatched_anchor():
    flags = _flags(
        {
            "sender": "Notice <n@mail.example>",
            "subject": "Form",
            "body": (
                '<form action="http://203.0.113.50/training-demo"></form> '
                "<script></script> <input type=\"password\"> "
                '<a href="http://203.0.113.50/training-demo">https://accounts.northwind.edu/login</a>'
            ),
        }
    )
    assert "html_form" in flags
    assert "html_script" in flags
    assert "html_password_input" in flags
    assert "link_text_mismatch" in flags
    assert "ip_url_count" in flags


def test_parse_message_with_headers_and_bare_body():
    parsed = parse_message_text(
        "From: Ada <a@northwind.edu>\nReply-To: a@cs.northwind.edu\nSubject: Hello\n\nSee you Thursday.\n"
    )
    assert parsed["sender"] == "Ada <a@northwind.edu>"
    assert parsed["reply_to"] == "a@cs.northwind.edu"
    assert parsed["subject"] == "Hello"
    assert parsed["body"] == "See you Thursday."

    bare = parse_message_text("Just a note for the group.\n")
    assert bare["body"] == "Just a note for the group."
    assert bare["subject"] == ""
