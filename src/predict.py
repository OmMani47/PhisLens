"""Classify one email from the command line.

Examples, from the repository root:

    python -m src.predict --demo suspension
    python -m src.predict --file message.txt
    python -m src.predict --sender "Prof. Ada Okonkwo <aokonkwo@northwind.edu>" \\
        --subject "Office hours" --body "Thursday, room 210."

A file may be a bare body, or headers followed by a blank line:

    From: Name <user@host>
    Reply-To: other@host
    Subject: Hello

    Body text.

Nothing is uploaded. The saved pipeline runs on this machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import normalize_record
from src.paths import DEMOS_PATH
from src.pipeline import explain, load_model

_HEADER_NAMES = {"from", "sender", "reply-to", "reply_to", "subject", "to", "date"}


def load_demos(path: Path | None = None) -> list[dict]:
    demo_path = path or DEMOS_PATH
    if not demo_path.exists():
        raise FileNotFoundError(f"No demo file at {demo_path}")
    payload = json.loads(demo_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Demo file must be a list of emails.")
    return payload


def parse_message_text(text: str) -> dict[str, str]:
    """Parse a pasted message. Unknown first lines are treated as the body."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")
    headers: dict[str, str] = {}
    index = 0
    saw_header = False
    while index < len(lines):
        line = lines[index]
        if line.strip() == "":
            if saw_header:
                index += 1
                break
            index += 1
            continue
        if ":" not in line:
            break
        key, value = line.split(":", 1)
        if key.strip().lower() not in _HEADER_NAMES:
            break
        headers[key.strip().lower()] = value.strip()
        saw_header = True
        index += 1
    if not saw_header:
        return {"sender": "", "reply_to": "", "subject": "", "body": text.strip()}
    return {
        "sender": headers.get("from", headers.get("sender", "")),
        "reply_to": headers.get("reply-to", headers.get("reply_to", "")),
        "subject": headers.get("subject", ""),
        "body": "\n".join(lines[index:]).strip(),
    }


def classify_email(record: dict, pipeline=None) -> dict:
    """Return the verdict, probabilities, fired checks, and n-gram contributions."""
    fields = normalize_record(record)
    if not fields["subject"] and not fields["body"]:
        raise ValueError("Add a subject or a body. A sender by itself is not an email to score.")
    model = pipeline or load_model()
    return explain(model, fields)


def format_report(result: dict) -> str:
    lines = [
        "PhishLens",
        f"Verdict: {result['label'].upper()}",
        f"Phishing probability:   {result['phishing_probability']:.1%}",
        f"Legitimate probability: {result['legitimate_probability']:.1%}",
        "",
        result["summary"],
        "",
        "Security checks that fired:",
    ]
    if result["flags"]:
        for flag in result["flags"]:
            lines.append(f"  - {flag['title']}: {flag['detail']}")
    else:
        lines.append("  (none)")

    def extend(title: str, cues: list[dict], toward: str) -> None:
        lines.extend(["", title])
        if not cues:
            lines.append("  (none)")
            return
        for cue in cues:
            lines.append(f"  - {cue['feature']}: {cue['contribution']:+.3f} toward {toward}")

    extend("Wording that pushed toward phishing:", result["top_phishing_cues"], "phishing")
    extend("Wording that pushed toward legitimate:", result["top_legitimate_cues"], "legitimate")
    if result["low_margin"]:
        lines.extend(["", "Low margin: read the message before acting on this verdict."])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one email with the saved PhishLens model.")
    parser.add_argument("--sender", default="", help="From header, for example Name <user@host>.")
    parser.add_argument("--reply-to", default="", help="Optional Reply-To address.")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", default="", help="Message body. If omitted, stdin is read when it is piped.")
    parser.add_argument("--file", type=Path, help="Text file with optional From/Reply-To/Subject headers.")
    parser.add_argument("--demo", help="Id of a preset in data/demo_emails.json.")
    parser.add_argument("--json", action="store_true", help="Print the raw explanation as JSON.")
    parser.add_argument("--model", default=None, help="Path to a pipeline.joblib file.")
    return parser.parse_args()


def _record_from_args(args: argparse.Namespace) -> dict:
    record = {"sender": "", "reply_to": "", "subject": "", "body": ""}
    if args.demo:
        matches = [demo for demo in load_demos() if demo.get("id") == args.demo]
        if not matches:
            known = ", ".join(demo.get("id", "?") for demo in load_demos())
            raise SystemExit(f"Unknown demo {args.demo!r}. Known ids: {known}")
        demo = matches[0]
        record = {
            "sender": demo.get("sender", ""),
            "reply_to": demo.get("reply_to", ""),
            "subject": demo.get("subject", ""),
            "body": demo.get("body", ""),
        }
    if args.file:
        parsed = parse_message_text(args.file.read_text(encoding="utf-8"))
        for key, value in parsed.items():
            if value:
                record[key] = value
    if args.sender:
        record["sender"] = args.sender
    if args.reply_to:
        record["reply_to"] = args.reply_to
    if args.subject:
        record["subject"] = args.subject
    if args.body:
        record["body"] = args.body
    elif not record["body"] and not sys.stdin.isatty():
        piped = sys.stdin.read()
        parsed = parse_message_text(piped)
        # A bare pipe fills the body. A piped message with headers fills all fields
        # that the command line left empty.
        if parsed["sender"] or parsed["subject"] or parsed["reply_to"]:
            for key, value in parsed.items():
                if value and not record[key]:
                    record[key] = value
        elif parsed["body"]:
            record["body"] = parsed["body"]
    return record


def main() -> None:
    args = _parse_args()
    record = _record_from_args(args)
    try:
        model = load_model(args.model) if args.model else load_model()
        result = classify_email(record, pipeline=model)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
