"""Interpretable email-security checks.

Each check is a heuristic a person can audit: who sent the message, where
links point, and whether the wording asks for money or a password. The
numbers land in a logistic regression next to TF-IDF text features. The
same function both builds those numbers and returns the plain-language
flags shown in the demo.

These rules are deliberately simple. They are a teaching model of an
inbox, not a mail gateway.
"""

from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

# Brand names a classroom detector should notice, plus the domains we treat
# as real for that brand. Subdomains of a listed domain count as official.
# "northwind.edu" is the fictional university in the teaching set.
BRANDS: dict[str, dict[str, set[str] | list[str]]] = {
    "paypal": {"names": ["paypal"], "domains": {"paypal.com", "paypal.example"}},
    "microsoft": {
        "names": ["microsoft", "office365", "outlook"],
        "domains": {"microsoft.com", "office.com", "live.com", "microsoft.example", "office365.example"},
    },
    "apple": {"names": ["apple", "icloud"], "domains": {"apple.com", "icloud.com", "apple.example"}},
    "google": {"names": ["google"], "domains": {"google.com", "gmail.com", "google.example"}},
    "amazon": {"names": ["amazon"], "domains": {"amazon.com", "amazon.example"}},
    "netflix": {"names": ["netflix"], "domains": {"netflix.com", "netflix.example"}},
    "northwind": {"names": ["northwind"], "domains": {"northwind.edu"}},
    "dhl": {"names": ["dhl"], "domains": {"dhl.com", "dhl.example"}},
    "fedex": {"names": ["fedex"], "domains": {"fedex.com", "fedex.example"}},
    "irs": {"names": ["irs"], "domains": {"irs.gov"}},
    "adobe": {"names": ["adobe"], "domains": {"adobe.com", "adobe.example"}},
    "dropbox": {"names": ["dropbox"], "domains": {"dropbox.com", "dropbox.example"}},
    "linkedin": {"names": ["linkedin"], "domains": {"linkedin.com", "linkedin.example"}},
    "harbor": {"names": ["harbor", "harborcu"], "domains": {"harborcu.example"}},
}

# Cheap or frequently abused TLDs. A teaching list, not a production blocklist:
# plenty of real organizations use some of these suffixes.
SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "click", "country", "gq", "ml", "cf", "ga", "tk",
    "rest", "loan", "cam", "bid", "work", "link", "buzz", "monster", "quest", "support",
}

# Live shortener hostnames. The dataset itself uses reserved stand-ins
# (bitly.example and similar) so a curious click does not hit a real redirect.
SHORTENER_HOSTS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "tiny.cc", "rb.gy", "shorturl.at", "rebrand.ly",
}
SHORTENER_STANDIN_LABELS = {
    "bitly", "tinyurl", "cuttly", "shorturl", "owly", "googl", "buffly", "rebrandly", "tinycc",
}
RESERVED_SUFFIXES = {"example", "invalid", "test"}

FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "proton.me", "protonmail.com", "gmx.com", "live.com", "msn.com",
}

# Extra characters allowed after a brand only when they form an obvious
# compound such as "paypal-secure" or "northwindsupport". This avoids
# flagging unrelated words that merely contain a brand substring.
COMPOUND_TAILS = {
    "support", "secure", "login", "account", "verify", "update", "alert", "team",
    "help", "desk", "online", "service", "billing", "refund", "prize", "mail",
    "notice", "center", "portal", "reset", "check", "app", "security", "services",
    "signin", "auth", "helpdesk",
}

FEATURE_NAMES = [
    "url_count",
    "ip_url_count",
    "shortener_count",
    "suspicious_tld_count",
    "url_userinfo",
    "lookalike_domain",
    "brand_impersonation",
    "urgency_hits",
    "credential_request",
    "payment_request",
    "reply_to_mismatch",
    "freemail_org_claim",
    "html_script",
    "html_form",
    "html_password_input",
    "link_text_mismatch",
    "subject_excess_caps",
    "subject_excess_punct",
]

# Flags that should not, by themselves, be the only reason a teaching-set
# phishing row exists. Urgency and a shouty subject also show up in real
# mail that is harmless.
STRONG_FLAG_IDS = {
    "ip_url_count",
    "shortener_count",
    "suspicious_tld_count",
    "url_userinfo",
    "lookalike_domain",
    "brand_impersonation",
    "credential_request",
    "payment_request",
    "reply_to_mismatch",
    "freemail_org_claim",
    "html_script",
    "html_form",
    "html_password_input",
    "link_text_mismatch",
}

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "@": "a", "$": "s"})

_EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
_URL_RE = re.compile(r"""https?://[^\s<>"')\]]+""", re.IGNORECASE)
_WWW_RE = re.compile(r"""(?<![@/])\bwww\.[^\s<>"')\]]+""", re.IGNORECASE)
_ANCHOR_RE = re.compile(
    r"""<a\b[^>]*?\bhref\s*=\s*(['"])(.*?)\1[^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)
_HOST_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)

# Requests, not mere mentions. A campus reminder that says "we will not ask
# for your password" should not trip the credential check.
_CREDENTIAL_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\benter (?:your |the )?(?:password|passcode|credentials|pin)\b",
        r"\b(?:confirm|verify|provide|send|type|share|submit|email) (?:your |the )?(?:password|passcode|credentials|login|pin)\b",
        r"\b(?:username|user name) and password\b",
        r"\bpassword and (?:username|user name)\b",
        r"\bverify your account\b",
        r"\bconfirm your identity\b",
        r"\bone-time (?:password|code|pin)\b",
        r"\b(?:ssn|social security number)\b",
        r"\bseed phrase\b",
        r"\blog in and (?:confirm|verify|enter)\b",
        r"\breset (?:your |the )?password (?:here|at|using)\b",
    )
]
_PAYMENT_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bgift cards?\b",
        r"\bwire transfer\b",
        r"\brouting number\b",
        r"\bbank account number\b",
        r"\bbitcoin\b",
        r"\bcrypto wallet\b",
        r"\bpay (?:this |the fee )?immediately\b",
        r"\boverdue invoice\b",
        r"\bpayment (?:is )?overdue\b",
        r"\bsend (?:the |your )?codes\b",
        r"\bprepaid cards?\b",
        r"\bconfirm payment details\b",
        r"\bupdate (?:your )?billing\b",
        r"\brelease the payment\b",
        r"\bholding fee\b",
    )
]
_URGENCY_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\burgent(?:ly)?\b",
        r"\bact now\b",
        r"\bimmediately\b",
        r"\bwithin \d+ hours\b",
        r"\bwithin 24 hours\b",
        r"\baccount (?:is |was |has been |will be )?suspended\b",
        r"\bsuspended\b",
        r"\baccount (?:is |was |has been |will be )?locked\b",
        r"\bfinal notice\b",
        r"\blast warning\b",
        r"\bexpires today\b",
        r"\basap\b",
        r"\bright away\b",
        r"\blimited time\b",
        r"\baction required\b",
        r"\baccount will be closed\b",
        r"\bfailure to (?:comply|respond)\b",
        r"\bunusual activity\b",
        r"\bdeactivated\b",
        r"\bforfeit",
    )
]
_ORG_CLAIM_RE = re.compile(
    r"\b(it support|it help\s?desk|help\s?desk|security team|security|support|account services?|"
    r"university|registrar|professor|admin(?:istrator)?|billing|customer service|"
    r"service desk|bank|customer support)\b",
    re.IGNORECASE,
)
_SCRIPT_RE = re.compile(r"<\s*script\b|javascript\s*:", re.IGNORECASE)
_FORM_RE = re.compile(r"<\s*form\b", re.IGNORECASE)
_PASSWORD_INPUT_RE = re.compile(
    r"<\s*input\b[^>]*\btype\s*=\s*['\"]?(?:password|hidden)\b",
    re.IGNORECASE,
)

_GENERIC_LABELS = {
    "www", "com", "edu", "org", "net", "gov", "example", "invalid", "test",
    "co", "ac", "mail", "email", "smtp", "mx",
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def normalize_record(record: dict) -> dict[str, str]:
    """Accept the CSV columns or the slightly different names the CLI allows."""
    sender = _clean(record.get("sender") or record.get("from") or record.get("From"))
    reply_to = _clean(
        record.get("reply_to") or record.get("reply-to") or record.get("Reply-To") or ""
    )
    subject = _clean(record.get("subject") or record.get("Subject"))
    body = _clean(record.get("body") or record.get("Body"))
    return {"sender": sender, "reply_to": reply_to, "subject": subject, "body": body}


def record_to_text(record: dict) -> str:
    """Text the TF-IDF branch sees. The label and category never enter this string."""
    fields = normalize_record(record)
    return "\n".join([fields["sender"], fields["reply_to"], fields["subject"], fields["body"]])


def parse_address(value: str) -> tuple[str, str]:
    """Split ``Display Name <user@host>`` into a display name and an email."""
    raw = _clean(value)
    if not raw:
        return "", ""
    bracket = re.search(r"<\s*([^>]+)\s*>", raw)
    if bracket:
        email = bracket.group(1).strip()
        display = raw[: bracket.start()].strip().strip('"').strip()
        return display, email
    found = _EMAIL_RE.search(raw)
    if not found:
        return raw, ""
    email = found.group(0).strip()
    display = f"{raw[: found.start()]} {raw[found.end():]}".strip().strip('"').strip("<>")
    return display, email


def domain_of(email: str) -> str:
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[1].strip().lower().rstrip(".")


def registered_domain(host: str) -> str:
    """Last two labels, which is enough for the hosts in this project.

    ``mail.northwind.edu`` and ``dept.cs.northwind.edu`` both reduce to
    ``northwind.edu``, so a reply-to inside the same school is not a mismatch.
    """
    host = (host or "").lower().strip().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host or _is_ip(host):
        return host
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return ".".join(parts)
    return ".".join(parts[-2:])


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _fold(token: str) -> str:
    return token.lower().translate(_LEET).replace("-", "").replace("_", "")


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, ca in enumerate(left, start=1):
        current = [i]
        for j, cb in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _name_match(token: str, name: str) -> str | None:
    """How ``token`` relates to a brand name.

    ``exact`` and ``leet`` are strict. ``edit`` and ``embedded`` only run on
    longer names so short tokens such as ``irs`` do not collide with
    ordinary words. Embedded matches must be a prefix or suffix plus a
    compound tail (``paypal`` + ``secure``), which keeps ``pineapple`` from
    looking like Apple.
    """
    raw = token.lower().strip(".-_")
    if not raw or raw in _GENERIC_LABELS:
        return None
    target = _fold(name)
    folded = _fold(raw)
    if raw == name or folded == target:
        return "exact" if raw == name else "leet"
    if len(target) < 5 or len(folded) < 5:
        return None
    if abs(len(folded) - len(target)) <= 1 and _edit_distance(folded, target) == 1:
        return "edit"
    rest = ""
    if folded.startswith(target) and folded != target:
        rest = folded[len(target) :]
    elif folded.endswith(target) and folded != target:
        rest = folded[: -len(target)]
    if rest in COMPOUND_TAILS:
        return "embedded"
    return None


def _official_host(host: str, brand_key: str) -> bool:
    if not host:
        return False
    host = host.lower().strip().rstrip(".")
    domains = BRANDS[brand_key]["domains"]
    assert isinstance(domains, set)
    for official in domains:
        if host == official or host.endswith("." + official):
            return True
    return False


def _host_labels(host: str) -> list[str]:
    labels: list[str] = []
    for label in host.lower().split("."):
        if not label or label in _GENERIC_LABELS:
            continue
        labels.append(label)
        labels.extend(part for part in re.split(r"[-_]", label) if part and part not in _GENERIC_LABELS)
    return labels


def _clean_url(raw: str) -> str:
    return raw.rstrip(".,);:!?\"'`>")


def extract_urls(text: str) -> list[str]:
    found = [_clean_url(match) for match in _URL_RE.findall(text or "")]
    found.extend(_clean_url("http://" + match) for match in _WWW_RE.findall(text or ""))
    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(url)
    return unique


def _hostname(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().strip().rstrip(".")


def _is_shortener(host: str) -> bool:
    if not host:
        return False
    if host.startswith("www."):
        host = host[4:]
    if host in SHORTENER_HOSTS:
        return True
    labels = host.split(".")
    if len(labels) < 2:
        return False
    suffix = ".".join(labels[1:])
    return labels[0] in SHORTENER_STANDIN_LABELS and suffix in RESERVED_SUFFIXES


def _pattern_hits(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def _subject_caps(subject: str) -> bool:
    letters = [char for char in subject if char.isalpha()]
    if len(letters) < 10:
        return False
    ratio = sum(char.isupper() for char in letters) / len(letters)
    return ratio >= 0.6


def _subject_punct(subject: str) -> bool:
    if subject.count("!") >= 3 or "$$$" in subject:
        return True
    return re.search(r"[!?]{2,}", subject) is not None


def _link_mismatches(text: str) -> list[str]:
    """Anchors whose visible text names a different host than ``href``."""
    mismatches: list[str] = []
    for match in _ANCHOR_RE.finditer(text or ""):
        href = match.group(2).strip()
        visible = re.sub(r"<[^>]+>", " ", match.group(3))
        href_host = _hostname(href)
        if not href_host:
            continue
        visible_hosts = [host.lower() for host in _HOST_RE.findall(visible)]
        for shown in visible_hosts:
            if registered_domain(shown) != registered_domain(href_host):
                mismatches.append(f"visible host {shown} points at {href_host}")
                break
    return mismatches


@dataclass
class Analysis:
    """Numeric features in ``FEATURE_NAMES`` order, plus the flags that fired."""

    values: dict[str, float]
    flags: list[dict[str, str]] = field(default_factory=list)

    def vector(self) -> list[float]:
        return [float(self.values[name]) for name in FEATURE_NAMES]


def _flag(flags: list[dict[str, str]], flag_id: str, title: str, detail: str) -> None:
    flags.append({"id": flag_id, "title": title, "detail": detail})


def analyze(record: dict) -> Analysis:
    """Score one email and list the security checks that fired.

    Counts are capped so one noisy field cannot swamp the linear model.
    Binary checks stay 0 or 1, which keeps a coefficient readable: if the
    check fired, its contribution is about the size of its weight.
    """
    fields = normalize_record(record)
    sender_display, sender_email = parse_address(fields["sender"])
    _, reply_email = parse_address(fields["reply_to"])
    sender_host = domain_of(sender_email)
    reply_host = domain_of(reply_email)
    combined = f"{fields['subject']}\n{fields['body']}"
    urls = extract_urls(combined)

    values = {name: 0.0 for name in FEATURE_NAMES}
    flags: list[dict[str, str]] = []

    ip_hosts: list[str] = []
    short_hosts: list[str] = []
    suspicious_hosts: list[str] = []
    userinfo_urls: list[str] = []
    for url in urls:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.username:
            userinfo_urls.append(url)
        if not host:
            continue
        if _is_ip(host):
            ip_hosts.append(host)
            continue
        if _is_shortener(host):
            short_hosts.append(host)
        tld = host.rsplit(".", 1)[-1]
        if tld in SUSPICIOUS_TLDS:
            suspicious_hosts.append(host)

    # The sender host is checked too. A display name can look fine while the
    # mailbox itself sits on a frequently abused suffix.
    if sender_host and not _is_ip(sender_host):
        sender_tld = sender_host.rsplit(".", 1)[-1]
        if sender_tld in SUSPICIOUS_TLDS and sender_host not in suspicious_hosts:
            suspicious_hosts.append(sender_host)

    values["url_count"] = float(min(len(urls), 5))
    values["ip_url_count"] = float(min(len(ip_hosts), 3))
    values["shortener_count"] = float(min(len(short_hosts), 3))
    values["suspicious_tld_count"] = float(min(len(suspicious_hosts), 3))
    values["url_userinfo"] = 1.0 if userinfo_urls else 0.0

    if ip_hosts:
        shown = ", ".join(dict.fromkeys(ip_hosts))
        _flag(
            flags,
            "ip_url_count",
            "Link to a raw IP address",
            f"A link uses the numeric host {shown}. Legitimate mail in this mailbox uses a name you can read.",
        )
    if short_hosts:
        shown = ", ".join(dict.fromkeys(short_hosts))
        _flag(
            flags,
            "shortener_count",
            "Shortened link",
            f"A link uses the shortener-style host {shown}, which hides the destination.",
        )
    if suspicious_hosts:
        shown = ", ".join(dict.fromkeys(suspicious_hosts))
        _flag(
            flags,
            "suspicious_tld_count",
            "Uncommon domain suffix",
            f"These hosts use a suffix from the teaching list of frequently abused TLDs: {shown}.",
        )
    if userinfo_urls:
        _flag(
            flags,
            "url_userinfo",
            "Link hides the real host",
            "A link contains an @ sign. Browsers go to the host after the @, and the text before it can look like a trusted site.",
        )

    lookalike_notes: list[str] = []
    impersonation_notes: list[str] = []
    local_part = sender_email.split("@", 1)[0] if "@" in sender_email else ""
    local_tokens = [tok for tok in re.split(r"[^a-z0-9]+", local_part.lower()) if tok]
    display_tokens = re.findall(r"[a-z0-9]+", sender_display.lower())

    for brand_key, spec in BRANDS.items():
        names = spec["names"]
        assert isinstance(names, list)
        official = _official_host(sender_host, brand_key) if sender_host else False
        if sender_host and not official:
            for label in _host_labels(sender_host):
                for name in names:
                    kind = _name_match(label, name)
                    if kind in {"leet", "edit", "embedded"}:
                        lookalike_notes.append(
                            f"“{label}” in {sender_host} imitates {name}"
                        )
                    elif kind == "exact":
                        impersonation_notes.append(
                            f"the sender domain {sender_host} uses the {name} name but is not a {name} domain"
                        )
        if official:
            continue
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", sender_display, re.IGNORECASE):
                impersonation_notes.append(
                    f"the display name “{sender_display}” claims {name}, and {sender_host or 'the sender'} is not a {name} domain"
                )
                break
            if any(_name_match(token, name) == "leet" for token in display_tokens):
                impersonation_notes.append(
                    f"the display name “{sender_display}” disguises {name} with lookalike characters"
                )
                break
        if any(_name_match(token, name) in {"exact", "leet"} for token in local_tokens for name in names):
            impersonation_notes.append(
                f"the mailbox name “{local_part}” suggests a protected brand, and {sender_host or 'the domain'} does not match it"
            )

    # De-duplicate while keeping the first, most specific notes.
    lookalike_notes = list(dict.fromkeys(lookalike_notes))
    impersonation_notes = list(dict.fromkeys(impersonation_notes))
    if lookalike_notes:
        values["lookalike_domain"] = 1.0
        _flag(
            flags,
            "lookalike_domain",
            "Lookalike sender domain",
            "The sender domain imitates a known name: " + "; ".join(lookalike_notes[:2]) + ".",
        )
    if impersonation_notes:
        values["brand_impersonation"] = 1.0
        _flag(
            flags,
            "brand_impersonation",
            "Brand or organization mismatch",
            "The visible identity does not match the sending domain: "
            + "; ".join(impersonation_notes[:2])
            + ".",
        )

    urgency = min(_pattern_hits(_URGENCY_RES, combined), 4)
    values["urgency_hits"] = float(urgency)
    if urgency:
        _flag(
            flags,
            "urgency_hits",
            "Urgency language",
            "The subject or body pressures the reader to act quickly or says the account is at risk. Urgency alone is not proof.",
        )

    if any(pattern.search(combined) for pattern in _CREDENTIAL_RES):
        values["credential_request"] = 1.0
        _flag(
            flags,
            "credential_request",
            "Asks for a secret or identity check",
            "The message asks the reader to enter, confirm, or send a password, a one-time code, or an identity number.",
        )
    if any(pattern.search(combined) for pattern in _PAYMENT_RES):
        values["payment_request"] = 1.0
        _flag(
            flags,
            "payment_request",
            "Payment or code request",
            "The message asks for a payment, a gift card, a wire, or codes from a prepaid card.",
        )

    if sender_host and reply_host and registered_domain(sender_host) != registered_domain(reply_host):
        values["reply_to_mismatch"] = 1.0
        _flag(
            flags,
            "reply_to_mismatch",
            "Reply-To is a different domain",
            f"From is {sender_host}, but Reply-To is {reply_host}. A reply would go somewhere else.",
        )

    if sender_host in FREEMAIL_DOMAINS and _ORG_CLAIM_RE.search(sender_display):
        values["freemail_org_claim"] = 1.0
        _flag(
            flags,
            "freemail_org_claim",
            "Organization name on a free mailbox",
            f"“{sender_display}” reads like an institution, but the mailbox is {sender_host}.",
        )

    if _SCRIPT_RE.search(fields["body"]):
        values["html_script"] = 1.0
        _flag(
            flags,
            "html_script",
            "Script hint in the body",
            "The body contains a script tag or a javascript: link. Ordinary campus mail in this set is plain text.",
        )
    if _FORM_RE.search(fields["body"]):
        values["html_form"] = 1.0
        _flag(
            flags,
            "html_form",
            "HTML form in the body",
            "The body contains a form tag. A form inside email is a common way to collect input.",
        )
    if _PASSWORD_INPUT_RE.search(fields["body"]):
        values["html_password_input"] = 1.0
        _flag(
            flags,
            "html_password_input",
            "Password or hidden field",
            "The body contains a password or hidden input. Email should not collect a password.",
        )

    mismatches = _link_mismatches(fields["body"])
    if mismatches:
        values["link_text_mismatch"] = 1.0
        _flag(
            flags,
            "link_text_mismatch",
            "Link text disagrees with the target",
            "An anchor shows one host and points at another: " + mismatches[0] + ".",
        )

    if _subject_caps(fields["subject"]):
        values["subject_excess_caps"] = 1.0
        _flag(
            flags,
            "subject_excess_caps",
            "Subject is mostly capitals",
            "Most letters in the subject are uppercase, which is a common attention grab.",
        )
    if _subject_punct(fields["subject"]):
        values["subject_excess_punct"] = 1.0
        _flag(
            flags,
            "subject_excess_punct",
            "Subject punctuation",
            "The subject stacks exclamation marks, question marks, or dollar signs.",
        )

    return Analysis(values=values, flags=flags)
