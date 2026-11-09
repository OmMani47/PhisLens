"""Build the synthetic teaching mailbox.

Every row is fiction for a defensive classifier. Phishing rows are obvious
classroom caricatures: the reserved ``.example`` suffix, documentation IP
addresses from RFC 5737, and blunt requests for a password or a payment.
They give the detector positive examples of known signals.
They are not templates for real messages.

Legitimate rows are ordinary mail at Northwind University, a fictional
school, plus a few fictional vendors. Some of them use words like "urgent"
or "password" in a harmless way so those words are not a shortcut by
themselves.

The generator is deterministic. ``python -m src.dataset`` rewrites
``data/sample_emails.csv``.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from src.features import STRONG_FLAG_IDS, analyze
from src.paths import DATA_PATH

FIELDS = ["email_id", "sender", "reply_to", "subject", "body", "label", "category"]

FIRST = [
    "Ada", "Luis", "Priya", "Jonah", "Mei", "Samir", "Elena", "Chris", "Noah",
    "Hana", "Owen", "Rosa", "Ibrahim", "Greta", "Theo", "Leila", "Marcus", "Yuki",
]
LAST = [
    "Okonkwo", "Martinez", "Shah", "Berg", "Chen", "Rahman", "Novak", "Ibarra",
    "Kim", "Almeida", "Dubois", "Nguyen", "Cohen", "Silva", "Andersen", "Rossi",
    "Patel", "Wright",
]
COURSES = [
    ("CS 484", "decision trees"),
    ("CS 331", "packet captures"),
    ("CYEN 301", "risk models"),
    ("AI 520", "evaluation splits"),
    ("STAT 401", "confidence intervals"),
    ("MATH 240", "proof drafts"),
    ("ENGL 210", "source notes"),
    ("BIO 150", "lab notebooks"),
]
PROFESSORS = [
    ("Ada Okonkwo", "aokonkwo"),
    ("Luis Martinez", "lmartinez"),
    ("Priya Shah", "pshah"),
    ("Jonah Berg", "jberg"),
    ("Mei Chen", "mchen"),
    ("Elena Novak", "enovak"),
    ("Samir Rahman", "srahman"),
    ("Hana Ibarra", "hibarra"),
]
PHISH_BRANDS = [
    "paypal", "microsoft", "apple", "google", "amazon", "netflix",
    "northwind", "adobe", "dropbox", "linkedin", "harbor",
]
BRAND_TITLE = {
    "paypal": "PayPal",
    "microsoft": "Microsoft",
    "apple": "Apple",
    "google": "Google",
    "amazon": "Amazon",
    "netflix": "Netflix",
    "northwind": "Northwind",
    "adobe": "Adobe",
    "dropbox": "Dropbox",
    "linkedin": "LinkedIn",
    "harbor": "Harbor Credit Union",
    "dhl": "DHL",
    "fedex": "FedEx",
    "irs": "IRS",
}
# Shared across both classes so reference words are not a label leak.
REF_TOKENS = [
    "amber", "birch", "cinder", "delta", "ember", "flint", "grove", "hazel",
    "ivory", "juniper", "kelp", "lark", "moss", "onyx", "pebble", "quartz",
    "raven", "spruce", "thistle", "willow",
]
DOC_NETS = ["203.0.113", "198.51.100", "192.0.2"]
REPLY_HOSTS = [
    "mail-collector.example",
    "reply-desk.example",
    "inbound-queue.example",
    "ticket-dump.example",
    "notice-box.example",
]
FREEMAIL = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]


def _pick(rng: random.Random, options: list):
    return options[rng.randrange(len(options))]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _title(brand: str) -> str:
    return BRAND_TITLE[brand]


def _person(rng: random.Random) -> tuple[str, str]:
    first = _pick(rng, FIRST)
    last = _pick(rng, LAST)
    local = (first[0] + last).lower()
    return f"{first} {last}", local


def _reference(rng: random.Random) -> str:
    """Keep the random stream stable without adding a trailer to the text.

    An earlier version ended every message with ``Reference amber-birch-cedar``.
    The shared word lined up with whatever token came before it and invented
    bigrams such as ``it reference``. The draw stays so the rest of each
    message does not reshuffle.
    """
    for _ in range(3):
        _pick(rng, REF_TOKENS)
    return ""


def _leet(brand: str) -> str:
    if "l" in brand:
        return brand.replace("l", "1", 1)
    if "o" in brand:
        return brand.replace("o", "0", 1)
    if "e" in brand:
        return brand.replace("e", "3", 1)
    if "a" in brand:
        return brand.replace("a", "4", 1)
    return brand + "1"


def lookalike_host(rng: random.Random, brand: str) -> str:
    """A sender host that should trip lookalike or brand-mismatch checks.

    Short names such as ``dhl`` skip leet and glued compounds. Those styles
    need a longer string before the feature code will call them an imitation.
    """
    tld = _pick(rng, ["example", "example", "example", "xyz", "top", "click"])
    styles = ["hyphen", "prefix", "subdomain", "www"]
    if len(brand) >= 5:
        styles.extend(["leet", "compound"])
    style = _pick(rng, styles)
    if style == "leet":
        return f"{_leet(brand)}-secure.{tld}"
    if style == "hyphen":
        return f"{brand}-verify.{tld}"
    if style == "prefix":
        return f"secure-{brand}.{tld}"
    if style == "compound":
        return f"{brand}support.{tld}"
    if style == "www":
        return f"www-{brand}-login.{tld}"
    tail = _pick(rng, ["account-check.example", "login-help.example", "secure-center.example"])
    return f"{brand}.{tail}"


def _phish_leaf(rng: random.Random) -> str:
    """Path text that is not a label leak.

    Every synthetic phish used to carry the word "demo" and the scheme http.
    Those shortcuts are gone. Some paths name a login or a payment page, which
    is a real cue, and some reuse the same neutral leaves as campus mail.
    """
    meaningful = ["login", "account", "reset", "verify", "invoice", "pay", "unlock"]
    neutral = ["page", "item", "note", "details", "overview"]
    leaf = _pick(rng, meaningful if rng.random() < 0.65 else neutral)
    return f"{leaf}/{rng.randint(10, 99)}"


def _ip_url(rng: random.Random) -> str:
    host = f"{_pick(rng, DOC_NETS)}.{rng.randint(1, 254)}"
    return f"https://{host}/{_phish_leaf(rng)}"


def _short_url(rng: random.Random) -> str:
    host = _pick(rng, ["bitly.example", "tinyurl.example", "cuttly.example", "shorturl.example"])
    return f"https://{host}/r/{rng.randint(1000, 9999)}"


def _tld_url(rng: random.Random, brand: str) -> str:
    tld = _pick(rng, ["xyz", "top", "click", "zip", "mov"])
    label = _pick(rng, [f"{brand}-check", f"secure-{brand}", f"{brand}login", "reward-claim"])
    return f"https://{label}.{tld}/{_phish_leaf(rng)}"


def _userinfo_url(rng: random.Random) -> str:
    host = f"{_pick(rng, DOC_NETS)}.{rng.randint(1, 254)}"
    disguise = _pick(rng, ["accounts.northwind.edu", "paypal.example", "login.microsoft.example"])
    return f"https://{disguise}@{host}/{_phish_leaf(rng)}"


def _bad_url(rng: random.Random, brand: str, bias: str | None = None) -> str:
    kind = bias or _pick(rng, ["ip", "lookalike", "short", "tld", "userinfo"])
    if kind == "ip":
        return _ip_url(rng)
    if kind == "short":
        return _short_url(rng)
    if kind == "tld":
        return _tld_url(rng, brand)
    if kind == "userinfo":
        return _userinfo_url(rng)
    return f"https://{lookalike_host(rng, brand)}/{_phish_leaf(rng)}"


def _campus_url(rng: random.Random, kind: str) -> str:
    hosts = {
        "canvas": "canvas.northwind.edu",
        "accounts": "accounts.northwind.edu",
        "news": "news.northwind.edu",
        "bookstore": "bookstore.northwind.edu",
        "registrar": "registrar.northwind.edu",
        "status": "status.northwind.edu",
        "bursar": "bursar.northwind.edu",
        "library": "library.northwind.edu",
        "harbor": "online.harborcu.example",
        "events": "events.lumenlecture.example",
        "books": "news.cedarbooks.example",
    }
    leaf = _pick(rng, ["overview", "details", "weekly", "policy", "hours", "item", "page", "note"])
    return f"https://{hosts[kind]}/{leaf}/{rng.randint(10, 99)}"


def _html_extra(rng: random.Random, url: str) -> str:
    """A short inert fragment so the HTML checks have positive examples.

    These are single tags, not pages. The dataset never includes a
    credential-harvesting site.
    """
    roll = rng.random()
    if roll < 0.08:
        return f' <form action="{url}"></form>'
    if roll < 0.14:
        return ' <input type="password">'
    if roll < 0.18:
        return " <script></script>"
    if roll < 0.30:
        return f' Shown link: <a href="{url}">https://accounts.northwind.edu/login</a>.'
    return ""


def _shout(rng: random.Random, subject: str, chance: float = 0.45) -> str:
    if rng.random() < chance:
        return subject.upper() + "!!!"
    return subject


def _row(sender: str, reply_to: str, subject: str, body: str, label: int, category: str) -> dict:
    return {
        "sender": _squash(sender),
        "reply_to": _squash(reply_to),
        "subject": _squash(subject),
        "body": _squash(body),
        "label": int(label),
        "category": category,
    }


def _phish_identity(rng: random.Random, brand: str) -> tuple[str, str]:
    host = lookalike_host(rng, brand)
    local = _pick(rng, ["support", "security", "alert", "notice", "billing", "noreply"])
    display = _pick(
        rng,
        [
            f"{_title(brand)} Security",
            f"{_title(brand)} Support",
            f"{_title(brand)} Account Services",
            f"{_title(brand)} Helpdesk",
        ],
    )
    return f"{display} <{local}@{host}>", host


def gen_suspension(rng: random.Random, i: int, category: str) -> dict:
    """Urgency, a credential ask, and a link that is not the real service."""
    brand = PHISH_BRANDS[i % len(PHISH_BRANDS)]
    name = _pick(rng, FIRST)
    hours = _pick(rng, [2, 4, 12, 24])
    url = _bad_url(rng, brand, bias=_pick(rng, ["ip", "lookalike", "tld", "userinfo"]))
    sender, _host = _phish_identity(rng, brand)
    reply = f"reply@{_pick(rng, REPLY_HOSTS)}"
    subject = _shout(
        rng,
        _pick(
            rng,
            [
                f"Your {_title(brand)} account is suspended",
                f"Final notice for your {_title(brand)} account",
                f"Unusual activity on your {_title(brand)} account",
                f"{_title(brand)} access expires today",
            ],
        ),
    )
    bodies = [
        (
            f"Hello {name}, the {_title(brand)} account linked to this mailbox is suspended. "
            f"Verify your account and enter your password at {url} within {hours} hours "
            f"or the account will be closed."
        ),
        (
            f"Dear customer, we locked a {_title(brand)} profile after unusual activity. "
            f"Act now. Open {url} and confirm your password. Failure to respond within {hours} hours "
            f"keeps the account suspended."
        ),
        (
            f"{_title(brand)} Security: this is a final notice. Confirm your identity at {url} "
            f"within {hours} hours. The account is suspended until you enter your password."
        ),
        (
            f"Hello, unusual activity will close your {_title(brand)} access today. "
            f"Reset your password at {url} immediately. This is a last warning."
        ),
    ]
    body = bodies[i % len(bodies)] + _html_extra(rng, url) + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def gen_password(rng: random.Random, i: int, category: str) -> dict:
    """A password-reset caricature sent from a lookalike domain."""
    brand = PHISH_BRANDS[(i + 3) % len(PHISH_BRANDS)]
    url = _bad_url(rng, brand, bias=_pick(rng, ["lookalike", "ip", "userinfo"]))
    sender, _host = _phish_identity(rng, brand)
    reply = "" if rng.random() < 0.25 else f"reset@{_pick(rng, REPLY_HOSTS)}"
    subject = _pick(
        rng,
        [
            f"Reset your {_title(brand)} password",
            f"{_title(brand)} password check",
            f"Sign-in help for {_title(brand)}",
            f"Your {_title(brand)} password expires today",
        ],
    )
    if rng.random() < 0.25:
        subject = subject.upper() + "!!!"
    bodies = [
        (
            f"A password reset was started for this mailbox. Confirm your password at {url} "
            f"so we can verify your account. If you do nothing within 2 hours the account will be locked."
        ),
        (
            f"Hello, enter your password and username at {url} to finish a {_title(brand)} reset. "
            f"Action required within 24 hours."
        ),
        (
            f"We need you to confirm your identity. Open {url} and type your password. "
            f"The {_title(brand)} profile expires today if this is missed."
        ),
        (
            f"Use {url} and submit your password. {_title(brand)} will mark the mailbox deactivated "
            f"if this reset is ignored. Act now."
        ),
    ]
    body = bodies[i % len(bodies)] + _html_extra(rng, url) + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def gen_invoice(rng: random.Random, i: int, category: str) -> dict:
    """Payment pressure, often with a Reply-To on a different domain."""
    url = _bad_url(rng, "northwind", bias=_pick(rng, ["ip", "tld", "lookalike"]))
    amount = rng.choice([128, 240, 860, 1490, 76])
    invoice = 4000 + i
    if rng.random() < 0.55:
        sender = f"Northwind Accounts Payable <billing@{lookalike_host(rng, 'northwind')}>"
    else:
        sender = f"Accounts Payable <billing@{_pick(rng, FREEMAIL)}>"
    reply = f"payments@{_pick(rng, REPLY_HOSTS)}"
    subject = _pick(
        rng,
        [
            f"Overdue invoice {invoice}",
            f"Confirm payment details for invoice {invoice}",
            f"Invoice {invoice} needs a wire transfer",
            f"Payment overdue on invoice {invoice}",
        ],
    )
    bodies = [
        (
            f"Accounts payable needs you to confirm payment details for overdue invoice {invoice}. "
            f"The balance of ${amount} must be released. Open {url} and enter your password to release the payment."
        ),
        (
            f"Invoice {invoice} is an overdue invoice. Send a wire transfer or pay immediately at {url}. "
            f"Update your billing after you log in and confirm the payment."
        ),
        (
            f"Hello, payment is overdue on invoice {invoice} for ${amount}. "
            f"Confirm payment details at {url}. A gift card is also accepted if a wire transfer is hard."
        ),
        (
            f"Please release the payment for invoice {invoice}. Enter your password at {url} "
            f"and send the bank account number shown on that page. This is an overdue invoice."
        ),
    ]
    body = bodies[i % len(bodies)] + _html_extra(rng, url) + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def gen_it(rng: random.Random, i: int, category: str) -> dict:
    """Help-desk impersonation from a free mailbox or a lookalike school domain."""
    url = _bad_url(rng, "northwind", bias=_pick(rng, ["ip", "tld", "userinfo", "lookalike"]))
    displays = [
        "Northwind IT Helpdesk",
        "Northwind Registrar",
        "Northwind Security",
        "University IT Support",
        "Professor Account Services",
    ]
    display = displays[i % len(displays)]
    if rng.random() < 0.5:
        local = _pick(rng, ["helpdesk", "it.support", "registrar.office", "admin.notice"])
        sender = f"{display} <{local}@{_pick(rng, FREEMAIL)}>"
    else:
        sender = f"{display} <helpdesk@{lookalike_host(rng, 'northwind')}>"
    reply = f"tickets@{_pick(rng, REPLY_HOSTS)}"
    subject = _pick(
        rng,
        [
            "Mailbox directory update",
            "Student account check from IT",
            "Action required: directory password",
            "Helpdesk request to verify your account",
        ],
    )
    bodies = [
        (
            f"This is the IT helpdesk. We are updating the student directory. "
            f"Verify your account and enter your password at {url}. Action required today."
        ),
        (
            f"Hello, the registrar copy of your mailbox must be checked. "
            f"Confirm your identity at {url} within 24 hours or the account will be locked."
        ),
        (
            f"University IT support needs you to submit your password at {url}. "
            f"Failure to respond today will leave the mailbox deactivated."
        ),
        (
            f"A directory review is open. Open {url} and type your password so the helpdesk "
            f"can verify your account. This is a final notice."
        ),
    ]
    body = bodies[i % len(bodies)] + _html_extra(rng, url) + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def gen_prize(rng: random.Random, i: int, category: str) -> dict:
    """A shouty prize caricature with a short link and a gift-card ask."""
    url = _short_url(rng) if rng.random() < 0.7 else _tld_url(rng, "reward")
    amount = _pick(rng, [100, 250, 500, 1000])
    sender_host = _pick(rng, ["claim-reward.xyz", "prize-desk.top", "winner-alert.click", "gift-code.xyz"])
    sender = f"Rewards Desk <prizes@{sender_host}>"
    reply = f"claim@{_pick(rng, REPLY_HOSTS)}"
    subject = _shout(rng, _pick(rng, [
        f"You won a ${amount} gift card",
        "Limited time reward selected for you",
        f"Congratulations, a ${amount} prize is waiting",
        "Final notice about your reward",
    ]), chance=0.85)
    bodies = [
        (
            f"Congratulations. You were selected for a limited time reward of a ${amount} gift card. "
            f"Act now. Open {url} and send the codes from the card. Failure to respond forfeits the prize."
        ),
        (
            f"A ${amount} prepaid card is on hold under your name. Right away, visit {url} and send the codes. "
            f"This offer expires today."
        ),
        (
            f"Lucky draw notice. Pay nothing, but send the gift card codes at {url} within 2 hours. "
            f"This is a last warning and the reward is a limited time claim."
        ),
        (
            f"Your mailbox won a ${amount} gift card. Act now at {url}. Immediately send the codes "
            f"or the prize will be forfeited."
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def gen_lookalike(rng: random.Random, i: int, category: str) -> dict:
    """Calmer wording. The tell is the sender domain, not a wall of capitals."""
    brand = PHISH_BRANDS[(i + 5) % len(PHISH_BRANDS)]
    name = _pick(rng, FIRST)
    url = _bad_url(rng, brand, bias="lookalike")
    sender, _host = _phish_identity(rng, brand)
    # Same-domain reply-to on purpose, so this row is not just a Reply-To mismatch.
    reply_host = sender.split("@", 1)[1].rstrip(">")
    reply = f"{_pick(rng, ['support', 'notice', 'mail'])}@{reply_host}"
    subject = _pick(
        rng,
        [
            f"Document shared with your {_title(brand)} account",
            f"{_title(brand)} sign-in check",
            f"Please review your {_title(brand)} profile",
            f"{_title(brand)} file is waiting",
        ],
    )
    bodies = [
        (
            f"Hello {name}, {_title(brand)} needs you to confirm your identity for this mailbox. "
            f"Open {url} and enter your password. We will not phone you about it."
        ),
        (
            f"Hello {name}, a file was shared to your {_title(brand)} profile. "
            f"Verify your account at {url} and type your password to view it."
        ),
        (
            f"{name}, confirm your password at {url} to keep the {_title(brand)} sign-in active. "
            f"The check is due within 24 hours."
        ),
        (
            f"Hello, this {_title(brand)} notice is only a sign-in check. "
            f"Submit your password using {url}. Unusual activity was noted on the profile."
        ),
    ]
    body = bodies[i % len(bodies)] + _html_extra(rng, url) + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def gen_package(rng: random.Random, i: int, category: str) -> dict:
    """A parcel-fee caricature. The fee request is the point of the example."""
    brand = "dhl" if i % 2 == 0 else "fedex"
    url = _ip_url(rng) if i % 2 == 0 else _tld_url(rng, brand)
    sender = f"{_title(brand)} Support <alert@{lookalike_host(rng, brand)}>"
    reply = f"parcels@{_pick(rng, REPLY_HOSTS)}"
    tracking = 700000 + i
    subject = _pick(
        rng,
        [
            f"{_title(brand)} parcel {tracking} is on hold",
            f"Holding fee for {_title(brand)} delivery {tracking}",
            f"{_title(brand)} delivery needs a payment",
            f"Pay the {_title(brand)} holding fee",
        ],
    )
    bodies = [
        (
            f"{_title(brand)} says parcel {tracking} is on hold. A holding fee of $2.80 is overdue. "
            f"Pay immediately with a gift card and send the codes at {url}."
        ),
        (
            f"Delivery {tracking} will be returned. Pay immediately at {url} using a prepaid card. "
            f"The holding fee is overdue and the window expires today."
        ),
        (
            f"A {_title(brand)} parcel is waiting. Confirm payment details for the holding fee at {url}. "
            f"A gift card is the only method this notice accepts."
        ),
        (
            f"Parcel {tracking} needs a holding fee. Open {url} and send the codes from a ${_pick(rng, [25, 50])} gift card. "
            f"Act now or the parcel notice expires today."
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def gen_refund(rng: random.Random, i: int, category: str) -> dict:
    """An obvious fake refund note. It asks for an identity number outright."""
    url = _bad_url(rng, "irs", bias=_pick(rng, ["ip", "tld", "userinfo"]))
    amount = _pick(rng, [842, 120, 64, 310])
    sender = f"IRS Refund Desk <refunds@{lookalike_host(rng, 'irs')}>"
    reply = f"refunds@{_pick(rng, REPLY_HOSTS)}"
    subject = _shout(rng, _pick(rng, [
        f"Tax refund of ${amount} waiting",
        "Refund desk needs your identity number",
        "Final notice from the refund desk",
        f"${amount} refund on hold",
    ]), chance=0.7)
    bodies = [
        (
            f"A refund of ${amount} is on hold. Send your social security number using {url} "
            f"within 12 hours or the refund is forfeited."
        ),
        (
            f"The refund desk is holding ${amount}. Confirm your identity at {url} and enter your password. "
            f"This is a final notice."
        ),
        (
            f"You must submit your social security number at {url} to release a ${amount} refund. "
            f"Act now. Failure to respond within 2 hours forfeits it."
        ),
        (
            f"Refund case {500 + i} is expiring. Open {url} and type your password plus your social security number. "
            f"The refund of ${amount} expires today."
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, reply, subject, body, 1, category)


def _professor(rng: random.Random) -> tuple[str, str]:
    name, local = _pick(rng, PROFESSORS)
    host = _pick(rng, ["northwind.edu", "cs.northwind.edu", "math.northwind.edu"])
    return f"Prof. {name} <{local}@{host}>", local


def gen_class(rng: random.Random, i: int, category: str) -> dict:
    sender, local = _professor(rng)
    course, topic = COURSES[i % len(COURSES)]
    url = _campus_url(rng, "canvas")
    day = _pick(rng, ["Tuesday", "Wednesday", "Thursday", "Friday"])
    subject = _pick(
        rng,
        [
            f"{course} assignment posted",
            f"{course} reading for {day}",
            f"Notes from {course} on {topic}",
            f"{course} room change",
        ],
    )
    click = f" You can click the course page for the rubric: {url}." if rng.random() < 0.35 else f" The page is {url}."
    bodies = [
        (
            f"Hello {course}, the written work on {topic} is posted. "
            f"Submit it by {day}.{click} Office hours stay in Room 210."
        ),
        (
            f"The {course} session on {topic} will meet in the usual room. "
            f"Bring one question. Materials are on {url}. See you {day}."
        ),
        (
            f"I posted comments on the {topic} drafts for {course}. "
            f"Nothing is due before {day}.{click} Reply in person if a note looks wrong."
        ),
        (
            f"{course} reminder: the lab writeup covers {topic}. "
            f"Use {url} and bring the draft to class on {day}."
        ),
    ]
    reply = f"{local}@northwind.edu"
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, reply, subject, body, 0, category)


def gen_coworker(rng: random.Random, i: int, category: str) -> dict:
    display, local = _person(rng)
    sender = f"{display} <{local}@northwind.edu>"
    other = _pick(rng, FIRST)
    subject = _pick(
        rng,
        [
            "Lab notes from Tuesday",
            "Study group this week",
            "Slides I promised",
            f"Question on the {COURSES[i % len(COURSES)][1]}",
            "Library table after 15:00",
        ],
    )
    bodies = [
        (
            f"Hi {other}, I left the lab notes on the shared drive. "
            f"The switch config we reviewed is in the Tuesday folder. "
            f"I will be in the library after 15:00 if you want to walk through question 4."
        ),
        (
            f"Hi {other}, the reading group is meeting at the campus coffee shop on Thursday. "
            f"Bring the two pages you liked. I can carry the printouts."
        ),
        (
            f"{other}, the slide deck is in our project folder. "
            f"I fixed the chart labels and left the speaker notes at the bottom. Thanks for the earlier pass."
        ),
        (
            f"Hi {other}, I can look at the lab notebook with you before class. "
            f"The quiet room on the second floor is free at 16:00. No need to bring anything new."
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, "", subject, body, 0, category)


def gen_receipt(rng: random.Random, i: int, category: str) -> dict:
    """A receipt whose From domain matches the store it names."""
    order = 48000 + i
    amount = _pick(rng, ["12.40", "24.00", "42.15", "64.20", "18.75"])
    kind = i % 3
    if kind == 0:
        sender = "Campus Bookstore <receipts@bookstore.northwind.edu>"
        reply = "receipts@northwind.edu"
        url = _campus_url(rng, "bookstore")
        subject = f"Bookstore receipt {order}"
        body = (
            f"Thanks for your order {order}. Item: course reader. Amount paid: ${amount}. "
            f"You can view it at {url}. This mailbox is unattended. {_reference(rng)}"
        )
    elif kind == 1:
        sender = "Harbor Credit Union <statements@harborcu.example>"
        reply = "statements@harborcu.example"
        url = _campus_url(rng, "harbor")
        subject = f"Harbor statement {order}"
        body = (
            f"Your monthly statement is ready. Ending balance: ${amount}. "
            f"View it by opening the credit union site yourself: {url}. "
            f"Harbor Credit Union will not ask you to email a password. {_reference(rng)}"
        )
    else:
        sender = "Cedar Books <news@cedarbooks.example>"
        reply = ""
        url = _campus_url(rng, "books")
        subject = f"Cedar Books order {order}"
        body = (
            f"Your paperback order {order} shipped. Amount paid: ${amount}. "
            f"The receipt page is {url}. Reply to the store if a title is wrong. {_reference(rng)}"
        )
    return _row(sender, reply, subject, body, 0, category)


def gen_security_notice(rng: random.Random, i: int, category: str) -> dict:
    """A calm notice. It tells the reader to use a campus address they type themselves."""
    sender = "Northwind IT <it-news@northwind.edu>"
    reply = "it-news@cs.northwind.edu"
    url = _campus_url(rng, "accounts")
    status = _campus_url(rng, "status")
    subject = _pick(
        rng,
        [
            "Account reminder from Northwind IT",
            "How campus account mail works",
            "Routine note from the identity team",
            "Campus account page, for your bookmarks",
        ],
    )
    bodies = [
        (
            f"Northwind IT is writing with a routine reminder. Staff will not ask you to send a password by email. "
            f"If you want to review your account, type {url} into your browser yourself. "
            f"The status page is {status}. No reply is required."
        ),
        (
            f"This is the monthly identity note. Unexpected mail sometimes pretends to be the help desk. "
            f"This note comes from northwind.edu and it does not include a sign-in form. "
            f"The account page you already use is {url}."
        ),
        (
            f"A calm reminder from campus IT: keep your mailbox phrase private. "
            f"We will not request a password in a message. The self-service page is {url}. "
            f"Outages, when they happen, are listed on {status}."
        ),
        (
            f"Hello from Northwind IT. If a message pushes you to hurry, close it and open {url} yourself. "
            f"This reminder does not need an answer. Service health is on {status}."
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, reply, subject, body, 0, category)


def gen_newsletter(rng: random.Random, i: int, category: str) -> dict:
    sender = "Northwind Weekly <news@news.northwind.edu>"
    reply = "news@northwind.edu"
    first = _campus_url(rng, "news")
    second = _campus_url(rng, "library")
    third = _campus_url(rng, "events") if rng.random() < 0.5 else ""
    extra = f" The lecture list is {third}." if third else ""
    subject = _pick(
        rng,
        [
            "This week at Northwind",
            "Campus weekly: seminars and hours",
            "Library hours and the Friday seminar",
            "Student weekly for this week",
        ],
    )
    bodies = [
        (
            f"This week at Northwind: the AI seminar is Thursday and the chess club meets Friday. "
            f"Details are on {first}. Library hours are on {second}.{extra} "
            f"You are receiving this because you subscribed at the student portal."
        ),
        (
            f"The campus weekly lists extended library hours during finals. See {second}. "
            f"Club notes are collected at {first}.{extra} Unsubscribe from the student portal if you prefer."
        ),
        (
            f"Hello, the lecture series posted a new recording. {first} has the summary and {second} has quiet-room hours."
            f"{extra} No reply is needed."
        ),
        (
            f"Weekly note: volunteering at the garden is Saturday, and the observatory is open Friday night. "
            f"Read more at {first} and {second}.{extra}"
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, reply, subject, body, 0, category)


def gen_meeting(rng: random.Random, i: int, category: str) -> dict:
    display, local = _person(rng)
    sender = f"{display} <{local}@northwind.edu>"
    other = _pick(rng, FIRST)
    day = _pick(rng, ["Monday", "Tuesday", "Wednesday", "Thursday"])
    subject = _pick(
        rng,
        [
            f"Project check-in on {day}",
            "Can we move our meeting",
            f"Room booked for {day}",
            "Advising slot this week",
        ],
    )
    bodies = [
        (
            f"{other}, can we move the project check-in to {day} at 11:00? "
            f"I booked the small room in the engineering building. Reply if that time is bad."
        ),
        (
            f"Hi {other}, I am holding a table at the library on {day} afternoon for the group. "
            f"Bring the outline if you have it. I will bring markers."
        ),
        (
            f"{other}, advising appointments on {day} are in Room 118. "
            f"Come with your course list. The visit is twenty minutes."
        ),
        (
            f"Hi {other}, the club planning meeting is {day} at 17:30 in the student lounge. "
            f"We will pick a date for the poster session and then leave."
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, "", subject, body, 0, category)


def gen_facilities(rng: random.Random, i: int, category: str) -> dict:
    """Legitimate mail that still says "urgent", so the word is not a shortcut."""
    sender = "Northwind Facilities <facilities@northwind.edu>"
    reply = "facilities@northwind.edu"
    url = _campus_url(rng, "status")
    place = _pick(rng, ["engineering building", "library basement", "north lab", "residence hall B"])
    subject = _pick(
        rng,
        [
            f"Facilities notice: water shutoff in the {place}",
            f"Urgent repair tonight in the {place}",
            "Building access change tonight",
            f"Heat outage in the {place}",
        ],
    )
    bodies = [
        (
            f"Facilities notice: there is an urgent water shutoff in the {place} tonight from 22:00 to 23:00. "
            f"No action is needed. The lab will reopen in the morning. Status page: {url}."
        ),
        (
            f"An urgent repair is underway in the {place}. Classes in that wing move to the usual backup room. "
            f"No action is needed beyond checking {url} in the morning."
        ),
        (
            f"The {place} has an urgent facilities issue tonight. Doors reopen at 07:00. "
            f"This note is informational. Updates are on {url}."
        ),
        (
            f"Please be aware of an urgent heat outage in the {place} this evening. "
            f"No reply is required. The morning status will be posted at {url}."
        ),
    ]
    body = bodies[i % len(bodies)] + " " + _reference(rng)
    return _row(sender, reply, subject, body, 0, category)


def gen_tuition(rng: random.Random, i: int, category: str) -> dict:
    sender = "Northwind Bursar <bursar@bursar.northwind.edu>"
    reply = "bursar@northwind.edu"
    url = _campus_url(rng, "bursar")
    term = _pick(rng, ["fall term", "spring term", "summer session"])
    amount = _pick(rng, ["3,240.00", "1,180.00", "860.50", "2,410.00"])
    subject = _pick(
        rng,
        [
            f"Tuition receipt for the {term}",
            f"{term.title()} bursar receipt",
            "Your campus billing receipt",
            f"Receipt posted for the {term}",
        ],
    )
    body = (
        f"Your tuition receipt for the {term} is ready. Amount paid: ${amount}. "
        f"The receipt sits on {url}. This note does not ask you to send money or a mailbox phrase. "
        f"Questions can wait for the bursar window on Thursday. {_reference(rng)}"
    )
    return _row(sender, reply, subject, body, 0, category)


PLAN = [
    ("account_suspension", gen_suspension, 48),
    ("password_reset", gen_password, 44),
    ("invoice_payment", gen_invoice, 40),
    ("it_impersonation", gen_it, 36),
    ("prize_claim", gen_prize, 32),
    ("lookalike_brand", gen_lookalike, 40),
    ("package_fee", gen_package, 24),
    ("refund_claim", gen_refund, 16),
    ("class_announcement", gen_class, 56),
    ("coworker_note", gen_coworker, 48),
    ("receipt", gen_receipt, 36),
    ("security_notice", gen_security_notice, 36),
    ("newsletter", gen_newsletter, 36),
    ("meeting", gen_meeting, 36),
    ("facilities_note", gen_facilities, 16),
    ("tuition_notice", gen_tuition, 16),
]


def audit_rows(rows: list[dict]) -> list[str]:
    """Every phishing row needs a real signal. Ordinary legit rows should be quiet."""
    problems: list[str] = []
    body_counts: dict[str, int] = {}
    body_labels: dict[str, set[int]] = {}
    for row in rows:
        body_counts[row["body"]] = body_counts.get(row["body"], 0) + 1
        body_labels.setdefault(row["body"], set()).add(int(row["label"]))
    extra_copies = sum(count - 1 for count in body_counts.values())
    # A few short coworker notes can repeat. Identical phishing text would
    # leak across the train/test split, so only a small legit overlap is allowed.
    if extra_copies > 20:
        problems.append(f"too many duplicate bodies ({extra_copies})")
    if any(
        labels != {0} and count > 1
        for body, count in body_counts.items()
        for labels in [body_labels[body]]
    ):
        problems.append("duplicate body is not a repeated legitimate note")
    for row in rows:
        flag_ids = {flag["id"] for flag in analyze(row).flags}
        strong = flag_ids & STRONG_FLAG_IDS
        if int(row["label"]) == 1 and not strong:
            problems.append(f"phishing without a strong signal ({row['category']}): {flag_ids or 'none'}")
        if int(row["label"]) == 0 and row["category"] != "facilities_note" and flag_ids:
            problems.append(f"unexpected flag on legit {row['category']}: {sorted(flag_ids)}")
        if int(row["label"]) == 0 and row["category"] == "facilities_note":
            extra = flag_ids - {"urgency_hits"}
            if extra or "urgency_hits" not in flag_ids:
                problems.append(f"facilities flags were {sorted(flag_ids)}")
    labels = [row["label"] for row in rows]
    if labels.count(1) < 200 or labels.count(0) < 200:
        problems.append("dataset is below the planned class size")
    if abs(labels.count(1) - labels.count(0)) > 40:
        problems.append("dataset is far from balanced")
    return problems


def generate_emails(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for category, function, count in PLAN:
        for index in range(count):
            rows.append(function(rng, index, category))
    rng.shuffle(rows)
    for number, row in enumerate(rows, start=1):
        row["email_id"] = f"mail-{number:04d}"
    problems = audit_rows(rows)
    if problems:
        preview = "\n".join(problems[:12])
        raise RuntimeError(f"Teaching-set audit failed ({len(problems)} issues).\n{preview}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def main() -> None:
    rows = generate_emails(seed=42)
    write_csv(DATA_PATH, rows)
    labels = [row["label"] for row in rows]
    print(f"Wrote {len(rows)} emails to {DATA_PATH}")
    print(f"legitimate={labels.count(0)} phishing={labels.count(1)}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
