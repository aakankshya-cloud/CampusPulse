"""Local retrieval of curated support cards."""

import hashlib
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


PATH = Path(__file__).resolve().parents[1] / "configs" / "support_cards.json"
CARDS = json.loads(PATH.read_text(encoding="utf-8"))

if len({card["id"] for card in CARDS}) != len(CARDS):
    raise ValueError("Support card IDs must be unique.")

BY_ID = {card["id"]: card for card in CARDS}

CORPUS = [
    f"{card['title']} {card['text']} {card['tags']}"
    for card in CARDS
]

VECTORIZER = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
).fit(CORPUS)

MATRIX = VECTORIZER.transform(CORPUS)


def unsupported_request(query):
    """Limited keyword rules, not a general intent classifier."""
    q = query.casefold()

    if re.search(
        r"\b(remind me|set (?:an? |the )?(?:reminder|alarm))\b", q
    ):
        return (
            "This app cannot schedule reminders or send notifications. "
            "Use a calendar or alarm app for that request."
        )

    if re.search(r"\b(timetable|study plan|revision plan)\b", q):
        return (
            "This app offers short routine suggestions; "
            "it cannot build a personalized study timetable."
        )

    if re.search(
        r"\b(chemistry|reactions|memor\w*|remember|retention)\b"
        r"|nothing (?:went|goes) in",
        q,
    ):
        return (
            "Subject tutoring and memory strategies are not covered "
            "by this collection. No matching advice is available here."
        )

    if re.search(
        r"\b(presentation|public speaking|exams?|revision)\b", q
    ):
        return (
            "Exam preparation and presentation coaching are not covered "
            "by this collection. You can choose a routine or "
            "study-break topic separately."
        )

    return None


def search_text(query):
    """Normalize a small set of words and clarify phone-focus queries."""
    q = query.casefold()
    q = re.sub(r"\b(sleeping|asleep|sleeps)\b", "sleep", q)
    q = re.sub(r"\b(studying|studied)\b", "study", q)
    q = re.sub(
        r"\bprocrastinat\w*\b",
        "study single assignment",
        q,
    )

    phone_context = re.search(r"\b(phone|notifications?)\b", q)
    focus_context = re.search(
        r"\b(focus|distract\w*|interrupt\w*|study)\b", q
    )
    bedtime_context = re.search(r"\b(sleep|bed|bedtime)\b", q)

    if phone_context and focus_context and not bedtime_context:
        q += " notifications distraction notifications"

    return q


def retrieve(query, topic=None, limit=2):
    if not isinstance(query, str) or len(query) > 240:
        raise ValueError(
            "Support query must be text of at most 240 characters."
        )

    if unsupported_request(query):
        return []

    query = search_text(query)
    scores = (
        MATRIX @ VECTORIZER.transform([query]).T
    ).toarray().ravel()

    eligible = [
        i for i, card in enumerate(CARDS)
        if topic is None or card["topic"] == topic
    ]

    ranked = sorted(
        eligible,
        key=lambda i: (-scores[i], i),
    )[:limit]

    return [
        {
            **CARDS[i],
            "similarity": round(float(scores[i]), 4),
        }
        for i in ranked
        if scores[i] > 0
    ]


def approved_wordings(card):
    result = [card["text"]]

    if card["topic"] == "movement":
        tail = card["text"].removeprefix(
            "If comfortable and accessible, "
        )
        result.append(
            "If you would like, and it feels comfortable "
            "and accessible, " + tail
        )
    elif card["text"].startswith("If it suits you, "):
        result.append(
            card["text"].replace(
                "If it suits you, ",
                "If you would like, ",
                1,
            )
        )
    elif card["text"].startswith("If you want, "):
        result.append(
            card["text"].replace(
                "If you want, ",
                "If you would like, ",
                1,
            )
        )

    return result


def corpus_sha256():
    return hashlib.sha256(PATH.read_bytes()).hexdigest()