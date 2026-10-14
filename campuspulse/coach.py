"""Offline retrieval with optional local LLM generation through Ollama."""

import json
import os
import re
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .contracts import validate_schema_input
from .data import validate_input
from .privacy import LOCAL_CHAT_URL, open_local_llm


CARDS = [
    {
        "id": "connection",
        "title": "A small connection",
        "text": (
            "If you would like, send one friend a short message "
            "or join a campus activity you enjoy."
        ),
        "tags": "social friends connection alone lonely isolated community",
        "source": (
            "Project-authored general support suggestion; "
            "not clinical evidence."
        ),
    },
    {
        "id": "routine",
        "title": "Make room for a routine",
        "text": (
            "If it suits you, try a consistent wind-down time "
            "and notice how your routine feels this week."
        ),
        "tags": "sleep bedtime routine tired rest irregular variability",
        "source": (
            "Project-authored general support suggestion; "
            "not clinical evidence."
        ),
    },
    {
        "id": "movement",
        "title": "A brief change of scene",
        "text": (
            "If comfortable and accessible, consider a short walk "
            "or another gentle activity you enjoy."
        ),
        "tags": "active activity exercise movement walk sedentary",
        "source": (
            "Project-authored general support suggestion; "
            "not clinical evidence."
        ),
    },
    {
        "id": "balance",
        "title": "Choose a small break",
        "text": (
            "If you want, pause one screen-based task for a few "
            "minutes and choose a restful break."
        ),
        "tags": "screen study work balance break phone overwhelmed",
        "source": (
            "Project-authored general support suggestion; "
            "not clinical evidence."
        ),
    },
]


FOCUS = {
    "general": (
        "A general suggestion",
        "connection routine break",
    ),
    "connection": (
        "Connection with others",
        "social friends connection community",
    ),
    "routine": (
        "Sleep and routine",
        "sleep bedtime routine rest",
    ),
    "movement": (
        "Movement",
        "activity exercise movement walk",
    ),
    "balance": (
        "Study and screen breaks",
        "screen study balance break",
    ),
}


OPTIONAL_OPENINGS = (
    "if you would like",
    "if you'd like",
    "if you’d like",
    "if you want",
    "if it suits you",
    "if comfortable and accessible",
)


CORPUS = [
    f"{card['title']} {card['text']} {card['tags']}"
    for card in CARDS
]

VECTORIZER = TfidfVectorizer(stop_words="english").fit(CORPUS)
MATRIX = VECTORIZER.transform(CORPUS)


def retrieve(query):
    """Retrieve the two most similar authored support cards."""
    similarity = (
        MATRIX @ VECTORIZER.transform([query]).T
    ).toarray().ravel()

    indices = np.argsort(-similarity, kind="stable")[:2]

    return [
        {
            **CARDS[index],
            "similarity": round(float(similarity[index]), 3),
        }
        for index in indices
    ]


def support(payload, consent=False, schema=None):
    """Return an optional suggestion after explicit consent."""
    if consent is not True:
        return {
            "mode": "disabled",
            "message": "Support suggestions are off until you opt in.",
            "sources": [],
        }

    values = (
        validate_schema_input(payload, schema)
        if schema
        else validate_input(payload)
    )

    focus = payload.get("weekly_focus", "general")

    if not isinstance(focus, str) or focus not in FOCUS:
        raise ValueError("Choose a supported weekly focus.")

    topics = []

    if schema:
        # Research features cannot be translated directly into
        # personal support needs.
        # Phone-locked time is not screen time.
        # Sensed conversation is not total social contact.
        topics = ["connection routine break"]
        sources = retrieve(" ".join(topics))
    else:
        sources = None

    # These rules apply only to the fictional demo schema.
    if not schema and values["social_minutes"] < 45:
        topics.append("social connection friends")

    if not schema and (
        values["sleep_hours"] < 6
        or values["routine_variability"] > 1.8
    ):
        topics.append("sleep bedtime routine")

    if not schema and values["active_minutes"] < 25:
        topics.append("activity movement")

    if not schema and values["screen_hours"] > 6:
        topics.append("screen break balance")

    sources = sources or retrieve(
        " ".join(topics) or "connection routine break"
    )

    # Explicit user focus takes priority over general retrieval.
    if focus != "general":
        topics = [FOCUS[focus][1]]
        sources = retrieve(topics[0])[:1]

    if focus != "general":
        reason = (
            f"You chose {FOCUS[focus][0].lower()} for this week."
        )
    else:
        reason = "You requested a general suggestion."

    reason += (
        " This suggestion is not selected by your predicted score."
    )

    def result(mode, message, note):
        return {
            "mode": mode,
            "message": message,
            "sources": sources,
            "note": note,
            "reason": reason,
            "weekly_focus": focus,
        }

    fallback = sources[0]["text"]
    model = os.getenv("OLLAMA_MODEL", "").strip()

    if not model:
        return result(
            "retrieval",
            fallback,
            "Offline TF-IDF retrieval. No LLM generated this message.",
        )

    # Only broad topics and authored cards go to the local service.
    # Identifiers, exact sensor values and predictions are excluded.
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite one provided card as one complete, "
                    "optional sentence under 40 words. "
                    "Preserve its action and any "
                    "comfort/accessibility condition. "
                    "For movement, keep both words 'comfortable' "
                    "and 'accessible' and an action such as 'consider'. "
                    "Do not add explanations, promised outcomes "
                    "or benefits, even tentative ones. "
                    "Do not claim enhanced or improved mood, energy, "
                    "wellbeing, relaxation or sleep. "
                    "Begin with an optional opening such as "
                    "'If you would like', 'If you want', "
                    "'If it suits you', or "
                    "'If comfortable and accessible'. "
                    "Repeating the provided card is acceptable. "
                    "Return only the suggestion, without a heading, "
                    "quotation marks or follow-up question. "
                    "Do not mention diagnoses, risk scores, medication, "
                    "treatment, certainty, or medical advice. "
                    "Do not invent resources or links."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "topics": topics,
                        "cards": [
                            {
                                "id": card["id"],
                                "text": card["text"],
                            }
                            for card in sources
                        ],
                    }
                ),
            },
        ],
        "options": {
            "temperature": 0.2,
            "num_predict": 100,
        },
    }

    try:
        request = urllib.request.Request(
            LOCAL_CHAT_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )

        with open_local_llm(request, timeout=20) as response:
            response_data = json.load(response)

        generated = response_data["message"]["content"]

        if not isinstance(generated, str):
            raise ValueError("The model response was not text.")

        generated = generated.strip()
        lowered = generated.lower()

        # Conservative heuristics, not a safety or grounding guarantee.
        blocked = re.search(
            r"diagnos|depress|disorder|medicat|treat|suicid|"
            r"guarantee|https?://|\d",
            generated,
            re.I,
        )

        added_benefit = re.search(
            r"benefici|boost|improv|enhanc|reduc|calm|relax|"
            r"grounded|energy|mood|well.?being",
            generated,
            re.I,
        )

        # Both the assignment and check remain inside this block.
        if sources[0]["id"] == "movement":
            preserves_conditions = (
                "comfortable" in lowered
                and "accessible" in lowered
            )

            if not preserves_conditions:
                raise ValueError(
                    "Output removed the movement accessibility conditions."
                )

        if added_benefit:
            raise ValueError(
                "Output added an unsupported benefit."
            )

        has_optional_opening = lowered.startswith(OPTIONAL_OPENINGS)

        if (
            not generated
            or len(generated.split()) > 60
            or blocked
            or not has_optional_opening
        ):
            raise ValueError(
                "Output did not pass the basic filter."
            )

        return result(
            "local_llm",
            generated,
            "Local LLM response with basic filtering; "
            "not clinically validated.",
        )

    except Exception as error:
        failure_note = (
            "Output rejected by the basic response checks."
            if isinstance(error, ValueError)
            else "Local LLM request failed or returned "
            "an unexpected response."
        )

        return result(
            "retrieval_fallback",
            fallback,
            failure_note + " Showing the authored suggestion.",
        )