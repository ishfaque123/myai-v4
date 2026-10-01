import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    target: str
    reason: str
    confidence: float


# These patterns describe task intent, not specific providers/models.
_CLOUD_PATTERNS = (
    r"\b(explain|why|how|compare|difference|analy[sz]e|analysis|research|reason|solve|debug|write|create|design|plan)\b",
    r"\b(code|program|python|javascript|typescript|api|database|github|sql|regex|algorithm)\b",
    r"\b(legal|law|policy|business|strategy|technical|architecture|documentation)\b",
    r"\b(step[- ]by[- ]step|in detail|deep|detailed|complex)\b",
)

_SIMPLE_PATTERNS = (
    r"^(hi|hello|hey|salam|assalam|aoa|thanks|thank you|ok|okay|bye)\b",
    r"\b(what is your name|who are you|your name)\b",
)

_QUESTION_STARTERS = (
    "what", "who", "when", "where", "which", "can", "could",
    "would", "should", "is", "are", "do", "does", "did",
)


def _normalize(text):
    text = (text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def route_message(message):
    text = _normalize(message)

    if not text:
        return Route("local", "empty_or_invalid", 1.0)

    # Calculator is handled before model routing by core.py.
    if len(text) >= 220:
        return Route("cloud", "long_request", 0.90)

    if any(re.search(pattern, text) for pattern in _CLOUD_PATTERNS):
        return Route("cloud", "complex_or_reasoning_task", 0.88)

    words = text.split()
    if text.endswith("?") and len(words) >= 10:
        return Route("cloud", "detailed_question", 0.82)

    if any(re.search(pattern, text) for pattern in _SIMPLE_PATTERNS):
        return Route("local", "simple_conversation", 0.94)

    if len(words) <= 8 and not any(word in _QUESTION_STARTERS for word in words[:1]):
        return Route("local", "short_request", 0.78)

    return Route("local", "normal_request", 0.70)
