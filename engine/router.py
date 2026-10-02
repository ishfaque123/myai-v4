import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    target: str
    reason: str
    confidence: float


_CALCULATOR_PATTERNS = (
    r"\b(calculate|calculator|sum|subtract|minus|plus|multiply|divide|percentage|percent|\d+\s*[+\-*/]\s*\d+)\b",
    r"\b(how much is|what is)\s+\d",
)

_CLOUD_PATTERNS = (
    r"\b(compare|difference|analy[sz]e|analysis|research|debug|architecture|algorithm)\b",
    r"\b(code|program|python|javascript|typescript|api|database|github|sql|regex)\b",
    r"\b(legal|law|policy|business|strategy|technical|documentation)\b",
    r"\b(step[- ]by[- ]step|in detail|deep|detailed|complex)\b",
    r"\b(explain)\b.{0,80}\b(why|how|work|works)\b",
)

_SIMPLE_PATTERNS = (
    r"^(hi|hello|hey|salam|assalam|aoa|thanks|thank you|ok|okay|bye)\b",
    r"\b(what is your name|who are you|your name)\b",
)

_KNOWLEDGE_PATTERNS = (
    r"\b(what is|who is|when was|where is|capital of|define|meaning of)\b",
    r"\b(kya hai|kon hai|kab tha|kahan hai|kiya hai)\b",
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

    words = text.split()

    if any(re.search(pattern, text) for pattern in _CALCULATOR_PATTERNS):
        return Route("calculator", "deterministic_calculation", 0.98)

    if len(text) >= 220:
        return Route("cloud", "long_request", 0.90)

    if any(re.search(pattern, text) for pattern in _CLOUD_PATTERNS):
        return Route("cloud", "complex_or_reasoning_task", 0.88)

    if any(re.search(pattern, text) for pattern in _SIMPLE_PATTERNS):
        return Route("local", "simple_conversation", 0.94)

    if any(re.search(pattern, text) for pattern in _KNOWLEDGE_PATTERNS):
        return Route("local", "knowledge_or_rag", 0.84)

    if text.endswith("?") and len(words) >= 14:
        return Route("cloud", "detailed_question", 0.82)

    if len(words) <= 10:
        return Route("local", "short_or_normal_request", 0.78)

    return Route("local", "normal_request", 0.70)
