import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE / "knowledge"

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of",
    "on", "or", "the", "to", "what", "when", "where", "who", "why",
    "with", "you", "your",
}

_MIN_SCORE = 2
_MAX_RESULTS = 4
_MAX_CONTEXT_CHARS = 1200


def _meaningful_words(query):
    words = re.findall(r"[a-z0-9]+", (query or "").lower())
    return {word for word in words if len(word) >= 3 and word not in _STOP_WORDS}


def build_rag_context(query):
    words = _meaningful_words(query)
    if not words or not KNOWLEDGE_DIR.exists():
        return ""

    matches = []
    for path in KNOWLEDGE_DIR.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        lowered = text.lower()
        score = sum(
            1 for word in words
            if re.search(rf"\b{re.escape(word)}\b", lowered)
        )
        if score < _MIN_SCORE:
            continue

        matches.append((score, path.name, text[:_MAX_CONTEXT_CHARS]))

    matches.sort(key=lambda item: (-item[0], item[1]))
    if not matches:
        return ""

    return "Relevant knowledge context:\n" + "\n".join(
        f"[{name}] {text}" for _, name, text in matches[:_MAX_RESULTS]
    )
