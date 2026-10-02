import math
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
_MAX_CONTEXT_CHARS = 3000
_CHUNK_WORDS = 180
_CHUNK_OVERLAP = 30


def _meaningful_words(query):
    words = re.findall(r"[a-z0-9]+", (query or "").lower())
    return {word for word in words if len(word) >= 3 and word not in _STOP_WORDS}


def _chunks(text):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + _CHUNK_WORDS)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - _CHUNK_OVERLAP, start + 1)
    return chunks


def search_knowledge(query, max_results=_MAX_RESULTS):
    words = _meaningful_words(query)
    if not words or not KNOWLEDGE_DIR.exists():
        return []

    matches = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        for index, chunk in enumerate(_chunks(text)):
            lowered = chunk.lower()
            hits = sum(
                1 for word in words
                if re.search(rf"\b{re.escape(word)}\b", lowered)
            )
            if hits < _MIN_SCORE:
                continue

            density = hits / max(1, len(_meaningful_words(chunk)))
            score = hits + math.log1p(density * 100)
            matches.append(
                {
                    "score": score,
                    "source": path.name,
                    "chunk": index + 1,
                    "text": chunk,
                    "matched_terms": sorted(
                        word for word in words
                        if re.search(rf"\b{re.escape(word)}\b", lowered)
                    ),
                }
            )

    matches.sort(key=lambda item: (-item["score"], item["source"], item["chunk"]))
    return matches[:max(1, min(int(max_results), 8))]


def build_rag_context(query):
    results = search_knowledge(query)
    if not results:
        return ""

    lines = ["Relevant knowledge context:"]
    used = 0
    for result in results:
        text = result["text"]
        remaining = _MAX_CONTEXT_CHARS - used
        if remaining <= 0:
            break
        snippet = text[:remaining]
        lines.append(
            f"[{result['source']}#{result['chunk']}] {snippet}"
        )
        used += len(snippet)

    return "\n".join(lines)


def knowledge_sources(query):
    return [
        {
            "source": item["source"],
            "chunk": item["chunk"],
            "score": round(item["score"], 3),
            "matched_terms": item["matched_terms"],
        }
        for item in search_knowledge(query)
    ]
