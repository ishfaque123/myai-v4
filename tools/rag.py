from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE / "knowledge"


def build_rag_context(query):
    words = set((query or "").lower().split())
    if not words or not KNOWLEDGE_DIR.exists():
        return ""
    matches = []
    for path in KNOWLEDGE_DIR.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        score = sum(1 for word in words if word in text.lower())
        if score:
            matches.append((score, path.name, text[:1200]))
    matches.sort(reverse=True)
    if not matches:
        return ""
    return "Relevant knowledge context:\n" + "\n".join(f"[{name}] {text}" for _, name, text in matches[:4])
