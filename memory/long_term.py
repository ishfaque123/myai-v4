import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROFILE_DIR = BASE / "memory" / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

MAX_FACTS = 30
MAX_FACT_CHARS = 240

_PATTERNS = (
    (re.compile(r"\bmy name is\s+(.+)$", re.I), "name"),
    (re.compile(r"\bi(?:'m| am)\s+from\s+(.+)$", re.I), "location"),
    (re.compile(r"\bi (?:like|love|prefer)\s+(.+)$", re.I), "preference"),
    (re.compile(r"\bi (?:use|work with)\s+(.+)$", re.I), "context"),
)

_STYLE_PATTERNS = (
    (re.compile(r"\b(?:only|just)\s+sindhi\s+roman(?:\s+words)?\b", re.I), "language_style: Sindhi Roman"),
    (re.compile(r"\bsindhi\s+roman(?:\s+words)?\b", re.I), "language_style: Sindhi Roman"),
    (re.compile(r"\broman\s+sindhi\b", re.I), "language_style: Sindhi Roman"),
)

_BLOCKED = re.compile(
    r"\b(password|passcode|otp|one[- ]time password|secret|api key|token|"
    r"credit card|debit card|bank account|cnic|social security|medical|"
    r"diagnosis|self[- ]harm|suicide)\b",
    re.I,
)

def _safe_id(value):
    value = re.sub(r"[^a-zA-Z0-9_-]", "", str(value or "default"))[:64]
    return value or "default"

def _path(session_id):
    return PROFILE_DIR / f"{_safe_id(session_id)}.json"

def load_profile(session_id="default"):
    path = _path(session_id)
    if not path.exists():
        return {"facts": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"facts": []}
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            facts = []
        return {"facts": [str(x)[:MAX_FACT_CHARS] for x in facts[-MAX_FACTS:]]}
    except (OSError, ValueError, TypeError):
        return {"facts": []}

def save_profile(session_id, profile):
    facts = profile.get("facts", [])[-MAX_FACTS:]
    _path(session_id).write_text(
        json.dumps({"facts": facts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def extract_fact(message):
    text = " ".join((message or "").strip().split())
    if not text or _BLOCKED.search(text):
        return None

    for pattern, fact in _STYLE_PATTERNS:
        if pattern.search(text):
            return fact

    for pattern, label in _PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip(" .!?")
            if 1 <= len(value) <= MAX_FACT_CHARS:
                return f"{label}: {value}"
    return None

def remember_from_message(session_id, message):
    fact = extract_fact(message)
    if not fact:
        return False

    profile = load_profile(session_id)
    facts = profile["facts"]
    if fact not in facts:
        facts.append(fact)
        profile["facts"] = facts[-MAX_FACTS:]
        save_profile(session_id, profile)
        return True
    return False

def build_memory_context(session_id):
    facts = load_profile(session_id).get("facts", [])
    if not facts:
        return ""

    lines = [
        "Long-term memory about this user:",
        *[f"- {fact}" for fact in facts[-MAX_FACTS:]],
        "Use this only when relevant. Do not invent additional facts.",
    ]
    return "\n".join(lines)
