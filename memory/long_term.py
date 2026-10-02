import json
import re
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROFILE_DIR = BASE / "memory" / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

MAX_FACTS = 40
MAX_FACT_CHARS = 240

_PATTERNS = (
    (re.compile(r"\bmy name is\s+(.+)$", re.I), "name", 0.95),
    (re.compile(r"\bi(?:'m| am)\s+from\s+(.+)$", re.I), "location", 0.9),
    (re.compile(r"\bi (?:like|love|prefer)\s+(.+)$", re.I), "preference", 0.8),
    (re.compile(r"\bi (?:use|work with)\s+(.+)$", re.I), "context", 0.75),
)

_STYLE_PATTERNS = (
    (re.compile(r"\b(?:only|just)\s+sindhi\s+roman(?:\s+words)?\b", re.I), "language_style", "Sindhi Roman", 0.9),
    (re.compile(r"\bsindhi\s+roman(?:\s+words)?\b", re.I), "language_style", "Sindhi Roman", 0.9),
    (re.compile(r"\broman\s+sindhi\b", re.I), "language_style", "Sindhi Roman", 0.9),
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


def _now():
    return datetime.now(timezone.utc).isoformat()


def _normalize_fact(item):
    if isinstance(item, str):
        return {
            "key": "legacy",
            "value": item[:MAX_FACT_CHARS],
            "confidence": 0.5,
            "updated_at": "",
        }

    if not isinstance(item, dict):
        return None

    key = str(item.get("key") or "").strip()[:40]
    value = str(item.get("value") or "").strip()[:MAX_FACT_CHARS]
    if not value:
        return None

    try:
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "key": key or "fact",
        "value": value,
        "confidence": confidence,
        "updated_at": str(item.get("updated_at") or ""),
    }


def load_profile(session_id="default"):
    path = _path(session_id)
    if not path.exists():
        return {"facts": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"facts": []}

    if not isinstance(data, dict):
        return {"facts": []}

    facts = []
    for item in data.get("facts", []):
        fact = _normalize_fact(item)
        if fact:
            facts.append(fact)

    return {"facts": facts[-MAX_FACTS:]}


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

    for pattern, key, value, confidence in _STYLE_PATTERNS:
        if pattern.search(text):
            return {
                "key": key,
                "value": value,
                "confidence": confidence,
            }

    for pattern, key, confidence in _PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip(" .!?")
            if 1 <= len(value) <= MAX_FACT_CHARS:
                return {
                    "key": key,
                    "value": value,
                    "confidence": confidence,
                }

    return None


def remember_from_message(session_id, message):
    fact = extract_fact(message)
    if not fact:
        return False

    profile = load_profile(session_id)
    facts = profile["facts"]
    record = {
        **fact,
        "updated_at": _now(),
    }

    replaced = False
    for index, existing in enumerate(facts):
        if existing.get("key") == fact["key"]:
            facts[index] = record
            replaced = True
            break

    if not replaced:
        facts.append(record)

    profile["facts"] = facts[-MAX_FACTS:]
    save_profile(session_id, profile)
    return True


def build_memory_context(session_id):
    facts = load_profile(session_id).get("facts", [])
    if not facts:
        return ""

    lines = ["Long-term memory about this user:"]
    for fact in facts[-MAX_FACTS:]:
        key = fact.get("key", "fact")
        value = fact.get("value", "")
        lines.append(f"- {key}: {value}")

    lines.append("Use memory only when relevant. Do not invent additional facts.")
    return "\n".join(lines)
