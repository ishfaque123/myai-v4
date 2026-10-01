import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FEEDBACK_DIR = BASE / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

MAX_COMMENT_CHARS = 1000


def save_feedback(session_id, rating, comment="", user_message="", assistant_response=""):
    rating = int(rating)
    if rating not in (1, -1):
        raise ValueError("rating must be 1 or -1")

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": str(session_id or "default")[:64],
        "rating": rating,
        "comment": str(comment or "")[:MAX_COMMENT_CHARS],
        "user_message": str(user_message or "")[:4000],
        "assistant_response": str(assistant_response or "")[:12000],
    }

    with FEEDBACK_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def load_feedback(limit=100):
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 100

    if not FEEDBACK_FILE.exists():
        return []

    records = []
    try:
        with FEEDBACK_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(item, dict):
                    records.append(item)
    except OSError:
        return []

    return records[-limit:]


def feedback_summary():
    records = load_feedback()
    positive = sum(1 for item in records if item.get("rating") == 1)
    negative = sum(1 for item in records if item.get("rating") == -1)
    total = positive + negative

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": round(positive / total, 4) if total else 0.0,
    }
