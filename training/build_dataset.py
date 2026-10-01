import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = BASE / "feedback" / "feedback.jsonl"
OUTPUT_FILE = BASE / "training" / "approved_dataset.jsonl"

MAX_USER_CHARS = 4000
MAX_ASSISTANT_CHARS = 12000
ERROR_MARKERS = (
    "could not generate a response",
    "something went wrong",
    "please try again",
)

def _clean(value, limit):
    return " ".join(str(value or "").strip().split())[:limit].strip()

def _valid_example(item):
    if item.get("rating") != 1:
        return None
    user = _clean(item.get("user_message"), MAX_USER_CHARS)
    assistant = _clean(item.get("assistant_response"), MAX_ASSISTANT_CHARS)
    if not user or not assistant:
        return None
    if any(marker in assistant.lower() for marker in ERROR_MARKERS):
        return None
    if len(assistant) < 2:
        return None
    return user, assistant

def build_dataset():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_FILE.exists():
        OUTPUT_FILE.write_text("", encoding="utf-8")
        return 0

    written = 0
    seen = set()
    with FEEDBACK_FILE.open("r", encoding="utf-8") as src, OUTPUT_FILE.open("w", encoding="utf-8") as dst:
        for line in src:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(item, dict):
                continue
            example = _valid_example(item)
            if example is None:
                continue
            user, assistant = example
            key = (user, assistant)
            if key in seen:
                continue
            seen.add(key)
            dst.write(json.dumps({
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }, ensure_ascii=False) + "\n")
            written += 1
    return written

if __name__ == "__main__":
    print(f"Approved examples: {build_dataset()}")
