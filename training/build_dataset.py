import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = BASE / "feedback" / "feedback.jsonl"
OUTPUT_FILE = BASE / "training" / "approved_dataset.jsonl"

def build_dataset():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_FILE.exists():
        return 0
    written = 0
    seen = set()
    with FEEDBACK_FILE.open("r", encoding="utf-8") as src, OUTPUT_FILE.open("w", encoding="utf-8") as dst:
        for line in src:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if item.get("rating") != 1:
                continue
            user = str(item.get("user_message") or "").strip()
            assistant = str(item.get("assistant_response") or "").strip()
            if not user or not assistant:
                continue
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
