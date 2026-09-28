import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from tools.qwen_tool import ask_qwen
from tools.calculator import calculate
MEMORY_DIR = BASE / "memory" / "sessions"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MAX_HISTORY = 12
MAX_MESSAGE_CHARS = 4000


def _safe_session_id(session_id):
    value = re.sub(r"[^a-zA-Z0-9_-]", "", str(session_id or "default"))[:64]
    return value or "default"


def _memory_path(session_id):
    return MEMORY_DIR / f"{_safe_session_id(session_id)}.json"


def load_memory(session_id="default"):
    path = _memory_path(session_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError, TypeError):
        return []


def save_memory(session_id, history):
    path = _memory_path(session_id)
    path.write_text(
        json.dumps(history[-MAX_HISTORY:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_prompt(user_message, history):
    lines = [
        "You are Nivora AI, a helpful, accurate and friendly multilingual AI assistant.",
        "Reply naturally in the same language as the user.",
        "Keep answers concise unless the user asks for detail.",
        "Do not invent facts. If you are unsure, say so.",
        "",
    ]
    for item in history[-6:]:
        lines.append(f"User: {item.get('user', '')}")
        lines.append(f"Nivora AI: {item.get('ai', '')}")
    lines.append(f"User: {user_message}")
    lines.append("Nivora AI:")
    return "\n".join(lines)


def chat(message, session_id="default"):
    user = (message or "").strip()
    sid = _safe_session_id(session_id)
    if not user:
        return {"reply": "Please enter a message.", "session_id": sid}

    if len(user) > MAX_MESSAGE_CHARS:
        return {
            "reply": f"Message bohat lamba hai. Maximum {MAX_MESSAGE_CHARS} characters allowed hain.",
            "session_id": sid,
        }

    history = load_memory(sid)

    calc = calculate(user)
    if calc is not None:
        answer = calc
    else:
        answer = ask_qwen(build_prompt(user, history))

    if not answer:
        answer = "Sorry, abhi response generate nahi ho saka. Please dobara try karein."

    history.append({"user": user, "ai": answer})
    save_memory(sid, history)
    return {"reply": answer, "session_id": sid}
