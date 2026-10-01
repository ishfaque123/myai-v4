import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from agent import AgentCore, Tool
from tools.qwen_tool import ask_qwen
from tools.cloud_ai import ask_cloud
from tools.calculator import calculate
from router import route_message
from memory.long_term import build_memory_context, remember_from_message
from tools.rag import build_rag_context
from feedback.evaluator import evaluate_response

MEMORY_DIR = BASE / "memory" / "sessions"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MAX_HISTORY = 12
MAX_MESSAGE_CHARS = 4000

AGENT = AgentCore(
    tools=(
        Tool(
            name="calculator",
            description="Solve basic arithmetic expressions.",
            run=calculate,
        ),
    ),
    max_steps=2,
)


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


def build_prompt(user_message, history, session_id):
    lines = [
        "You are Nivora AI, a helpful and accurate multilingual AI assistant.",
        "Your name is Nivora AI. If asked your name, say: My name is Nivora AI.",
        "Reply in the same language and style as the user.",
        "Answer the exact question directly and completely.",
        "For factual questions, give the complete factual answer, not a partial sentence.",
        "Keep answers short and natural unless the user asks for detail.",
        "Do not repeat the user's question or leave sentences incomplete.",
        "If unsure about a fact, say so instead of inventing an answer.",
        "Founder of Nivora AI: Ishfaque Ahmed, from Thari Mirwah, Khairpur, Sindh, Pakistan.",
        "",
    ]

    memory_context = build_memory_context(session_id)
    if memory_context:
        lines.extend([memory_context, ""])

    rag_context = build_rag_context(user_message)
    if rag_context:
        lines.extend([
            rag_context,
            "Use the relevant knowledge context when it helps answer the user's question.",
            "Do not invent facts that are not supported by the knowledge context.",
            "",
        ])

    for item in history[-2:]:
        lines.append(f"User: {item.get('user', '')}")
        lines.append(f"Nivora AI: {item.get('ai', '')}")
    lines.append(f"User: {user_message}")
    lines.append("Nivora AI:")
    return "\n".join(lines)


def _generate_for_route(user, prompt):
    decision = route_message(user)

    if decision.target == "cloud":
        answer = ask_cloud(prompt)
        if answer:
            return answer, "cloud"
        answer = ask_qwen(prompt)
        return answer, "local_fallback" if answer else "cloud_failed"

    answer = ask_qwen(prompt)
    if answer:
        return answer, "local"
    answer = ask_cloud(prompt)
    return answer, "cloud_fallback" if answer else "local_failed"


def chat(message, session_id="default"):
    user = (message or "").strip()
    sid = _safe_session_id(session_id)

    if not user:
        return {"reply": "Please enter a message.", "session_id": sid}

    if len(user) > MAX_MESSAGE_CHARS:
        return {
            "reply": f"Message is too long. Maximum {MAX_MESSAGE_CHARS} characters are allowed.",
            "session_id": sid,
        }

    history = load_memory(sid)
    prompt = build_prompt(user, history, sid)

    route_name = None

    def generate(_):
        nonlocal route_name
        answer, route_name = _generate_for_route(user, prompt)
        return answer

    result = AGENT.run(user, generate)

    if result.answer:
        answer = result.answer
        if result.tool:
            route_name = result.tool
        route = route_name or result.status
    else:
        answer = "Sorry, I could not generate a response right now. Please try again."
        route = route_name or "failed"

    evaluation = evaluate_response(user, answer)

    history.append({"user": user, "ai": answer})
    save_memory(sid, history)
    remember_from_message(sid, user)

    return {
        "reply": answer,
        "session_id": sid,
        "route": route,
        "agent_steps": result.steps,
        "evaluation": evaluation,
    }
