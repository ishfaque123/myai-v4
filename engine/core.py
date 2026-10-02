import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from engine.agent import AgentCore, Tool
from tools.qwen_tool import ask_qwen
from tools.cloud_ai import ask_cloud, ask_cloud_with_tools
from tools.calculator import calculate
from engine.router import route_message
from engine.model_manager import select_model_plan
from memory.long_term import build_memory_context, remember_from_message
from tools.rag import build_rag_context
from feedback.evaluator import evaluate_response
from engine.quality import clean_response, is_usable_response
from tools.time_tool import answer_time_query

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
    max_steps=3,
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


def _language_style(text):
    value = (text or "").strip()
    if not value:
        return "English"
    if re.search(r"[\u0900-\u097F]", value):
        return "Hindi Devanagari"
    if re.search(r"[\u0600-\u06FF]", value):
        return "Urdu/Arabic script"

    lower = value.lower()
    sindhi_markers = (
        "sindhi", "galh", "aahe", "aahiyan", "tho", "thi", "thea",
        "maan", "tawhan", "tuhnjo", "cha", "chha", "kandho", "karyo",
    )
    roman_markers = (
        "hai", "hain", "ka", "ki", "ke", "ko", "kya", "kyun",
        "mujhe", "mera", "meri", "aap", "ap", "tum", "tumhai",
        "kr", "karo", "krr", "nahi", "nhai", "abhi", "bta",
        "bata", "chahiye", "sahi", "acha", "achha", "wala", "wali",
        "boht", "bhai", "mujh", "mujhe", "baare", "bara", "bari",
        "mai", "mein", "do", "dee", "de", "sedha", "seedha",
    )
    words = re.findall(r"[a-zA-Z]+", lower)

    if any(word in sindhi_markers for word in words):
        return "Sindhi Roman"
    if any(word in roman_markers for word in words):
        return "Roman Urdu/Hinglish"
    return "English"


def _is_short_request(text):
    lower = (text or "").lower()
    return bool(
        re.search(
            r"\b(short|brief|briefly|sedha|seedha|sirf|bas|chhota|"
            r"thora|thori|2.?3 lines?|few lines?|one line)\b",
            lower,
        )
    )


def _response_matches_style(text, style):
    value = text or ""
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", value))
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", value))

    if style == "Hindi Devanagari":
        return not has_arabic
    if style == "Urdu/Arabic script":
        return not has_devanagari
    if style in ("English", "Roman Urdu/Hinglish", "Sindhi Roman"):
        return not has_devanagari and not has_arabic
    return True


def _language_repair_prompt(prompt, style):
    return (
        f"{prompt}\n\n"
        f"IMPORTANT OUTPUT CHECK: The required output style is {style}. "
        f"Return the answer again using only that language/script. "
        f"Do not use Hindi Devanagari or Urdu/Arabic script when the required "
        f"style uses Latin letters. Return only the corrected answer."
    )


def build_prompt(user_message, history, session_id):
    style = _language_style(user_message)
    short = _is_short_request(user_message)

    lines = [
        "You are Nivora AI, a helpful and accurate multilingual AI assistant.",
        "Your name is Nivora AI. If asked your name, say: My name is Nivora AI.",
        f"Output language/style: {style}. Keep exactly the same language, script, and style as the user.",
        "If the target style uses Latin letters, use Latin letters only.",
        "Do not translate Roman Urdu/Hinglish or Sindhi Roman into Hindi or Urdu script.",
        "Answer the exact question directly and completely.",
        "Do not repeat the user's question.",
        "If unsure about a fact, say so instead of inventing an answer.",
        "Keep answers short and natural unless the user asks for detail.",
    ]

    if short:
        lines.extend([
            "SHORT-ANSWER MODE IS ON.",
            "Give only the essential answer, preferably 1-4 short sentences.",
            "Do not add a long explanation, tips, or conclusion unless requested.",
        ])

    lines.extend([
        "Treat the current user message as the primary task.",
        "Founder of Nivora AI: Ishfaque Ahmed, from Thari Mirwah, Khairpur, Sindh, Pakistan.",
        "",
    ])

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
    model_plan = select_model_plan(decision.target)
    style = _language_style(user)

    def safe_answer(answer):
        cleaned = clean_response(answer)
        if is_usable_response(
            user,
            cleaned,
            style_checker=lambda value: _response_matches_style(value, style),
        ):
            return cleaned
        return None

    if decision.target == "time":
        return answer_time_query(user, style), model_plan[0].name

    if decision.target == "web":
        answer = safe_answer(
            ask_cloud_with_tools(prompt, required_tool="web_search")
        )
        if answer:
            return answer, model_plan[0].name

        repaired = safe_answer(
            ask_cloud_with_tools(
                _language_repair_prompt(prompt, style),
                required_tool="web_search",
            )
        )
        if repaired:
            return repaired, f"{model_plan[0].name}_repair"

        answer = safe_answer(ask_cloud(prompt))
        return answer, model_plan[0].name if answer else "web_failed"

    if decision.target == "cloud":
        answer = safe_answer(ask_cloud_with_tools(prompt))
        if answer:
            return answer, model_plan[0].name

        repaired = safe_answer(
            ask_cloud_with_tools(_language_repair_prompt(prompt, style))
        )
        if repaired:
            return repaired, f"{model_plan[0].name}_repair"

        answer = safe_answer(ask_qwen(prompt))
        return answer, model_plan[-1].name if answer else "cloud_failed"

    answer = safe_answer(ask_qwen(prompt))
    if answer:
        return answer, model_plan[0].name

    repaired = safe_answer(
        ask_qwen(_language_repair_prompt(prompt, style))
    )
    if repaired:
        return repaired, f"{model_plan[0].name}_repair"

    answer = safe_answer(ask_cloud(prompt))
    return answer, model_plan[-1].name if answer else "local_failed"


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
    decision = route_message(user)

    def generate(_):
        nonlocal route_name
        answer, route_name = _generate_for_route(user, prompt)
        return answer

    result = AGENT.run(user, generate, verify=lambda value: is_usable_response(
        user,
        value,
        style_checker=lambda item: _response_matches_style(item, _language_style(user)),
    ))

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
        "trace": {
            "router_target": decision.target,
            "router_reason": decision.reason,
            "router_confidence": decision.confidence,
            "model_plan": [model.name for model in select_model_plan(decision.target)],
            "memory_used": bool(build_memory_context(sid)),
            "rag_used": bool(build_rag_context(user)),
            "agent_status": result.status,
            "tool_used": result.tool,
            "model_route": route,
        },
    }
