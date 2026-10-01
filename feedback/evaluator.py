import re

def evaluate_response(user_message, answer):
    user = (user_message or "").strip()
    reply = (answer or "").strip()

    checks = {
        "non_empty": bool(reply),
        "not_too_short": len(reply) >= 2,
        "not_error_message": "could not generate a response" not in reply.lower(),
        "reasonable_length": len(reply) <= 12000,
        "question_addressed": _has_overlap(user, reply),
    }

    passed = sum(checks.values())
    score = round(passed / len(checks), 2)

    return {
        "score": score,
        "passed": score >= 0.6,
        "checks": checks,
    }


def _has_overlap(user, reply):
    words = set(re.findall(r"[a-zA-Z0-9_]+", user.lower()))
    if not words:
        return True
    reply_words = set(re.findall(r"[a-zA-Z0-9_]+", reply.lower()))
    meaningful = {word for word in words if len(word) >= 3}
    return not meaningful or bool(meaningful & reply_words)
