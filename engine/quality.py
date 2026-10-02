import re


_MAX_RESPONSE_CHARS = 12000


def clean_response(text):
    value = str(text or "").strip()
    if not value:
        return ""

    value = re.sub(r"(?is)<think>.*?</think>", "", value)
    value = re.sub(r"(?is)<think>.*$", "", value)
    value = re.sub(r"(?im)^\s*(loading model|llama\.cpp|build:.*|model:.*|system info:.*)\s*$", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()

    if value.lower().startswith("nivora ai:"):
        value = value[len("nivora ai:"):].lstrip()

    return value[:_MAX_RESPONSE_CHARS].strip()


def is_usable_response(user_message, answer, style_checker=None):
    cleaned = clean_response(answer)
    if not cleaned or len(cleaned) < 2:
        return False
    if "could not generate a response right now" in cleaned.lower():
        return False
    if style_checker is not None and not style_checker(cleaned):
        return False
    return True
