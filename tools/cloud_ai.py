import json
import os
import urllib.error
import urllib.request

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "openai/gpt-oss-120b:cheapest"


def ask_cloud(prompt, timeout=60):
    token = os.getenv("HF_TOKEN")
    if not token:
        return None

    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Nivora AI, a helpful and accurate multilingual AI assistant. "
                    "Your name is Nivora AI. Never identify yourself as another model or provider. "
                    "Reply in the same language and style as the user. "
                    "Answer the exact question directly and completely. "
                    "Keep answers natural and reasonably concise. "
                    "If unsure, say so instead of inventing facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.4,
    }

    request = urllib.request.Request(
        HF_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
            if attempt == 1:
                return None

    choices = data.get("choices") or []
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    return content.strip() or None
