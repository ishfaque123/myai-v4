import json
import os
import urllib.error
import urllib.request

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODELS = (
    "openai/gpt-oss-120b:fastest",
    "openai/gpt-oss-120b:cheapest",
)


def ask_cloud(prompt, timeout=60):
    token = os.getenv("HF_TOKEN")
    if not token:
        return None

    user_prompt = str(prompt or "").strip()
    if not user_prompt:
        return None

    for model in HF_MODELS:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Nivora AI, a helpful and accurate multilingual AI assistant. "
                        "Your name is Nivora AI. Never identify yourself as another model or provider. "
                        "Reply in the same language and script as the user. "
                        "Never switch to Hindi, Urdu, or another script unless the user used that script "
                        "or explicitly requested that language. "
                        "Answer the exact question directly and completely. "
                        "Keep answers natural and reasonably concise. "
                        "If unsure, say so instead of inventing facts."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 768,
            "temperature": 0.4,
            "stream": False,
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

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
            continue

        choices = data.get("choices") or []
        if not choices:
            continue

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, str) and content.strip():
            return content.strip()

    return None
