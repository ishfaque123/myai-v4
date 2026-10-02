import json
import os
import urllib.error
import urllib.request

from engine.reliability import run_with_retries
from tools.registry import TOOL_DEFINITIONS, execute_tool_call

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODELS = (
    "openai/gpt-oss-120b:fastest",
    "openai/gpt-oss-120b:cheapest",
)

_SYSTEM_PROMPT = (
    "You are Nivora AI, a helpful and accurate multilingual AI assistant. "
    "Your name is Nivora AI. Never identify yourself as another model or provider. "
    "Reply in the same language and script as the user. "
    "Never switch to Hindi, Urdu, or another script unless the user used that script "
    "or explicitly requested that language. "
    "Answer the exact question directly and completely. "
    "Keep answers natural and reasonably concise. "
    "If unsure, say so instead of inventing facts."
)


def _post_chat_completion(model, messages, timeout=60, tools=None, tool_choice=None):
    token = os.getenv("HF_TOKEN")
    if not token:
        return None

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 768,
        "temperature": 0.4,
        "stream": False,
    }
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = tool_choice or "auto"

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
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _message_from_response(data):
    choices = (data or {}).get("choices") or []
    if not choices:
        return None
    return choices[0].get("message") or {}


def _tool_calls(message):
    calls = (message or {}).get("tool_calls") or []
    return calls if isinstance(calls, list) else []


def _execute_tool_call(tool_call):
    return execute_tool_call(tool_call)


def _usable_message(data):
    message = _message_from_response(data)
    content = (message or {}).get("content") or ""
    return bool(isinstance(content, str) and content.strip())


def _request_model(model, messages, timeout=60, tools=None, tool_choice=None):
    result = run_with_retries(
        lambda: _post_chat_completion(
            model,
            messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
        ),
        attempts=2,
        delay=0.15,
        is_success=_usable_message if not tools else lambda data: bool(_message_from_response(data)),
    )
    return result.value


def ask_cloud(prompt, timeout=60):
    user_prompt = str(prompt or "").strip()
    if not user_prompt:
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for model in HF_MODELS:
        data = _request_model(model, messages, timeout=timeout)
        message = _message_from_response(data)
        content = (message or {}).get("content") or ""
        if isinstance(content, str) and content.strip():
            return content.strip()

    return None


def ask_cloud_with_tools(prompt, timeout=60, max_rounds=3, required_tool=None):
    user_prompt = str(prompt or "").strip()
    if not user_prompt:
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for model in HF_MODELS:
        working = list(messages)

        for round_index in range(max(1, int(max_rounds))):
            tool_choice = "auto"
            if required_tool and round_index == 0:
                tool_choice = {
                    "type": "function",
                    "function": {"name": required_tool},
                }

            data = _request_model(
                model,
                working,
                timeout=timeout,
                tools=TOOL_DEFINITIONS,
                tool_choice=tool_choice,
            )
            message = _message_from_response(data)
            if not message:
                break

            calls = _tool_calls(message)
            if not calls:
                content = message.get("content") or ""
                if isinstance(content, str) and content.strip():
                    return content.strip()
                break

            assistant_message = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls,
            }
            working.append(assistant_message)

            executed = False
            for call in calls:
                tool_message = _execute_tool_call(call)
                if tool_message is not None:
                    working.append(tool_message)
                    executed = True

            if not executed:
                break

    return None
