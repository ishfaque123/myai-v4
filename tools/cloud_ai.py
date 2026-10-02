import json
import os
import urllib.error
import urllib.request

from tools.calculator import calculate
from tools.rag import build_rag_context
from tools.web_search import format_web_search

HF_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODELS = (
    "openai/gpt-oss-120b:fastest",
    "openai/gpt-oss-120b:cheapest",
)

TOOL_DEFINITIONS = (
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Solve basic arithmetic expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to calculate.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "Search Nivora AI's trusted local knowledge base for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The knowledge question or topic to search for.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current information and return concise source snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's web search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
)

_TOOL_EXECUTORS = {
    "calculator": lambda arguments: calculate(
        str(arguments.get("expression", ""))
    ),
    "knowledge_search": lambda arguments: build_rag_context(
        str(arguments.get("query", ""))
    ),
    "web_search": lambda arguments: format_web_search(
        str(arguments.get("query", ""))
    ),
}

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


def _post_chat_completion(model, messages, timeout=60, tools=None):
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
        payload["tool_choice"] = "auto"

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
    function = (tool_call or {}).get("function") or {}
    name = function.get("name")
    call_id = tool_call.get("id")
    executor = _TOOL_EXECUTORS.get(name)

    if not call_id or not executor:
        return None

    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )
    except (TypeError, ValueError):
        arguments = {}

    if not isinstance(arguments, dict):
        arguments = {}

    try:
        result = executor(arguments)
    except Exception:
        result = None

    if result is None:
        result = "Tool could not produce a result."

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": str(result),
    }


def ask_cloud(prompt, timeout=60):
    user_prompt = str(prompt or "").strip()
    if not user_prompt:
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for model in HF_MODELS:
        data = _post_chat_completion(model, messages, timeout=timeout)
        message = _message_from_response(data)
        content = (message or {}).get("content") or ""
        if isinstance(content, str) and content.strip():
            return content.strip()

    return None


def ask_cloud_with_tools(prompt, timeout=60, max_rounds=3):
    user_prompt = str(prompt or "").strip()
    if not user_prompt:
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for model in HF_MODELS:
        working = list(messages)

        for _ in range(max(1, int(max_rounds))):
            data = _post_chat_completion(
                model,
                working,
                timeout=timeout,
                tools=TOOL_DEFINITIONS,
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
