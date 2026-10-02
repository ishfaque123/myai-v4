import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    run: Callable[[Dict[str, Any]], Optional[str]]


def _valid_string(arguments, key, max_length=2000):
    if not isinstance(arguments, dict):
        return None
    value = arguments.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > max_length:
        return None
    return value


def _calculator(arguments):
    from tools.calculator import calculate
    value = _valid_string(arguments, "expression", 500)
    return calculate(value) if value else None


def _knowledge_search(arguments):
    from tools.rag import build_rag_context
    value = _valid_string(arguments, "query", 1000)
    return build_rag_context(value) if value else None


def _web_search(arguments):
    from tools.web_search import format_web_search
    value = _valid_string(arguments, "query", 1000)
    return format_web_search(value) if value else None


def _web_research(arguments):
    from tools.web_research import research_web
    value = _valid_string(arguments, "query", 1000)
    return research_web(value) if value else None


def _tool(name, description, parameter_name, parameter_description, runner):
    return ToolSpec(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                parameter_name: {
                    "type": "string",
                    "description": parameter_description,
                }
            },
            "required": [parameter_name],
            "additionalProperties": False,
        },
        run=runner,
    )


TOOL_SPECS = (
    _tool(
        "calculator",
        "Solve basic arithmetic expressions.",
        "expression",
        "Arithmetic expression to calculate.",
        _calculator,
    ),
    _tool(
        "knowledge_search",
        "Search Nivora AI's trusted local knowledge base.",
        "query",
        "Knowledge question or topic to search.",
        _knowledge_search,
    ),
    _tool(
        "web_search",
        "Search the public web for current information.",
        "query",
        "Public web search query.",
        _web_search,
    ),
    _tool(
        "web_research",
        "Search the web and read relevant result pages for a grounded summary.",
        "query",
        "Research question for multi-source web research.",
        _web_research,
    ),
)


TOOL_DEFINITIONS = tuple(
    {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }
    for spec in TOOL_SPECS
)

TOOL_MAP = {spec.name: spec for spec in TOOL_SPECS}


def execute_tool(name, arguments):
    spec = TOOL_MAP.get(name)
    if spec is None or not isinstance(arguments, dict):
        return None
    try:
        return spec.run(arguments)
    except Exception:
        return None


def execute_tool_call(tool_call):
    function = (tool_call or {}).get("function") or {}
    name = function.get("name")
    call_id = tool_call.get("id")
    if not call_id or not isinstance(name, str):
        return None

    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except (TypeError, ValueError):
        return None

    if not isinstance(arguments, dict):
        return None

    result = execute_tool(name, arguments)
    if result is None:
        return None

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": str(result),
    }
