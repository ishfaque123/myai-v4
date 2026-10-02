from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable[[str], Optional[str]]


@dataclass(frozen=True)
class AgentResult:
    answer: Optional[str]
    tool: Optional[str]
    steps: int
    status: str


class AgentCore:
    """Small, bounded agent loop for Nivora AI.

    Deterministic local tools run before model generation. Cloud model
    function-calling is handled by the cloud tool loop.
    """

    def __init__(self, tools=(), max_steps=2):
        self.tools = tuple(tools)
        self.max_steps = max(1, int(max_steps))

    def _find_tool_result(self, message):
        for tool in self.tools:
            try:
                result = tool.run(message)
            except Exception:
                result = None
            if result is not None:
                return result, tool.name
        return None, None

    def run(self, message, generate):
        user = (message or "").strip()
        if not user:
            return AgentResult(None, None, 0, "empty")

        if self.max_steps >= 1:
            result, tool_name = self._find_tool_result(user)
            if result is not None:
                return AgentResult(result, tool_name, 1, "local_tool")

        if self.max_steps >= 2:
            try:
                answer = generate(user)
            except Exception:
                answer = None
            if answer:
                return AgentResult(answer, None, 2, "model")

        return AgentResult(None, None, self.max_steps, "failed")
