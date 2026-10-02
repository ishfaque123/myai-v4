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
    """Bounded agent loop for Nivora AI.

    The loop follows observe -> tool/action -> generate -> verify, with a
    strict step limit. Tool failures never escape the agent boundary.
    """

    def __init__(self, tools=(), max_steps=3):
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

    def run(self, message, generate, verify=None):
        user = (message or "").strip()
        if not user:
            return AgentResult(None, None, 0, "empty")

        steps = 0

        if steps < self.max_steps:
            steps += 1
            result, tool_name = self._find_tool_result(user)
            if result is not None:
                if verify is None or verify(result):
                    return AgentResult(result, tool_name, steps, "tool_verified")
                if steps >= self.max_steps:
                    return AgentResult(None, tool_name, steps, "tool_rejected")

        last_answer = None
        for _ in range(self.max_steps - steps):
            steps += 1
            try:
                answer = generate(user)
            except Exception:
                answer = None

            if answer:
                last_answer = answer
                try:
                    valid = True if verify is None else bool(verify(answer))
                except Exception:
                    valid = False
                if valid:
                    status = "model_verified" if verify is not None else "model"
                    return AgentResult(answer, None, steps, status)

        return AgentResult(
            last_answer,
            None,
            steps,
            "unverified" if last_answer else "failed",
        )
