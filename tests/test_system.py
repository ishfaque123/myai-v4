import unittest

from agent import AgentCore, Tool
from memory.long_term import extract_fact
from router import route_message
from tools.rag import build_rag_context
from feedback.store import save_feedback


class TestNivoraSystem(unittest.TestCase):
    def test_router_sends_complex_task_to_cloud(self):
        route = route_message("Explain how an API database architecture works in detail.")
        self.assertEqual(route.target, "cloud")

    def test_agent_uses_tool_before_model(self):
        calls = []

        def tool_run(message):
            return "42"

        def generate(message):
            calls.append(message)
            return "model"

        agent = AgentCore((Tool("test", "test tool", tool_run),), max_steps=2)
        result = agent.run("calculate", generate)

        self.assertEqual(result.answer, "42")
        self.assertEqual(result.tool, "test")
        self.assertEqual(calls, [])

    def test_agent_is_bounded(self):
        calls = []

        def generate(message):
            calls.append(message)
            return None

        agent = AgentCore((), max_steps=2)
        result = agent.run("hello", generate)

        self.assertEqual(result.steps, 2)
        self.assertLessEqual(len(calls), 1)

    def test_rag_returns_knowledge(self):
        context = build_rag_context("What is Nivora AI?")
        self.assertIn("Nivora AI", context)

    def test_memory_blocks_sensitive_fact(self):
        self.assertIsNone(extract_fact("my password is 123456"))

    def test_feedback_accepts_valid_rating(self):
        record = save_feedback("test-suite", 1, "Good")
        self.assertEqual(record["rating"], 1)
        self.assertEqual(record["session_id"], "test-suite")


if __name__ == "__main__":
    unittest.main()
