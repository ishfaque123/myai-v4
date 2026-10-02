import unittest
from unittest.mock import patch

from tools.cloud_ai import TOOL_DEFINITIONS, _execute_tool_call, ask_cloud
from engine.reliability import run_with_retries


class TestCloudTools(unittest.TestCase):
    def test_expected_tools_are_registered(self):
        names = {
            item["function"]["name"]
            for item in TOOL_DEFINITIONS
        }
        self.assertIn("calculator", names)
        self.assertIn("knowledge_search", names)
        self.assertIn("web_search", names)
        self.assertIn("web_research", names)

    def test_calculator_tool_executes(self):
        result = _execute_tool_call(
            {
                "id": "call-1",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"25 * 4"}',
                },
            }
        )
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "call-1")
        self.assertEqual(result["name"], "calculator")
        self.assertIn("100", result["content"])

    def test_knowledge_tool_executes(self):
        result = _execute_tool_call(
            {
                "id": "call-2",
                "function": {
                    "name": "knowledge_search",
                    "arguments": '{"query":"What is Nivora AI?"}',
                },
            }
        )
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "call-2")
        self.assertEqual(result["name"], "knowledge_search")
        self.assertIn("Nivora AI", result["content"])

    def test_web_tool_handles_empty_query(self):
        result = _execute_tool_call(
            {
                "id": "call-3",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query":""}',
                },
            }
        )
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["name"], "web_search")
        self.assertIn("No web results", result["content"])

    def test_unknown_tool_is_rejected(self):
        result = _execute_tool_call(
            {
                "id": "call-4",
                "function": {
                    "name": "unknown_tool",
                    "arguments": "{}",
                },
            }
        )
        self.assertIsNone(result)

    def test_registry_rejects_invalid_arguments(self):
        from tools.registry import execute_tool
        self.assertIsNone(execute_tool("calculator", {"expression": 123}))
        self.assertIsNone(execute_tool("web_search", {"query": ""}))

    def test_registry_executes_valid_calculator(self):
        from tools.registry import execute_tool
        self.assertEqual(execute_tool("calculator", {"expression": "12 * 3"}), "36")

    def test_web_research_rejects_invalid_url(self):
        from tools.web_research import fetch_webpage
        self.assertEqual(fetch_webpage("not-a-url"), "")

    def test_retry_succeeds_after_transient_failure(self):
        calls = {"count": 0}

        def operation():
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError()
            return "ok"

        result = run_with_retries(operation, attempts=2, delay=0)
        self.assertEqual(result.value, "ok")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.reason, "ok")

    def test_retry_stops_after_bound(self):
        calls = {"count": 0}

        def operation():
            calls["count"] += 1
            return None

        result = run_with_retries(operation, attempts=2, delay=0)
        self.assertIsNone(result.value)
        self.assertEqual(result.attempts, 2)

    @patch("tools.cloud_ai._post_chat_completion")
    def test_empty_cloud_content_falls_through_to_next_model(self, mock_post):
        mock_post.side_effect = [
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": "Nivora response"}}]},
        ]
        answer = ask_cloud("hello", timeout=1)
        self.assertEqual(answer, "Nivora response")
        self.assertEqual(mock_post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
