import unittest

from tools.cloud_ai import TOOL_DEFINITIONS, _execute_tool_call


class TestCloudTools(unittest.TestCase):
    def test_expected_tools_are_registered(self):
        names = {
            item["function"]["name"]
            for item in TOOL_DEFINITIONS
        }
        self.assertIn("calculator", names)
        self.assertIn("knowledge_search", names)
        self.assertIn("web_search", names)

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


if __name__ == "__main__":
    unittest.main()
