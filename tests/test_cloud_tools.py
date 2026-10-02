import unittest

from tools.cloud_ai import TOOL_DEFINITIONS, _execute_tool_call


class TestCloudTools(unittest.TestCase):
    def test_calculator_tool_is_registered(self):
        names = {
            item["function"]["name"]
            for item in TOOL_DEFINITIONS
        }
        self.assertIn("calculator", names)

    def test_knowledge_tool_is_registered(self):
        names = {
            item["function"]["name"]
            for item in TOOL_DEFINITIONS
        }
        self.assertIn("knowledge_search", names)

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

    def test_unknown_tool_is_rejected(self):
        result = _execute_tool_call(
            {
                "id": "call-3",
                "function": {
                    "name": "unknown_tool",
                    "arguments": "{}",
                },
            }
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
