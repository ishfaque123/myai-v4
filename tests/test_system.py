import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ENGINE = BASE / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from agent import AgentCore, Tool
from memory.long_term import extract_fact
from router import route_message
from engine.core import _language_style, _is_short_request
from engine.quality import clean_response, is_usable_response
from tools.rag import build_rag_context
from tools.time_tool import answer_time_query
from feedback.evaluator import evaluate_response
import feedback.store as feedback_store


class TestNivoraSystem(unittest.TestCase):
    def test_router_sends_complex_task_to_cloud(self):
        route = route_message("Explain how an API database architecture works in detail.")
        self.assertEqual(route.target, "cloud")

    def test_router_keeps_simple_question_local(self):
        route = route_message("How many days are in a week?")
        self.assertEqual(route.target, "local")

    def test_router_detects_calculation(self):
        route = route_message("Calculate 25 * 4")
        self.assertEqual(route.target, "calculator")

    def test_router_detects_time_request(self):
        route = route_message("Kon sa year chal raha hai aur date kya hai?")
        self.assertEqual(route.target, "time")

    def test_time_answer_uses_live_clock(self):
        answer = answer_time_query("Kon sa year chal raha hai aur date kya hai?", "Roman Urdu/Hinglish")
        self.assertRegex(answer, r"\d{2} [A-Za-z]+ \d{4}")
        self.assertIn("hai", answer)

    def test_router_detects_web_request(self):
        route = route_message("What is the latest news about Pakistan today?")
        self.assertEqual(route.target, "web")

    def test_router_detects_knowledge_query(self):
        route = route_message("What is Nivora AI?")
        self.assertEqual(route.target, "local")
        self.assertEqual(route.reason, "knowledge_or_rag")

    def test_router_keeps_long_complex_request_cloud(self):
        message = "Please compare " + ("different AI architectures and explain the tradeoffs " * 30)
        route = route_message(message)
        self.assertEqual(route.target, "cloud")


    def test_language_style_keeps_roman_urdu_latin(self):
        self.assertEqual(_language_style("Pakistan ka bari Mai kujh details Doo"), "Roman Urdu/Hinglish")

    def test_language_style_detects_sindhi_roman(self):
        self.assertEqual(_language_style("Sindhi Mai galh kena"), "Sindhi Roman")

    def test_short_request_is_detected(self):
        self.assertTrue(_is_short_request("Pakistan ke bare mein short mein batao"))
        self.assertTrue(_is_short_request("Koy code lekh kr doo short Mai py ka"))

    def test_quality_layer_removes_thinking_and_prefix(self):
        answer = clean_response("<think>hidden reasoning</think>Nivora AI: Final answer")
        self.assertEqual(answer, "Final answer")

    def test_quality_layer_rejects_empty_answer(self):
        self.assertFalse(is_usable_response("hello", ""))

    def test_agent_retries_after_failed_generation(self):
        calls = {"count": 0}

        def generate(_):
            calls["count"] += 1
            return "" if calls["count"] == 1 else "valid answer"

        result = AgentCore(max_steps=3).run(
            "test",
            generate,
            verify=lambda value: value == "valid answer",
        )
        self.assertEqual(result.answer, "valid answer")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.status, "model_verified")

    def test_agent_rejects_unverified_generation(self):
        result = AgentCore(max_steps=2).run(
            "test",
            lambda _: "bad answer",
            verify=lambda _: False,
        )
        self.assertIsNone(result.answer)
        self.assertEqual(result.status, "unverified")
        self.assertEqual(result.steps, 2)

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

    def test_rag_exposes_ranked_sources(self):
        from tools.rag import knowledge_sources
        sources = knowledge_sources("What is Nivora AI?")
        self.assertTrue(sources)
        self.assertIn("source", sources[0])
        self.assertIn("matched_terms", sources[0])

    def test_rag_ignores_common_words(self):
        context = build_rag_context("What is the answer?")
        self.assertEqual(context, "")

    def test_rag_requires_multiple_meaningful_matches(self):
        context = build_rag_context("Nivora")
        self.assertEqual(context, "")

        context = build_rag_context("Nivora AI")
        self.assertIn("Nivora AI", context)

    def test_memory_blocks_sensitive_fact(self):
        self.assertIsNone(extract_fact("my password is 123456"))

    def test_response_evaluation(self):
        result = evaluate_response(
            "What is Nivora AI?",
            "Nivora AI is a multilingual AI assistant.",
        )
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["score"], 0.6)

    def test_feedback_accepts_valid_rating(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_store.FEEDBACK_FILE = Path(temp_dir) / "feedback.jsonl"
            feedback_store.FEEDBACK_DIR = Path(temp_dir)

            record = feedback_store.save_feedback("test-suite", 1, "Good")
            self.assertEqual(record["rating"], 1)
            self.assertEqual(record["session_id"], "test-suite")
            self.assertEqual(feedback_store.feedback_summary()["positive"], 1)


if __name__ == "__main__":
    unittest.main()
