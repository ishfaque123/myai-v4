import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

MODEL_PATH = Path(
    os.getenv(
        "NIVORA_MODEL_PATH",
        str(Path.home() / "MyAI" / "models" / "Qwen3-0.6B-Q8_0.gguf"),
    )
)
LLAMA_CLI = os.getenv("LLAMA_CLI", "llama-cli")
DEFAULT_SYSTEM = (
    "You are Nivora AI, a helpful and accurate multilingual AI assistant.\n"
    "Your name is Nivora AI. If asked your name, say: My name is Nivora AI.\n"
    "Reply in the same language and style as the user.\n"
    "Answer the exact question directly and completely.\n"
    "For factual questions, give the complete factual answer, not a partial sentence.\n"
    "Keep answers short and natural unless the user asks for detail.\n"
    "Do not repeat the user's question or leave sentences incomplete.\n"
    "If unsure about a fact, say so instead of inventing an answer.\n"
    "Founder of Nivora AI: Ishfaque Ahmed, from Thari Mirwah, Khairpur, Sindh, Pakistan."
)

logger = logging.getLogger(__name__)


def extract_answer(stdout, user_message):
    text = (stdout or "").replace("\r", "")
    if not text.strip():
        return ""

    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.split(r"(?im)^\s*Exiting\.\.\.\s*$", text, maxsplit=1)[0]

    lines = text.splitlines()
    user_message = str(user_message).strip()
    exact_prompt = re.compile(r"^\s*>\s*" + re.escape(user_message) + r"\s*$") if user_message else None
    prompt_index = None
    for index, line in enumerate(lines):
        if exact_prompt and exact_prompt.match(line):
            prompt_index = index
            break
    if prompt_index is None:
        for index, line in enumerate(lines):
            if re.match(r"^\s*>\s+", line):
                prompt_index = index
                break
    if prompt_index is not None:
        lines = lines[prompt_index + 1:]

    while lines and not lines[-1].strip():
        lines.pop()
    while lines and re.match(r"^\s*>\s*$", lines[-1]):
        lines.pop()

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        if re.fullmatch(r"[\s\u2580-\u259F]+", stripped):
            continue
        if re.match(r"^(?:Loading model\.\.\.|build\s*:|model\s*:|ftype\s*:|modalities\s*:)", stripped, re.I):
            continue
        if re.match(r"^\[\s*Prompt\s*:", stripped, re.I):
            continue
        if re.match(r"^available commands:\s*$", stripped, re.I):
            continue
        if re.match(r"^/(?:exit|regen|clear|read|glob)\b", stripped, re.I):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned).strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"^\s*Nivora AI\s*:\s*", "", text, count=1, flags=re.I)
    return text.strip()


def ask_qwen(question, timeout=120, system=None):
    if not MODEL_PATH.is_file() or not shutil.which(LLAMA_CLI):
        return None

    user_message = str(question).strip()
    if not user_message:
        return None
    system_prompt = str(system).strip() if system else DEFAULT_SYSTEM

    try:
        result = subprocess.run(
            [
                LLAMA_CLI,
                "-m", str(MODEL_PATH),
                "--single-turn",
                "--reasoning", "off",
                "-c", "2048",
                "-n", "256",
                "--temp", "0.4",
                "--top-p", "0.8",
                "--min-p", "0.05",
                "--no-display-prompt",
                "--no-show-timings",
                "--color", "off",
                "-sys", system_prompt,
                "-p", user_message,
            ],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Qwen CLI execution failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.error("Qwen CLI failed: %s", (result.stderr or "")[-500:].strip())
        return None

    answer = extract_answer(result.stdout, user_message)
    return answer or None


if __name__ == "__main__":
    answer = ask_qwen("Pakistan ka capital kya hai?")
    print(answer or "Qwen model/llama-cli not available.")
