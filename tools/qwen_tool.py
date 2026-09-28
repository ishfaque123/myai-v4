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


def ask_qwen(question, timeout=120):
    if not MODEL_PATH.is_file() or not shutil.which(LLAMA_CLI):
        return None

    prompt = str(question).strip()
    if not prompt:
        return None

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
                "--simple-io",
                "--no-show-timings",
                "--color", "off",
                "-p", prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    def extract_answer(stream):
        text = (stream or "").strip()
        if not text:
            return ""

        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        text = re.sub(r"(?m)^\s*(Loading model\.\.\.|build\s*:.*|model\s*:.*|ftype\s*:.*|modalities\s*:.*)\s*$", "", text)
        text = re.sub(r"(?m)^\s*available commands:\s*$", "", text)
        text = re.sub(r"(?m)^\s*[/]?(exit|regen|clear|read|glob)\b.*$", "", text)
        text = re.sub(r"(?m)^\s*\[\s*Prompt:.*$", "", text)
        text = re.sub(r"(?m)^\s*Exiting\.\.\.\s*$", "", text)

        # If the model output contains the full Nivora prompt, keep only the
        # generated text after the final prompt marker.
        marker_matches = list(re.finditer(r"(?im)^\s*Nivora AI\s*:\s*", text))
        if marker_matches:
            text = text[marker_matches[-1].end():]

        # The CLI prompt marker may appear before the model output. If the
        # stream contains our full Nivora prompt, use the final generated marker.
        if "You are Nivora AI" in text:
            marker_matches = list(re.finditer(r"(?im)^\s*Nivora AI\s*:\s*", text))
            if marker_matches:
                text = text[marker_matches[-1].end():]

        # llama-cli may echo the full prompt before the generated answer.
        # When that happens, discard everything through the final prompt marker.
        if "You are Nivora AI" in text:
            marker = text.rfind("Nivora AI:")
            if marker >= 0:
                text = text[marker + len("Nivora AI:"):]

        # Never expose an echoed system prompt in the user-facing response.
        if "You are Nivora AI" in text:
            text = text.split("You are Nivora AI", 1)[0].strip()

        text = text.strip()
        if text.startswith("> "):
            text = text[2:].lstrip()

        text = re.sub(r"(?m)^\s*\[\s*Prompt:.*$", "", text)
        text = re.sub(r"(?m)^\s*Exiting\.\.\.\s*$", "", text)
        return text.strip()

    # Parse stdout and stderr independently. They can contain different parts
    # of llama-cli; combining them first can place a prompt marker after the
    # generated answer and make the parser discard the real response.
    candidates = [
        extract_answer(result.stdout),
        extract_answer(result.stderr),
    ]
    candidates = [item for item in candidates if item]
    if not candidates:
        return None

    # Prefer the candidate that does not look like an echoed user prompt.
    for candidate in candidates:
        if not re.search(r"(?im)^\s*User\s*:", candidate):
            return candidate

    return candidates[-1]


if __name__ == "__main__":
    answer = ask_qwen("Pakistan ka capital kya hai?")
    print(answer or "Qwen model/llama-cli not available.")
