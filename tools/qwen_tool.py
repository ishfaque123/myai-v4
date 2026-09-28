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
                "-st",
                "--reasoning", "off",
                "-c", "2048",
                "-n", "256",
                "--temp", "0.7",
                "--top-p", "0.9",
                "--no-display-prompt",
                "-p", prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = result.stdout.strip()
    if not output:
        return None

    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)

    # llama-cli can return the generated answer together with its terminal UI.
    # Keep only the text after the final prompt marker when present.
    if "\n> " in output:
        output = output.rsplit("\n> ", 1)[-1].strip()

    # Remove llama.cpp metadata/UI lines if they are still present.
    output = re.sub(r"(?m)^\s*(Loading model\.\.\.|build\s*:.*|model\s*:.*|ftype\s*:.*|modalities\s*:.*)\s*$", "", output)
    output = re.sub(r"(?m)^\s*available commands:.*$", "", output)

    # Keep the generated response, not echoed conversation/history.
    output = re.sub(r"^\s*(Nivora AI|AI)\s*:\s*", "", output, flags=re.IGNORECASE)
    output = re.split(r"\n\s*(User|Nivora AI|AI)\s*:", output, maxsplit=1)[0]

    return output.strip() or None


if __name__ == "__main__":
    answer = ask_qwen("Pakistan ka capital kya hai?")
    print(answer or "Qwen model/llama-cli not available.")
