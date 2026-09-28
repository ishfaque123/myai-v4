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

    # Termux llama.cpp can split prompt/UI and generated text across stdout/stderr.
    # Combine both streams before parsing so a non-empty prompt stream cannot hide
    # the actual generated answer in the other stream.
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if not output:
        return None

    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)

    # Remove terminal metadata that can surround the generated answer.
    output = re.sub(r"(?m)^\s*(Loading model\.\.\.|build\s*:.*|model\s*:.*|ftype\s*:.*|modalities\s*:.*)\s*$", "", output)
    output = re.sub(r"(?m)^\s*available commands:\s*$", "", output)
    output = re.sub(r"(?m)^\s*[/]?(exit|regen|clear|read|glob)\b.*$", "", output)
    output = re.sub(r"(?m)^\s*\[\s*Prompt:.*$", "", output)
    output = re.sub(r"(?m)^\s*Exiting\.\.\.\s*$", "", output)

    # llama-cli may echo the complete prompt/history before the answer.
    # The final Nivora AI marker is the assistant turn we need.
    markers = list(re.finditer(r"(?im)^\s*Nivora AI\s*:\s*", output))
    if markers:
        output = output[markers[-1].end():]

    # Fallback for output that starts directly with the generated answer.
    output = output.strip()
    if output.startswith("> "):
        output = output[2:].lstrip()

    output = re.sub(r"(?m)^\s*\[\s*Prompt:.*$", "", output)
    output = re.sub(r"(?m)^\s*Exiting\.\.\.\s*$", "", output)

    return output.strip() or None


if __name__ == "__main__":
    answer = ask_qwen("Pakistan ka capital kya hai?")
    print(answer or "Qwen model/llama-cli not available.")
