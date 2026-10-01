import os
import pty
import re
import shutil
import subprocess
import time
from pathlib import Path


MODEL_PATH = Path(
    os.getenv(
        "NIVORA_MODEL_PATH",
        str(Path.home() / "MyAI" / "models" / "Qwen3-0.6B-Q8_0.gguf"),
    )
)
LLAMA_CLI = os.getenv("LLAMA_CLI", "llama-cli")


def _run_llama_cli(args, timeout):
    """Run llama-cli through a PTY so Termux's terminal-only output is captured."""
    master_fd = None
    slave_fd = None
    process = None
    chunks = []

    try:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            args,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = None

        deadline = time.monotonic() + timeout

        while True:
            if time.monotonic() >= deadline:
                process.kill()
                raise subprocess.TimeoutExpired(args, timeout)

            import select

            readable, _, _ = select.select([master_fd], [], [], 0.25)
            if readable:
                try:
                    data = os.read(master_fd, 8192)
                except OSError:
                    data = b""
                if data:
                    chunks.append(data.decode("utf-8", errors="replace"))

            if process.poll() is not None:
                # Drain any final bytes emitted just before process exit.
                while True:
                    import select

                    readable, _, _ = select.select([master_fd], [], [], 0)
                    if not readable:
                        break
                    try:
                        data = os.read(master_fd, 8192)
                    except OSError:
                        break
                    if not data:
                        break
                    chunks.append(data.decode("utf-8", errors="replace"))
                break

        process.wait(timeout=2)
        return "".join(chunks)

    except (OSError, subprocess.SubprocessError):
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.SubprocessError:
                pass
        return None
    finally:
        if slave_fd is not None:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass


def _extract_answer(stream):
    text = (stream or "").strip()
    if not text:
        return ""

    # PTY output can contain terminal control sequences and carriage returns.
    text = text.replace("\r", "")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(
        r"(?m)^\s*(Loading model\.\.\.|build\s*:.*|model\s*:.*|"
        r"ftype\s*:.*|modalities\s*:.*)\s*$",
        "",
        text,
    )
    text = re.sub(r"(?m)^\s*available commands:\s*$", "", text)
    text = re.sub(r"(?m)^\s*[/]?(exit|regen|clear|read|glob)\b.*$", "", text)
    text = re.sub(r"(?m)^\s*\[\s*Prompt:.*$", "", text)
    text = re.sub(r"(?m)^\s*Exiting\.\.\.\s*$", "", text)

    # Current llama-cli prints the generated response after this marker.
    marker_matches = list(re.finditer(r"(?im)^\s*Nivora AI\s*:\s*", text))
    if marker_matches:
        text = text[marker_matches[-1].end():]
    else:
        prompt_matches = list(re.finditer(r"(?m)^\s*>\s*.*$", text))
        if prompt_matches:
            text = text[prompt_matches[-1].end():]

    # Remove an accidentally echoed system prompt.
    if "You are Nivora AI" in text:
        marker = text.rfind("Nivora AI:")
        if marker >= 0:
            text = text[marker + len("Nivora AI:"):]
        else:
            text = text.split("You are Nivora AI", 1)[-1]

    text = re.sub(r"(?m)^\s*\[\s*Prompt:.*$", "", text)
    text = re.sub(r"(?m)^\s*Exiting\.\.\.\s*$", "", text)
    return text.strip()


def ask_qwen(question, timeout=120):
    if not MODEL_PATH.is_file() or not shutil.which(LLAMA_CLI):
        return None

    prompt = str(question).strip()
    if not prompt:
        return None

    args = [
        LLAMA_CLI,
        "-m", str(MODEL_PATH),
        "--single-turn",
        "--reasoning", "off",
        "-c", "2048",
        "-n", "256",
        "--temp", "0.4",
        "--top-p", "0.8",
        "--min-p", "0.05",
        "--no-show-timings",
        "--color", "off",
        "-p", prompt + "\nNivora AI:",
    ]

    raw_output = _run_llama_cli(args, timeout)
    answer = _extract_answer(raw_output)
    return answer or None


if __name__ == "__main__":
    answer = ask_qwen("Pakistan ka capital kya hai?")
    print(answer or "Qwen model/llama-cli not available.")
