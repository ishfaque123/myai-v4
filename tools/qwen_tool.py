import subprocess
import re
from pathlib import Path

MODEL_PATH = Path.home() / "MyAI" / "models" / "Qwen3-0.6B-Q8_0.gguf"

def ask_qwen(question, timeout=120):
    prompt = (
        "/no_think\n"
        "You are MyAI, a helpful multilingual assistant. "
        "Reply naturally, briefly (1-2 sentences), in the same language as the user. "
        f"User: {question}\nAI:"
    )
    try:
        result = subprocess.run(
            [
                "llama-cli",
                "-m", str(MODEL_PATH),
                "-no-cnv",
                "-c", "2048",
                "-n", "100",
                "--temp", "0.7",
                "--top-p", "0.9",
                "-p", prompt
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout
        output = output.split("AI:")[-1]
        output = re.sub(r"\[.*?\]", "", output, flags=re.DOTALL)
        output = re.split(r"(User:|>|\[end)", output)[0].strip()
        return output if output else None
    except Exception as e:
        print("Qwen error:", e)
        return None

if __name__ == "__main__":
    print(ask_qwen("Pakistan ka capital kya hai?"))
