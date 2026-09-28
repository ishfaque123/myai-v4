import json
import re
import sys
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE / "model"))
sys.path.append(str(BASE / "tools"))
from model import TinyTransformer
from calculator import calculate

VOCAB = BASE / "tokenizer" / "vocab.json"
MODEL_FILE = BASE / "model" / "myai_v4.npz"
MEMORY_FILE = BASE / "memory" / "memory.json"

vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
id_to_token = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

model = TinyTransformer(vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256, context_size=16)
model.load(MODEL_FILE)

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

def generate(text, max_tokens=25, temperature=0.8):
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokenize(text)]
    if not ids:
        return "Mujhe kuch samajh nahi aaya."
    result = []
    for _ in range(max_tokens):
        context = ids[-model.context_size:]
        logits = model.forward(context)[-1]
        logits = logits / temperature
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        next_id = np.random.choice(len(probs), p=probs)
        token = id_to_token.get(next_id, "<UNK>")
        if token in ["<PAD>", "<BOS>", "<UNK>"]:
            ids.append(next_id)
            continue
        if token == "<EOS>":
            break
        result.append(token)
        ids.append(next_id)
    return re.sub(r"\s+([?.!,])", r"\1", " ".join(result)).strip()

def load_memory():
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_memory(history):
    MEMORY_FILE.write_text(json.dumps(history[-20:], ensure_ascii=False, indent=2), encoding="utf-8")

history = load_memory()
print("MyAI v4 — Real Transformer Engine")
print("Type 'exit' to quit, 'clear' to clear memory.")

while True:
    try:
        user = input("\nYou: ").strip()
    except KeyboardInterrupt:
        print("\nAI: Goodbye!")
        break
    if not user:
        continue
    if user.lower() == "exit":
        print("AI: Goodbye!")
        break
    if user.lower() == "clear":
        history = []
        save_memory(history)
        print("AI: Memory clear kar di.")
        continue

    calc = calculate(user)
    if calc is not None:
        answer = calc
    else:
        context_parts = []
        for item in history[-3:]:
            context_parts.append(item["user"])
            context_parts.append(item["ai"])
        context_parts.append(user)
        answer = generate(" ".join(context_parts))
        if not answer:
            answer = "Mujhe jawab generate nahi ho raha."

    print("AI:", answer)
    history.append({"user": user, "ai": answer})
    save_memory(history)
