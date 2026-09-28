import json
import sys
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE / "model"))
from model import TinyTransformer

VOCAB = BASE / "tokenizer" / "vocab.json"
TOKENS = BASE / "training" / "tokens.json"
CKPT_DIR = BASE / "model" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
ids = json.loads(TOKENS.read_text(encoding="utf-8"))["ids"]

vocab_size = len(vocab)
context_size = 16
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 256
lr = 0.01
epochs = 300

model = TinyTransformer(vocab_size, d_model, n_heads, n_layers, d_ff, context_size)

def loss_and_grad(logits, targets):
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    T = logits.shape[0]
    loss = -np.log(probs[np.arange(T), targets] + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(T), targets] -= 1
    dlogits /= T
    return loss, dlogits

print("MyAI v4 Training")
print("Vocab:", vocab_size, "Tokens:", len(ids))
print("Layers:", n_layers, "d_model:", d_model, "heads:", n_heads, "context:", context_size)
print("--------------------------------")

for epoch in range(epochs):
    total_loss, count = 0.0, 0
    for start in range(0, len(ids) - context_size - 1):
        chunk = ids[start:start + context_size + 1]
        if len(chunk) < context_size + 1:
            continue
        inputs, targets = chunk[:-1], np.array(chunk[1:])
        logits = model.forward(inputs)
        loss, dlogits = loss_and_grad(logits, targets)
        model.backward(dlogits)
        model.step(lr)
        total_loss += loss
        count += 1
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/max(count,1):.4f}")
    if (epoch + 1) % 100 == 0:
        model.save(CKPT_DIR / f"ckpt_{epoch+1}.npz")

model.save(BASE / "model" / "myai_v4.npz")
print("--------------------------------")
print("Training complete. Final model saved.")
