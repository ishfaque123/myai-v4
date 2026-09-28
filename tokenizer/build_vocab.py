import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "data" / "corpus.txt"
VOCAB_OUT = BASE / "tokenizer" / "vocab.json"

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

text = CORPUS.read_text(encoding="utf-8")
tokens = tokenize(text)

unique = sorted(set(tokens))
special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
vocab = {tok: i for i, tok in enumerate(special)}
for tok in unique:
    if tok not in vocab:
        vocab[tok] = len(vocab)

VOCAB_OUT.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
print("Vocab size:", len(vocab))
print("Saved:", VOCAB_OUT)
