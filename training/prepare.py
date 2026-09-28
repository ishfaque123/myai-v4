import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "data" / "corpus.txt"
VOCAB = BASE / "tokenizer" / "vocab.json"
OUT = BASE / "training" / "tokens.json"

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
text = CORPUS.read_text(encoding="utf-8")
tokens = tokenize(text)
ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]

OUT.write_text(json.dumps({"ids": ids}, ensure_ascii=False), encoding="utf-8")
print("Total tokens:", len(ids))
print("Saved:", OUT)
