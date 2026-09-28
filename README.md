# Nivora AI

Nivora AI is a lightweight multilingual AI assistant built around a local GGUF model through llama.cpp.

## Architecture

- Flask API
- Qwen GGUF model through `llama-cli`
- Per-session JSON memory
- Calculator tool
- CLI chat
- Environment-configurable model and llama.cpp paths

## Local setup

```bash
pip install -r requirements.txt
export NIVORA_MODEL_PATH="$HOME/MyAI/models/Qwen3-0.6B-Q8_0.gguf"
python engine/api.py
```

API endpoints:

- `GET /health`
- `POST /chat`

Example request:

```json
{"message":"Hello","session_id":"demo"}
```

For production hosting, move memory to a persistent database and provide the GGUF model through server storage or a dedicated model service. Never commit credentials or secrets.
