from flask import Flask, jsonify, request
from core import chat

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Nivora AI"})


@app.post("/chat")
def chat_endpoint():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body required"}), 400

    message = data.get("message")
    session_id = data.get("session_id") or request.headers.get("X-Session-ID") or "default"

    if not isinstance(message, str):
        return jsonify({"error": "message must be a string"}), 400

    return jsonify(chat(message, session_id))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
