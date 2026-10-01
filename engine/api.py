from flask import Flask, jsonify, request
from core import chat
from feedback.store import save_feedback

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


@app.post("/feedback")
def feedback_endpoint():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body required"}), 400

    session_id = data.get("session_id") or request.headers.get("X-Session-ID") or "default"
    rating = data.get("rating")
    comment = data.get("comment", "")

    try:
        record = save_feedback(session_id, rating, comment)
    except (TypeError, ValueError):
        return jsonify({"error": "rating must be 1 or -1"}), 400

    return jsonify({"status": "saved", "feedback": record})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
