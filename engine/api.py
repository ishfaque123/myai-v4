from flask import Flask, jsonify, request
from engine.core import chat
from feedback.store import save_feedback, feedback_summary
from pathlib import Path

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
    user_message = data.get("user_message", "")
    assistant_response = data.get("assistant_response", "")

    try:
        record = save_feedback(session_id, rating, comment, user_message, assistant_response)
    except (TypeError, ValueError):
        return jsonify({"error": "rating must be 1 or -1"}), 400

    return jsonify({"status": "saved", "feedback": record})


@app.get("/dashboard")
def dashboard():
    path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    if not path.exists():
        return jsonify({"error": "dashboard not installed"}), 404
    return path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.get("/feedback/summary")
def feedback_summary_endpoint():
    return jsonify(feedback_summary())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
