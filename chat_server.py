"""
chat_server.py  —  Flask backend for the WhatsApp-style Malayalam chat app.

Endpoints:
  POST /api/send       — classify a message and store it
  GET  /api/messages   — retrieve full message history
  POST /api/feedback   — mark a message as hate speech → appends to training CSV
"""

import csv
import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow the HTML file to call the API from any origin

DATASET_PATH = os.path.join(os.path.dirname(__file__), "mal_full_offensive_train.csv")
FEEDBACK_LOG  = os.path.join(os.path.dirname(__file__), "feedback_log.csv")
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "finetuned_model")

# In-memory message store  {id: message_dict}
messages: dict = {}
message_order: list = []   # keeps insertion order

# ─────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────
print("Loading fine-tuned model...")
classifier = pipeline("text-classification", model=MODEL_DIR, top_k=None)
print("Model loaded successfully!")

SAFE_LABELS = {"Not_offensive", "Not_in_intended_language"}

LABEL_META = {
    "Not_offensive":           {"display": "Safe",                  "severity": "safe"},
    "Not_in_intended_language":{"display": "Not Malayalam",         "severity": "safe"},
    "Off_target_group":        {"display": "Group Hate Speech",     "severity": "high"},
    "Profanity":               {"display": "Profanity / Vulgarity", "severity": "high"},
    "Off_target_ind":          {"display": "Personal Attack",       "severity": "high"},
}

def classify(text: str) -> dict:
    """Run the model and return a structured result dict."""
    raw = classifier(text)
    results = raw[0] if (raw and isinstance(raw[0], list)) else raw

    safe_score      = sum(r["score"] for r in results if r["label"] in SAFE_LABELS)
    offensive_score = 1.0 - safe_score
    is_offensive    = offensive_score > 0.5

    top = max(results, key=lambda r: r["score"])
    meta = LABEL_META.get(top["label"], {"display": top["label"], "severity": "unknown"})

    # Determine UI severity bucket
    if not is_offensive:
        bucket = "safe"
    elif offensive_score < 0.80:
        bucket = "mild"
    elif offensive_score < 0.95:
        bucket = "high"
    else:
        bucket = "extreme"

    return {
        "label":           top["label"],
        "label_display":   meta["display"],
        "offensive_score": round(offensive_score, 4),
        "safe_score":      round(safe_score, 4),
        "is_offensive":    is_offensive,
        "bucket":          bucket,
        "all_scores":      {r["label"]: round(r["score"], 4) for r in results},
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/api/send", methods=["POST"])
def send_message():
    """
    Body JSON:
      { "text": "...", "sender": "me" | "other", "username": "Alice" }
    Returns the full message object including classification.
    """
    data     = request.get_json(force=True)
    text     = data.get("text", "").strip()
    sender   = data.get("sender", "me")       # "me" = right side, "other" = left side
    username = data.get("username", "User")

    if not text:
        return jsonify({"error": "empty text"}), 400

    result = classify(text)

    msg = {
        "id":         str(uuid.uuid4()),
        "text":       text,
        "sender":     sender,
        "username":   username,
        "timestamp":  datetime.now().strftime("%H:%M"),
        "date":       datetime.now().isoformat(),
        "revealed":   False,
        "reported":   False,
        **result,
    }

    messages[msg["id"]] = msg
    message_order.append(msg["id"])

    return jsonify(msg), 200


@app.route("/api/messages", methods=["GET"])
def get_messages():
    """Return all messages in chronological order."""
    return jsonify([messages[mid] for mid in message_order]), 200


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """
    Mark a message as confirmed hate speech.
    Appends it to the training dataset CSV so future retraining benefits.

    Body JSON:  { "message_id": "...", "correct_label": "Profanity" }
    """
    data       = request.get_json(force=True)
    mid        = data.get("message_id")
    user_label = data.get("correct_label", "Offensive_Untargetede")

    if mid not in messages:
        return jsonify({"error": "message not found"}), 404

    msg = messages[mid]
    msg["reported"] = True

    text = msg["text"]

    # 1. Append to the main training CSV
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([text, user_label, "feedback"])

    # 2. Append to a separate feedback log for auditing
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            mid,
            text,
            user_label,
            msg.get("offensive_score", ""),
            msg.get("label", ""),
        ])

    return jsonify({"status": "ok", "message": "Feedback saved to dataset"}), 200


@app.route("/api/reveal", methods=["POST"])
def reveal():
    """Toggle the 'revealed' flag so the UI can show blurred content."""
    data = request.get_json(force=True)
    mid  = data.get("message_id")
    if mid not in messages:
        return jsonify({"error": "not found"}), 404
    messages[mid]["revealed"] = not messages[mid]["revealed"]
    return jsonify(messages[mid]), 200


@app.route("/api/stats", methods=["GET"])
def stats():
    """Quick summary stats for the dashboard badge."""
    total     = len(messages)
    offensive = sum(1 for m in messages.values() if m["is_offensive"])
    reported  = sum(1 for m in messages.values() if m["reported"])
    return jsonify({
        "total":     total,
        "offensive": offensive,
        "safe":      total - offensive,
        "reported":  reported,
    }), 200


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nChat server running at http://127.0.0.1:5000")
    print("Open chat.html in your browser to start chatting.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
