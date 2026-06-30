"""
chat_server.py  —  Flask backend for the WhatsApp-style Malayalam chat app.

Endpoints:
  POST /api/send       — classify a message and store it
  GET  /api/messages   — retrieve full message history
  POST /api/feedback   — mark a message as hate speech → appends to training CSV
  POST /api/classify   — user-classify a message as hate_speech / not_hate_speech
"""

import csv
import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, make_response
from transformers import pipeline
import re

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__)

# Manual CORS handler — needed because file:// sends Origin: null
# which Flask-CORS doesn't handle properly with wildcards
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        resp = make_response("", 200)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

DATASET_PATH = os.path.join(os.path.dirname(__file__), "mal_full_offensive_train.csv")
FEEDBACK_LOG  = os.path.join(os.path.dirname(__file__), "feedback_log.csv")
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "finetuned_model")
USER_CHAT_DATASET = os.path.join(os.path.dirname(__file__), "user_chat_dataset.csv")

# Load existing chat texts to prevent duplication
saved_chat_texts = set()
if os.path.exists(USER_CHAT_DATASET):
    with open(USER_CHAT_DATASET, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row:
                saved_chat_texts.add(row[0])

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

# ─────────────────────────────────────────────
# Normalization & Transliteration Pipeline
# ─────────────────────────────────────────────
# Using Regex allows us to catch extended words (e.g. 'myreee', 'mandannna')
REGEX_VARIANTS = [
    (r'\bkashuvandi[a-z]*\b', 'cashew'), # PROTECT safe words from 'andi' subword tokenization
    (r'\bmandan[a-z]*\b', 'mandan'), # mandanna, mandanaa
    (r'\bpann[i]+[a-z]*\b', 'panni'),     # pannii, panniii
    (r'\bm[ya]i?r[a-z]*\b', 'myr'),   # myre, myree, myran, myresh
    (r'\bpo[o]+d[a-z]*\b', 'poda'),   # poda, pooda, podaa
    (r'\b(?:kunj[u]?)?andi[a-z]*\b', 'myr'),    # FORCE ambiguous 'andi' to definitive 'myr'
    (r'\bkunda[a-z]*\b', 'kundan'), # kundan, kundappy, kundaa
    (r'\bpo[o]+ri[a-z]*\b', 'poori'), # poorimone, poorimakkal
    (r'\bthayo[a-z]*\b', 'thayoli'), # thayoli, thayolikal
    (r'\bkunn[a-z]*\b', 'kunna'),   # kunna, kunnayoli
    (r'\bpa[a]+ri[a-z]*\b', 'pari'),   # paari, pariyol
    (r'\bchettat[th]*aram[a-z]*\b', 'myr'), # Force block 'chettatharam', 'chettattharam'
    (r'\bthanthayillath[a-z]*\b', 'myr'), # thanthayillathavane (fatherless)
    (r'\bkundi[a-z]*\b', 'myr'), # kundi, kundimyre
    (r'\bkindi[a-z]*\b', 'myr'), # kindi (slang for stupid/useless)
    (r'\bpolayadi[a-z]*\b', 'myr'), # polayadi (casteist/prostitute slur)
    (r'\bpunda[a-z]*\b', 'myr'), # pundachi
]

def remove_repeated_characters(text):
    # Reduce 3 or more consecutive identical characters to 2
    # So "myreeeeee" becomes "myree", which our Regex will easily catch
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def custom_manglish_dictionary(text):
    # Apply regex rules to catch extended slurs
    for pattern, replacement in REGEX_VARIANTS:
        text = re.sub(pattern, replacement, text)
    return text

def preprocess_text(text):
    text = text.lower()
    text = remove_repeated_characters(text)
    text = custom_manglish_dictionary(text)
    return text

def classify(text: str) -> dict:
    """Run the model and return a structured result dict."""
    processed_text = preprocess_text(text)
    print(f"Normalized & Transliterated: '{text}' -> '{processed_text}'")
    raw = classifier(processed_text)
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
    
    # Save user chat without duplication
    if text not in saved_chat_texts:
        with open(USER_CHAT_DATASET, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            # We save the raw text, the AI's predicted label, and the source
            writer.writerow([text, result["label"], "user_chat"])
        saved_chat_texts.add(text)

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


@app.route("/api/classify", methods=["POST", "OPTIONS"])
def classify_message():
    """
    User-classify a message as 'hate_speech' or 'not_hate_speech'.
    This labels the message and appends it to the training CSV + feedback log.

    Body JSON:  { "message_id": "...", "classification": "hate_speech" | "not_hate_speech" }
    """
    data           = request.get_json(force=True)
    mid            = data.get("message_id")
    classification = data.get("classification", "").strip().lower()

    if mid not in messages:
        return jsonify({"error": "message not found"}), 404

    if classification not in ("hate_speech", "not_hate_speech"):
        return jsonify({"error": "classification must be 'hate_speech' or 'not_hate_speech'"}), 400

    msg = messages[mid]
    msg["user_classification"] = classification

    # Map to training label
    train_label = "Offensive_Untargetede" if classification == "hate_speech" else "Not_offensive"

    # 1. Append to the main training CSV
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([msg["text"], train_label, "user_classify"])

    # 2. Append to separate feedback log for auditing
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            mid,
            msg["text"],
            train_label,
            msg.get("offensive_score", ""),
            msg.get("label", ""),
            f"user_classify:{classification}",
        ])

    return jsonify({
        "status": "ok",
        "classification": classification,
        "message": f"Classified as {classification} and saved to dataset"
    }), 200


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
    classified_hate     = sum(1 for m in messages.values() if m.get("user_classification") == "hate_speech")
    classified_not_hate = sum(1 for m in messages.values() if m.get("user_classification") == "not_hate_speech")
    return jsonify({
        "total":               total,
        "offensive":           offensive,
        "safe":                total - offensive,
        "reported":            reported,
        "classified_hate":     classified_hate,
        "classified_not_hate": classified_not_hate,
    }), 200


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nChat server running at http://127.0.0.1:5000")
    print("Open chat.html in your browser to start chatting.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
