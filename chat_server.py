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
from flask import Flask, request, jsonify, make_response, send_file
from transformers import pipeline
import re
import signal
import sys

# Fix Windows console encoding — allows printing Malayalam/Unicode characters
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import subprocess
import json
import tempfile

# Ensure winget's actual ffmpeg binary folder is in PATH
ffmpeg_bin = r"C:\Users\basti\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
if ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + ffmpeg_bin

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

# Load User Overrides (Whitelist) from the training dataset
USER_OVERRIDES = {}
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3 and row[2] in ("user_classify", "feedback"):
                USER_OVERRIDES[row[0].strip().lower()] = row[1]

# In-memory message store  {id: message_dict}
messages: dict = {}
message_order: list = []   # keeps insertion order

CHAT_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                for msg in history:
                    messages[msg["id"]] = msg
                    message_order.append(msg["id"])
        except Exception as e:
            print(f"Error loading chat history: {e}")

def save_chat_history():
    history = [messages[mid] for mid in message_order]
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

load_chat_history()

# ─────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────
print("Loading fine-tuned model...")
classifier = pipeline("text-classification", model=MODEL_DIR, top_k=None)
print("Model loaded successfully!")

print("Loading Whisper STT model (faster-whisper)...")
try:
    from faster_whisper import WhisperModel
    whisper_stt = WhisperModel("./vegam-model", device="auto", compute_type="int8")
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_stt = None

SAFE_LABELS = {"Not_offensive", "Not_in_intended_language"}

LABEL_META = {
    "Not_offensive":           {"display": "Safe",                  "severity": "safe"},
    "Not_in_intended_language":{"display": "Not Malayalam",         "severity": "safe"},
    "Off_target_group":        {"display": "Group Hate Speech",     "severity": "high"},
    "Profanity":               {"display": "Profanity / Vulgarity", "severity": "high"},
    "Off_target_ind":          {"display": "Personal Attack",       "severity": "high"},
}

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess_text, BASE_SLURS

def classify(text: str, is_audio: bool = False) -> dict:
    """Run the model and return a structured result dict."""
    raw_text = text.strip().lower()
    
    # 1. Check User Override Cache (Instant Bypass)
    if raw_text in USER_OVERRIDES:
        forced_label = USER_OVERRIDES[raw_text]
        print(f"Bypassing AI -> User Override Cache hit for '{raw_text}': {forced_label}")
        is_offensive = forced_label != "Not_offensive"
        meta = LABEL_META.get(forced_label, {"display": forced_label, "severity": "high" if is_offensive else "safe"})
        return {
            "label": forced_label,
            "label_display": meta["display"],
            "offensive_score": 1.0 if is_offensive else 0.0,
            "safe_score": 0.0 if is_offensive else 1.0,
            "is_offensive": is_offensive,
            "bucket": meta["severity"],
            "all_scores": {forced_label: 1.0}
        }

    # 2. Normal AI Pipeline
    if is_audio:
        # Voice input: already in Malayalam script, feed directly to model (no transliteration)
        processed_text = text.strip()
        print(f"Voice input (Malayalam script, no transliteration): '{processed_text}'")

    else:
        # Typed input: Manglish, apply full preprocessing with transliteration
        processed_text = preprocess_text(text)
        print(f"Normalized & Transliterated: '{text}' -> '{processed_text}'")

    # Intercept single safe words
    if processed_text.strip() == "friend":
        return {
            "label": "Not_offensive", 
            "label_display": "Safe",
            "offensive_score": 0.0,
            "safe_score": 1.0,
            "is_offensive": False,
            "bucket": "safe",
            "all_scores": {"Not_offensive": 1.0}
        }

    raw = classifier(processed_text)
    results = raw[0] if (raw and isinstance(raw[0], list)) else raw

    safe_score      = sum(r["score"] for r in results if r["label"] in SAFE_LABELS)
    offensive_score = 1.0 - safe_score
    
    # ── Keyword Gating (Slur Catch-all) ──
    # If a known slur survives preprocessing but the AI misses it due to bias,
    # force the offensive score high.
    if not is_audio:
        words = processed_text.split()
        # Check if any BASE_SLUR is an EXACT match of any word in the processed text.
        # (We removed the substring 'in' check so names like 'Poorna' aren't falsely flagged as 'poor').
        if any(slur == word for word in words for slur in BASE_SLURS):
            if offensive_score < 0.8:
                print(f"Keyword Gating triggered! Forced Profanity for text containing slur: {processed_text}")
                offensive_score = 0.99
                safe_score = 0.01
                results = [{"label": "Profanity", "score": 0.99}, {"label": "Not_offensive", "score": 0.01}]

    is_offensive = offensive_score > 0.8  # Increased threshold to 80% to maximize precision and reduce false positives

    # If keyword gating fired, force the top label to Profanity
    if offensive_score == 0.99:
        top = {"label": "Profanity", "score": 0.99}
    else:
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

@app.route("/")
def index():
    return send_file("chat.html")


@app.route("/api/send", methods=["POST"])
def send_message():
    """
    Body JSON:
      { "text": "...", "sender": "me" | "other", "username": "Alice", "is_audio": boolean }
    Returns the full message object including classification.
    """
    data     = request.get_json(force=True)
    text     = data.get("text", "").strip()
    sender   = data.get("sender", "me")       # "me" = right side, "other" = left side
    username = data.get("username", "User")
    is_audio = data.get("is_audio", False)
    guard_on = data.get("guard_on", True)

    if not text:
        return jsonify({"error": "empty text"}), 400

    if guard_on:
        result = classify(text, is_audio=is_audio)
    else:
        result = {
            "label": "Not_offensive",
            "offensive_score": 0.0,
            "bucket": "safe",
            "transliterated": text
        }
    
    # Save user chat without duplication
    if text not in saved_chat_texts:
        with open(USER_CHAT_DATASET, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
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
        "is_audio":   is_audio,
        **result,
    }

    messages[msg["id"]] = msg
    message_order.append(msg["id"])

    save_chat_history()

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
        
    # Instant Cache Update
    USER_OVERRIDES[text.strip().lower()] = user_label

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

    save_chat_history()

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
        
    # Instant Cache Update
    USER_OVERRIDES[msg["text"].strip().lower()] = train_label

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

    save_chat_history()

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
    save_chat_history()
    return jsonify({"status": "ok", "revealed": messages[mid]["revealed"]}), 200


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


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    """Transcribes an uploaded audio blob using Whisper."""
    if whisper_stt is None:
        return jsonify({"error": "Whisper model not loaded on server."}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty audio file"}), 400

    # Save to a temporary file for Whisper to read
    fd, tmp_path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    
    try:
        audio_file.save(tmp_path)
        # Run Whisper inference
        segments, info = whisper_stt.transcribe(tmp_path, beam_size=5)
        transcribed_text = "".join([segment.text for segment in segments]).strip()
        
        # ─────────────────────────────────────────────
        # Whisper Hallucination Filter
        # ─────────────────────────────────────────────
        # Whisper often hallucinates looping text (e.g., "കോട്ട്ട്ട്ട്ട്ട്...") during pure silence or static.
        # If the text is abnormally long but contains very few unique characters, squash it.
        if len(transcribed_text) > 15 and len(set(transcribed_text)) < 8:
            print(f"[Whisper] Suppressed silence hallucination: {transcribed_text}")
            transcribed_text = ""
            
        return jsonify({"status": "ok", "text": transcribed_text}), 200
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return jsonify({"error": "Failed to transcribe audio"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def graceful_shutdown(signum, frame):
    print("\n\n[SafeChat Server] Received shutdown signal (Ctrl+C).")
    print("[SafeChat Server] All datasets (Memory) are already safely flushed to disk.")
    
    try:
        ans = input("\nDo you want to export/update the AI Brain with the latest memory before closing? (y/n): ")
        if ans.lower().strip() == 'y':
            print("[SafeChat Server] Starting Brain Export. DO NOT close this window!")
            subprocess.run(["python", "update_brain.py"])
        else:
            print("Skipping Brain Export. Exiting immediately.")
    except Exception:
        print("\nExiting.")
        
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, graceful_shutdown)
    print("\nChat server running at http://127.0.0.1:5000")
    print("Open chat.html in your browser to start chatting.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
