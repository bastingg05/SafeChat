"""
label_tool_server.py
--------------------
A mini web server that serves a beautiful human labeling interface.
Run: python label_tool_server.py
Then open: http://127.0.0.1:5001
"""

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

INPUT_CSV  = "user_chat_dataset.csv"
OUTPUT_JSON = "human_verified_labels.json"

# Load the chat dataset
df = pd.read_csv(INPUT_CSV, sep='\t', header=None, names=['text', 'ai_label', 'source'], on_bad_lines='skip')
df = df.dropna(subset=['text', 'ai_label']).drop_duplicates(subset=['text'])
records = df.to_dict(orient='records')

# Load existing progress
verified = {}
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
        verified = json.load(f)

@app.route('/')
def index():
    return send_from_directory('.', 'label_tool.html')

@app.route('/api/items')
def get_items():
    return jsonify({
        "items": records,
        "total": len(records),
        "verified": verified
    })

@app.route('/api/label', methods=['POST'])
def save_label():
    data = request.get_json()
    text = data.get('text')
    human_label = data.get('label')
    verified[text] = human_label
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)
    return jsonify({"saved": len(verified), "total": len(records)})

@app.route('/api/evaluate')
def evaluate():
    """Compare AI labels vs human labels and compute real metrics."""
    if len(verified) < 20:
        return jsonify({"error": f"Need at least 20 verified labels. You have {len(verified)}."})

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    texts, ai_labels, human_labels = [], [], []
    for r in records:
        t = r['text']
        if t in verified:
            texts.append(t)
            ai_labels.append(r['ai_label'])
            human_labels.append(verified[t])

    # Normalize labels
    def normalize(label):
        offensive = {'Profanity', 'Offensive_Untargetede', 'Off_target_ind', 'Off_target_group', 'offensive', 'hate_speech'}
        if label in offensive or label.lower() in {'offensive', 'hate_speech', 'profanity'}:
            return 'Offensive'
        return 'Not_offensive'

    ai_norm = [normalize(l) for l in ai_labels]
    hu_norm = [normalize(l) for l in human_labels]

    accuracy  = accuracy_score(hu_norm, ai_norm)
    precision = precision_score(hu_norm, ai_norm, pos_label='Offensive', zero_division=0)
    recall    = recall_score(hu_norm, ai_norm, pos_label='Offensive', zero_division=0)
    f1        = f1_score(hu_norm, ai_norm, pos_label='Offensive', zero_division=0)
    report    = classification_report(hu_norm, ai_norm, zero_division=0)

    return jsonify({
        "verified_count": len(texts),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "report": report
    })

if __name__ == '__main__':
    print(f"\nHuman Labeling Tool ready!")
    print(f"Open: http://127.0.0.1:5001")
    print(f"Total messages to label: {len(records)}")
    print(f"Already verified: {len(verified)}\n")
    app.run(port=5001, debug=False)
