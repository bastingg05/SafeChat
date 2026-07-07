"""
generate_and_test.py
--------------------
Generates a fresh, unseen adversarial test dataset with known ground-truth labels,
then evaluates the SafeChat AI against it.

These sentences use patterns NOT present in the training data,
so this is a genuine out-of-distribution test.
"""

import random
import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

random.seed(7)  # Fixed seed for reproducibility

# ── SAFE: Template slots ───────────────────────────────────
SAFE_SUBJECTS   = ["njan", "nee", "avan", "aval", "njangal", "ningal", "friend", "bro", "chetta", "teacher"]
SAFE_VERBS      = ["varunnu", "pokunnu", "parayunnu", "kelkkunnu", "kazhichu", "cheyyunnu", "nokkunu", "paranju", "undaavum", "theernu"]
SAFE_OBJECTS    = ["school il", "college il", "home il", "market il", "hospital il", "library il", "station il", "bus il", "park il", "beach il"]
SAFE_FOOD       = ["poori", "biriyani", "chaya", "sadya", "pizza", "sandwich", "parippuvada", "mango", "pazham pori", "fish curry", "rice", "cake", "idli", "dosa", "puttu"]
SAFE_TOPICS     = ["exam", "homework", "project", "result", "match", "movie", "game", "ticket", "photo", "class", "lab", "attendance", "note", "wifi", "phone"]
SAFE_ADJ        = ["nalla", "nannayundu", "superb", "ok", "ready", "late", "clear", "good", "correct", "fine"]
SAFE_FILLERS    = ["aano", "aanu", "aayee", "undo", "cheythiyo", "paranju", "kittiyo", "poyi", "vanna", "undaakkano"]

safe_templates = [
    "{sub} {obj} {verb}",
    "{sub} {topic} {filler}",
    "{food} kazhicchu {sub}",
    "{food} undaakkan {verb}",
    "{sub} {adj} aanu {topic}",
    "{topic} {filler} bro",
    "{sub} {obj} {adj} aanu",
    "enthu {topic} {filler}",
    "{food} and curry {adj} aanu",
    "{sub} {topic} cheyyano",
    "nale {topic} undaavum",
    "{obj} {adj} aayirunnu",
    "{sub} {adj} aanu ippo",
    "{topic} kittiyo {sub}",
    "{food} order cheyyam",
]

def gen_safe(n):
    rows = []
    for _ in range(n):
        tmpl = random.choice(safe_templates)
        s = tmpl.format(
            sub   = random.choice(SAFE_SUBJECTS),
            verb  = random.choice(SAFE_VERBS),
            obj   = random.choice(SAFE_OBJECTS),
            food  = random.choice(SAFE_FOOD),
            topic = random.choice(SAFE_TOPICS),
            adj   = random.choice(SAFE_ADJ),
            filler= random.choice(SAFE_FILLERS),
        )
        rows.append({"text": s.strip(), "true_label": "Not_offensive"})
    return rows

# ── OFFENSIVE: Template slots ──────────────────────────────
OFF_SLURS      = ["thendi", "thayoli", "myre", "pundi", "punda", "kunna", "kundi", "panni", "naari", "kazhutha"]
OFF_TARGETS    = ["nee", "ninte", "avan", "aval", "ninakk", "ninte amme", "ninte achan", "ee manushyan", "ee patti", "ningal"]
OFF_VERBS      = ["aanu", "aanennu ariyam", "pole parayunnu", "aayee theernu", "aayi nadakku", "mathram aanu", "ayi irikkum", "pole undu", "aayi poyi", "pole kaanunnu"]
OFF_INTENSIFIERS = ["oru", "ee", "absolute", "certified", "parama", "valiya", "chinna", "pure", "complete", "oru valiya"]

off_templates = [
    "{target} {intensifier} {slur} {verb}",
    "poda {slur} {target}",
    "{target} {slur} aanennu ellarkum ariyam",
    "enthoru {slur} {target}",
    "{intensifier} {slur} aanu {target}",
    "{target} {slur} pole {verb}",
    "nee {intensifier} {slur} aanu",
    "{slur} aayi theernu {target}",
    "{target} {slur} aanu ithu sathyam",
    "eda {slur} {target} enthu parayunnu",
    "{target} life {slur} pole aanu",
    "kure {slur} ningalude koode varathe",
    "{slur} mathram aanu {target}",
    "ninte face {slur} pole undu",
    "{target} {intensifier} {slur} anennu parayunnu",
]

def gen_offensive(n):
    rows = []
    for _ in range(n):
        tmpl = random.choice(off_templates)
        s = tmpl.format(
            slur        = random.choice(OFF_SLURS),
            target      = random.choice(OFF_TARGETS),
            verb        = random.choice(OFF_VERBS),
            intensifier = random.choice(OFF_INTENSIFIERS),
        )
        rows.append({"text": s.strip(), "true_label": "Offensive"})
    return rows

N_EACH = 5000  # 5000 safe + 5000 offensive = 10000 total (large scale)
safe_data      = gen_safe(N_EACH)
offensive_data = gen_offensive(N_EACH)

test_data = safe_data + offensive_data
random.shuffle(test_data)

print(f"Generated {len(test_data)} unseen test sentences ({N_EACH} safe + {N_EACH} offensive)\n")

# Save the generated test set
df_gen = pd.DataFrame(test_data)
df_gen.to_csv("generated_test_dataset.csv", index=False)
print("Saved generated test dataset → generated_test_dataset.csv\n")

# ─────────────────────────────────────────────────────────
# 2. VERIFY NO OVERLAP WITH TRAINING DATA
# ─────────────────────────────────────────────────────────
print("Checking for overlap with training data...")
df_train = pd.read_csv('mal_full_offensive_train.csv', sep='\t', header=None,
                       names=['text', 'label', 'extra'], on_bad_lines='skip')
train_texts = set(df_train['text'].dropna().str.strip().str.lower())
gen_texts = set(df_gen['text'].str.strip().str.lower())

overlap = train_texts & gen_texts
if overlap:
    print(f"⚠️  WARNING: {len(overlap)} overlapping sentences found and will be removed!")
    df_gen = df_gen[~df_gen['text'].str.lower().isin(overlap)]
else:
    print(f"✅  Zero overlap with training data! All {len(df_gen)} sentences are truly unseen.\n")

# ─────────────────────────────────────────────────────────
# 3. RUN MODEL INFERENCE
# ─────────────────────────────────────────────────────────
print("Loading AI model from ./finetuned_model ...")
MODEL_DIR = "./finetuned_model"
if not os.path.exists(MODEL_DIR):
    print("ERROR: finetuned_model not found!")
    exit(1)

# Import the SAME preprocessing used by chat_server.py
from preprocessing import preprocess_text
print("Preprocessing module loaded.")

classifier = pipeline("text-classification", model=MODEL_DIR, top_k=None)
print("Model loaded. Running inference with full preprocessing pipeline...\n")

SAFE_LABELS = {"Not_offensive", "Not_in_intended_language"}

results = []
for _, row in df_gen.iterrows():
    # Apply FULL preprocessing pipeline (same as chat_server.py)
    processed = preprocess_text(row["text"])
    raw = classifier(processed)
    preds = raw[0] if isinstance(raw[0], list) else raw
    safe_score = sum(r["score"] for r in preds if r["label"] in SAFE_LABELS)
    offensive_score = 1.0 - safe_score
    is_offensive = offensive_score > 0.8
    predicted = "Offensive" if is_offensive else "Not_offensive"
    results.append({
        "text": row["text"],
        "processed": processed,
        "true_label": row["true_label"],
        "predicted": predicted,
        "offensive_score": round(offensive_score, 4),
        "correct": (row["true_label"] == "Offensive") == is_offensive
    })

df_results = pd.DataFrame(results)

# ─────────────────────────────────────────────────────────
# 4. COMPUTE HONEST METRICS
# ─────────────────────────────────────────────────────────
true_labels = df_results["true_label"].tolist()
pred_labels = df_results["predicted"].tolist()

accuracy  = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, pos_label="Offensive", zero_division=0)
recall    = recall_score(true_labels, pred_labels, pos_label="Offensive", zero_division=0)
f1        = f1_score(true_labels, pred_labels, pos_label="Offensive", zero_division=0)

print("=" * 55)
print("   HONEST EVALUATION ON UNSEEN GENERATED DATA")
print("=" * 55)
print(f"  Test Set Size  : {len(df_results)} sentences")
print(f"  Accuracy       : {accuracy * 100:.2f}%")
print(f"  Precision      : {precision * 100:.2f}%")
print(f"  Recall         : {recall * 100:.2f}%")
print(f"  F1-Score       : {f1 * 100:.2f}%")
print("=" * 55)
print("\nDetailed Classification Report:")
print(classification_report(true_labels, pred_labels, zero_division=0))

# Print mistakes
mistakes = df_results[df_results["correct"] == False]
if len(mistakes) > 0:
    print(f"\nMistakes made by AI ({len(mistakes)} total):")
    for _, m in mistakes.iterrows():
        print(f"  ❌ [{m['true_label']}] → predicted [{m['predicted']}]  | '{m['text']}'")
else:
    print("\n✅  No mistakes! Perfect score on unseen data.")

# ─────────────────────────────────────────────────────────
# 5. SAVE VISUALIZATIONS
# ─────────────────────────────────────────────────────────
labels_order = ["Not_offensive", "Offensive"]
cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_order, yticklabels=labels_order)
plt.title('Honest Confusion Matrix\n(Unseen Generated Test Data)', fontsize=14)
plt.ylabel('Ground Truth Label')
plt.xlabel('AI Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_unseen.png', dpi=300)
plt.close()

# Confidence distribution
scores = df_results["offensive_score"].tolist()
plt.figure(figsize=(10, 5))
correct_scores   = [r["offensive_score"] for _, r in df_results.iterrows() if r["correct"]]
incorrect_scores = [r["offensive_score"] for _, r in df_results.iterrows() if not r["correct"]]
plt.hist(correct_scores,   bins=20, alpha=0.7, color='#22c55e', label='Correct predictions')
plt.hist(incorrect_scores, bins=20, alpha=0.7, color='#ef4444', label='Wrong predictions')
plt.axvline(x=0.8, color='orange', linestyle='--', label='Decision threshold (0.8)')
plt.xlabel('Offensive Score')
plt.ylabel('Count')
plt.title('Confidence Distribution on Unseen Test Data')
plt.legend()
plt.tight_layout()
plt.savefig('confidence_histogram_unseen.png', dpi=300)
plt.close()

print("\nSaved → confusion_matrix_unseen.png")
print("Saved → confidence_histogram_unseen.png")
print("\n✅  Evaluation complete!")
