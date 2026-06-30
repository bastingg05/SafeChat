import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import os

print("========================================")
print("AI BRAIN EVALUATION & VISUALIZATION")
print("========================================")

print("1. Loading unseen test dataset...")
df_raw = pd.read_csv('mal_full_offensive_train.csv', sep='\t', header=None, names=['text', 'label', 'extra'], on_bad_lines='skip')
df_valid = df_raw.dropna(subset=['text', 'label'])
df_test = df_valid.sample(n=min(len(df_valid), 2000), random_state=99).reset_index(drop=True)

label_mapping = {
    'Not_offensive': 'Not_offensive',
    'not-malayalam': 'Not_in_intended_language',
    'Offensive_Targeted_Insult_Group': 'Off_target_group',
    'Offensive_Untargetede': 'Profanity',
    'Offensive_Targeted_Insult_Individual': 'Off_target_ind'
}
df_test['mapped_label'] = df_test['label'].map(label_mapping)
df_test = df_test.dropna(subset=['mapped_label'])

print("2. Loading exported AI Brain (./finetuned_model)...")
if not os.path.exists("./finetuned_model"):
    print("Error: Exported brain not found in ./finetuned_model")
    exit(1)

classifier = pipeline("text-classification", model="./finetuned_model", top_k=1, device=-1)

texts = df_test["text"].tolist()
true_labels = df_test["mapped_label"].tolist()

print("3. Running Inference to calculate scores...")
batch_size = 32
predicted_labels = []
confidence_scores = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    preds = classifier(batch_texts)
    for p in preds:
        top_pred = p[0]
        predicted_labels.append(top_pred['label'])
        confidence_scores.append(top_pred['score'])

print("\n========================================")
print("PERFORMANCE METRICS")
print("========================================")

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
avg_confidence = np.mean(confidence_scores)

print(f"Accuracy         : {accuracy * 100:.2f}%")
print(f"Precision        : {precision * 100:.2f}%")
print(f"Recall           : {recall * 100:.2f}%")
print(f"F1-Score         : {f1 * 100:.2f}%")
print(f"Avg Confidence   : {avg_confidence * 100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(true_labels, predicted_labels, zero_division=0))

print("\n========================================")
print("GENERATING VISUALIZATIONS")
print("========================================")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_labels, predicted_labels)
labels = sorted(list(set(true_labels)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('AI Brain Confusion Matrix')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("   Saved Confusion Matrix to 'confusion_matrix.png'")
plt.close()

# 2. Confidence Score Histogram
plt.figure(figsize=(10, 6))
sns.histplot(confidence_scores, bins=20, kde=True, color='purple')
plt.title('AI Brain Confidence Score Distribution')
plt.xlabel('Confidence Probability (0 to 1)')
plt.ylabel('Number of Messages')
plt.axvline(x=avg_confidence, color='r', linestyle='--', label=f'Avg: {avg_confidence:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('confidence_histogram.png', dpi=300)
print("   Saved Confidence Histogram to 'confidence_histogram.png'")
plt.close()

print("\nEvaluation complete! Use the generated PNG files for your project report.")
