import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

print("Loading original dataset...")
# Load ONLY the real human-labeled data
df_raw = pd.read_csv('mal_full_offensive_train.csv', sep='\t', header=None, names=['text', 'label', 'extra'], on_bad_lines='skip')
df_original = df_raw.iloc[:36011].dropna(subset=['text', 'label'])

# Recreate the exact 4000 rows used for training so we can EXCLUDE them
train_sample = df_original.sample(n=min(len(df_original), 4000), random_state=42)

# Get the remaining ~32,000 rows that the model has NEVER seen
df_unseen_real = df_original.drop(train_sample.index)

# Sample 2000 rows from the unseen data to keep evaluation reasonably fast
df_test = df_unseen_real.sample(n=2000, random_state=99).reset_index(drop=True)

label_mapping = {
    'Not_offensive': 'Not_offensive',
    'not-malayalam': 'Not_in_intended_language',
    'Offensive_Targeted_Insult_Group': 'Off_target_group',
    'Offensive_Untargetede': 'Profanity',
    'Offensive_Targeted_Insult_Individual': 'Off_target_ind'
}
df_test['mapped_label'] = df_test['label'].map(label_mapping)
df_test = df_test.dropna(subset=['mapped_label'])

print(f"Loaded strict eval dataset with {len(df_test)} UNSEEN real human-labeled examples.")

print("Loading fine-tuned model...")
classifier = pipeline("text-classification", model="./finetuned_model", top_k=1, device=-1)

texts = df_test["text"].tolist()
true_labels = df_test["mapped_label"].tolist()

print("Running strict inference...")
batch_size = 32
results = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    preds = classifier(batch_texts)
    for p in preds:
        results.append(p[0]['label'])

# Calculate Metrics
from collections import Counter
print(f"Label Distribution in test set: {Counter(true_labels)}")

accuracy = accuracy_score(true_labels, results)
cm = confusion_matrix(true_labels, results)
report = classification_report(true_labels, results)

print("\n--- Model Evaluation Results ---")
print(f"Accuracy : {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
print("--------------------------------")
