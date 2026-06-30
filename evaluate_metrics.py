import pandas as pd
from datasets import Dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

print("Loading dataset...")
df_raw = pd.read_csv('mal_full_offensive_train.csv', sep='\t', header=None, names=['text', 'label', 'extra'], on_bad_lines='skip')

# Apply the same data filtering as training to get the same test set
df_original = df_raw.iloc[:36011].dropna(subset=['text', 'label'])
df_synthetic = df_raw.iloc[36011:].dropna(subset=['text', 'label'])

df_original_sampled = df_original.sample(n=min(len(df_original), 4000), random_state=42)
df = pd.concat([df_original_sampled, df_synthetic]).reset_index(drop=True)

label_mapping = {
    'Not_offensive': 'Not_offensive',
    'not-malayalam': 'Not_in_intended_language',
    'Offensive_Targeted_Insult_Group': 'Off_target_group',
    'Offensive_Untargetede': 'Profanity',
    'Offensive_Targeted_Insult_Individual': 'Off_target_ind'
}
df['mapped_label'] = df['label'].map(label_mapping)
df = df.dropna(subset=['mapped_label'])

# Recreate the train/test split to isolate the exact 10% eval set
dataset = Dataset.from_pandas(df[['text', 'mapped_label']].rename(columns={'mapped_label': 'label'}))
dataset = dataset.train_test_split(test_size=0.1, seed=42)
test_dataset = dataset["test"]

print(f"Loaded eval dataset with {len(test_dataset)} examples.")

print("Loading fine-tuned model...")
classifier = pipeline("text-classification", model="./finetuned_model", top_k=1, device=-1) # CPU for simplicity/compat

true_labels = []
pred_labels = []

print("Running inference on eval set...")
# Running in batches to speed up
batch_size = 32
texts = test_dataset["text"]
true_labels = test_dataset["label"]

results = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    preds = classifier(batch_texts)
    # The pipeline with top_k=1 returns a list of lists of dicts
    for p in preds:
        results.append(p[0]['label'])

# Calculate Metrics
accuracy = accuracy_score(true_labels, results)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, results, average='weighted', zero_division=0)

print("\n--- Model Evaluation Results ---")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")
print("--------------------------------")
