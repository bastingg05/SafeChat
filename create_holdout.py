"""
create_holdout.py
-----------------
Splits the full dataset into:
  - train_only.csv     (80%) → used exclusively for training
  - test_holdout.csv   (20%) → locked away, NEVER seen during training

Run this ONCE before training. After running, always:
  1. Train on train_only.csv
  2. Evaluate on test_holdout.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

FULL_CSV = "mal_full_offensive_train.csv"
TRAIN_CSV = "train_only.csv"
HOLDOUT_CSV = "test_holdout.csv"

print("="*50)
print("CREATING HONEST HOLDOUT SPLIT")
print("="*50)

# Load and clean the full dataset
df_raw = pd.read_csv(FULL_CSV, sep='\t', header=None, names=['text', 'label', 'extra'], on_bad_lines='skip')

# Merge User Chat Dataset
if os.path.exists("user_chat_dataset.csv"):
    df_user = pd.read_csv("user_chat_dataset.csv", sep='\t', header=None, names=['text', 'label', 'extra'], on_bad_lines='skip')
    df_raw = pd.concat([df_raw, df_user], ignore_index=True)

# Merge Feedback Log
if os.path.exists("feedback_log.csv"):
    try:
        df_feed = pd.read_csv("feedback_log.csv", header=None, on_bad_lines='skip')
        if not df_feed.empty and len(df_feed.columns) >= 4:
            df_f_clean = pd.DataFrame()
            df_f_clean['text'] = df_feed[2]
            df_f_clean['label'] = df_feed[3]
            df_f_clean['extra'] = 'user_feedback'
            df_raw = pd.concat([df_raw, df_f_clean], ignore_index=True)
    except Exception as e:
        print(f"Warning: Could not merge feedback_log.csv: {e}")

df = df_raw.dropna(subset=['text', 'label']).drop_duplicates(subset=['text'], keep='last')

label_mapping = {
    'Not_offensive': 'Not_offensive',
    'not-malayalam': 'Not_in_intended_language',
    'Offensive_Targeted_Insult_Group': 'Off_target_group',
    'Offensive_Untargetede': 'Profanity',
    'Offensive_Targeted_Insult_Individual': 'Off_target_ind',
    'Profanity': 'Profanity',
    'Off_target_group': 'Off_target_group',
    'Off_target_ind': 'Off_target_ind',
    'Not_in_intended_language': 'Not_in_intended_language'
}

df['mapped_label'] = df['label'].map(label_mapping)
df = df.dropna(subset=['mapped_label'])

print(f"Total cleaned rows: {len(df)}")
print(f"Label distribution:\n{df['mapped_label'].value_counts()}\n")

# Filter out classes with fewer than 2 instances to prevent stratification errors
class_counts = df['mapped_label'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['mapped_label'].isin(valid_classes)]

# Stratified split: 80% train, 20% holdout
# Using a DIFFERENT seed from train_model.py (which uses seed=42) to ensure NO overlap
df_train, df_holdout = train_test_split(
    df,
    test_size=0.20,
    random_state=99,       # Deliberately different seed than train_model.py
    stratify=df['mapped_label']
)

print(f"Training set size : {len(df_train)} rows (80%)")
print(f"Holdout set size  : {len(df_holdout)} rows (20%)")
print(f"\nHoldout label distribution:\n{df_holdout['mapped_label'].value_counts()}\n")

# Save the splits
df_train[['text', 'label', 'extra']].to_csv(TRAIN_CSV, sep='\t', header=False, index=False)
df_holdout[['text', 'mapped_label']].to_csv(HOLDOUT_CSV, sep='\t', header=False, index=False)

print(f"Saved training data   -> {TRAIN_CSV}")
print(f"Saved holdout data    -> {HOLDOUT_CSV}")
print("\nIMPORTANT: Now retrain your model on train_only.csv, then evaluate on test_holdout.csv")
print("Never add test_holdout.csv into training data!")
