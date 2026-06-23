import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

print("Loading dataset...")
df_raw = pd.read_csv('mal_full_offensive_train.csv', sep='\t', header=None, names=['text', 'label', 'extra'], on_bad_lines='skip')

# The original dataset has 36011 lines. Let's split using the row index in raw DataFrame:
df_original = df_raw.iloc[:36011].dropna(subset=['text', 'label'])
df_synthetic = df_raw.iloc[36011:].dropna(subset=['text', 'label'])

# Sample original data to reduce training time
df_original_sampled = df_original.sample(n=min(len(df_original), 4000), random_state=42)

# Concatenate original sampled with synthetic data
df = pd.concat([df_original_sampled, df_synthetic]).reset_index(drop=True)

# Map dataset string labels to model expected string labels
label_mapping = {
    'Not_offensive': 'Not_offensive',
    'not-malayalam': 'Not_in_intended_language',
    'Offensive_Targeted_Insult_Group': 'Off_target_group',
    'Offensive_Untargetede': 'Profanity',
    'Offensive_Targeted_Insult_Individual': 'Off_target_ind'
}

df['mapped_label'] = df['label'].map(label_mapping)
df = df.dropna(subset=['mapped_label'])

# Map model string labels to IDs
id2label = {0: 'Not_offensive', 1: 'Not_in_intended_language', 2: 'Off_target_group', 3: 'Profanity', 4: 'Off_target_ind'}
label2id = {v: k for k, v in id2label.items()}

df['label_id'] = df['mapped_label'].map(label2id)

print(f"Sampled original examples: {len(df_original_sampled)}, synthetic examples: {len(df_synthetic)}")
print(f"Total valid training examples: {len(df)}")

dataset = Dataset.from_pandas(df[['text', 'label_id']].rename(columns={'label_id': 'label'}))

# Split into train and eval (90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

model_id = "Hate-speech-CNERG/deoffxlmr-mono-malyalam"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    use_cpu=not torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("Starting Fine-tuning...")
trainer.train()

print("Saving Fine-tuned model...")
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
print("Done! Model saved to ./finetuned_model")
