import pandas as pd
import os
import csv

# 1. Define files and mapping
files = [
    'data_exploration/DravidianCodeMix-2020/DravidianCodeMix/mal_full_offensive.csv',
    'data_exploration/DravidianCodeMix-2020/DravidianCodeMix/mal_full_sentiment.tsv',
    'data_exploration/DravidianCodeMix-2020/DravidianCodeMix/mal_full_offensive_dev.csv',
    'data_exploration/DravidianCodeMix-2020/DravidianCodeMix/mal_full_sentiment_dev.csv',
    'data_exploration/DravidianCodeMix-2020/DravidianCodeMix/mal_full_offensive_test.csv',
    'data_exploration/DravidianCodeMix-2020/DravidianCodeMix/mal_full_sentiment_test.csv',
    'data_exploration/DravidianCodeMix-2020/DravidianCodeMix/mal_full_sentiment_train.csv',
    'data_exploration/Offensive_language_Malayalam-main/Malayalam_offensive_data_Training-YT (1).csv'
]

# Map Sentiment to Safe, and map any legacy offensive terms to our 5 core labels
LABEL_MAPPING = {
    # Sentiment mappings
    'Positive': 'Not_offensive',
    'Neutral': 'Not_offensive',
    'Mixed_feelings': 'Not_offensive',
    'Negative': 'Not_offensive',
    'unknown_state': 'Not_in_intended_language',
    
    # Clean offensive mappings just in case
    'Offensive_Untargetede': 'Profanity',  # Treating untargeted as profanity in our UI
    'not-malayalam': 'Not_in_intended_language'
}

master_csv = 'mal_full_offensive_train.csv'

print("Starting massive dataset merge...")
total_added = 0

with open(master_csv, 'a', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out, delimiter='\t')
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Skipping (not found): {file_path}")
            continue
            
        print(f"Processing {file_path}...")
        sep = '\t' if file_path.endswith('.tsv') else ','
        
        try:
            df = pd.read_csv(file_path, sep=sep, header=None, on_bad_lines='skip', engine='python')
            
            # Assume last column is label, the one before is text (if there's an ID column)
            # Generally, DravidianCodeMix has [id, text, label]
            if len(df.columns) >= 2:
                text_col = df.columns[-2]
                label_col = df.columns[-1]
                
                # If there's no ID column (just text, label), text_col will be 0 and label_col 1
                if len(df.columns) == 2:
                    text_col = df.columns[0]
                    label_col = df.columns[1]
                
                df_clean = df[[text_col, label_col]].dropna().copy()
                df_clean[text_col] = df_clean[text_col].astype(str).str.strip()
                df_clean[label_col] = df_clean[label_col].astype(str).str.strip()
                
                # Apply mapping, if not in mapping, keep original label
                df_clean[label_col] = df_clean[label_col].map(lambda x: LABEL_MAPPING.get(x, x))
                
                for _, row in df_clean.iterrows():
                    text = row[text_col]
                    label = row[label_col]
                    if text and label:
                        writer.writerow([text, label, f"mass_merge_{os.path.basename(file_path)}"])
                        total_added += 1
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

print(f"Appended {total_added} raw rows. Now optimizing master dataset...")

# Optimize and drop duplicates
master_df = pd.read_csv(master_csv, sep='\t', header=None, on_bad_lines='skip')
original_len = len(master_df)
master_df = master_df.drop_duplicates(subset=[0], keep='last')
final_len = len(master_df)
master_df.to_csv(master_csv, sep='\t', index=False, header=False)

print(f"Optimization complete!")
print(f"Original master size: {original_len}")
print(f"Final completely unique master size: {final_len}")
print(f"Total net new rows added to Brain: {final_len - 25308}")
