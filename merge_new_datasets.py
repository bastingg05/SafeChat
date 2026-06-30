import pandas as pd
import os

# Load original dataset
master_file = 'mal_full_offensive_train.csv'
df_master = pd.read_csv(master_file, sep='\t', header=None, names=['text', 'label', 'extra'], on_bad_lines='skip')
original_count = len(df_master)

new_data = []

# YouTube Training Data
yt_train = 'data_exploration/Offensive_language_Malayalam-main/Malayalam_offensive_data_Training-YT (1).csv'
if os.path.exists(yt_train):
    df_yt = pd.read_csv(yt_train)
    for index, row in df_yt.iterrows():
        text = str(row['Tweets']).replace('\t', ' ').replace('\n', ' ')
        label = row['Labels']
        mapped_label = 'Not_offensive' if label == 'NOT' else 'Offensive_Untargetede'
        new_data.append({'text': text, 'label': mapped_label, 'extra': 'youtube_train'})

# YouTube Test Data
yt_test = 'data_exploration/Offensive_language_Malayalam-main/final_test_mal-offensive-with-labels (1).csv'
if os.path.exists(yt_test):
    df_yt_test = pd.read_csv(yt_test)
    for index, row in df_yt_test.iterrows():
        text = str(row['Tweets']).replace('\t', ' ').replace('\n', ' ')
        label = row['Labels']
        mapped_label = 'Not_offensive' if label == 'NOT' else 'Offensive_Untargetede'
        new_data.append({'text': text, 'label': mapped_label, 'extra': 'youtube_test'})

# Combine and drop duplicates
df_new = pd.DataFrame(new_data)
df_combined = pd.concat([df_master, df_new]).drop_duplicates(subset=['text'])
new_count = len(df_combined)

# Save back to master
df_combined.to_csv(master_file, sep='\t', index=False, header=False)

print(f"Original dataset size: {original_count}")
print(f"New combined dataset size: {new_count}")
print(f"Successfully added {new_count - original_count} new unique records.")
