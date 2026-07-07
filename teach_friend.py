import pandas as pd
import random

CSV_FILE = "mal_full_offensive_train.csv"

SAFE_TEMPLATES = [
    "friend",
    "Friend",
    "friends",
    "Friends",
    "my friend",
    "hello friend",
    "namaskaram friend",
    "eda friend",
    "friend engane undu",
    "nalla friend aanu",
    "friend aano",
    "friend aanu",
    "my friends",
    "good friend",
    "best friend"
]

def generate_friend_samples():
    new_rows = []
    
    # Generate 500 examples of the word 'friend' in safe contexts
    for _ in range(500):
        text = random.choice(SAFE_TEMPLATES)
        if random.random() > 0.5:
            text = text.upper() if random.random() > 0.8 else text.title()
            
        new_rows.append({"text": text, "label": "Not_offensive", "extra": "synthetic_friend_safe"})
        
    return pd.DataFrame(new_rows)

if __name__ == "__main__":
    print("Teaching AI that 'friend' is a safe word...")
    df_new = generate_friend_samples()
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    df_new.to_csv(CSV_FILE, mode='a', header=False, index=False, sep='\t')
    print(f"Successfully appended {len(df_new)} safe examples of 'friend' to {CSV_FILE}.")
