import zipfile
import os

os.makedirs('data_exploration', exist_ok=True)

targets = {
    'DravidianCodeMix-2020.zip': ['malayalam_hasoc_tanglish_test.tsv', 'malayalam_hasoc_tanglish_train.tsv', 'malayalam_train.tsv', 'malayalam_test.tsv', 'malayalam_dev.tsv'],
    'Offensive_language_Malayalam-main.zip': ['Offensive_language_Malayalam-main/Malayalam_offensive_data_Training-YT (1).csv', 'Offensive_language_Malayalam-main/final_test_mal-offensive-with-labels (1).csv']
}

for zip_name, files in targets.items():
    try:
        with zipfile.ZipFile(zip_name, 'r') as z:
            for f in files:
                z.extract(f, 'data_exploration')
                print(f"Extracted {f}")
    except Exception as e:
        print(f"Error with {zip_name}: {e}")

print("Extraction complete.")
