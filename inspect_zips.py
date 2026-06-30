import zipfile
import os

zips = [
    'DravidianCodeMix-2020.zip',
    'Dialect_Based_Speech_Recognition-20260624T174052Z-3-001.zip',
    'Train-20260624T174236Z-3-001.zip',
    'nlp-for-manglish-master.zip',
    'Offensive_language_Malayalam-main.zip'
]

for z in zips:
    print(f"\n=== Contents of {z} ===")
    try:
        with zipfile.ZipFile(z, 'r') as zip_ref:
            for f in zip_ref.infolist():
                if not f.is_dir():
                    print(f.filename)
    except Exception as e:
        print(f"Error reading {z}: {e}")
