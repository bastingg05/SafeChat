import os
import requests
from tqdm import tqdm

MODEL_DIR = "./vegam-model"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "config.json": "https://huggingface.co/kurianbenoy/vegam-whisper-medium-ml-int8/resolve/main/config.json",
    "vocabulary.txt": "https://huggingface.co/kurianbenoy/vegam-whisper-medium-ml-int8/resolve/main/vocabulary.txt",
    "model.bin": "https://huggingface.co/kurianbenoy/vegam-whisper-medium-ml-int8/resolve/main/model.bin"
}

print("Starting direct download of Vegam Medium Model...\n")

for filename, url in FILES.items():
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        # Check if the file is completely downloaded (larger than 100MB for the model.bin)
        file_size = os.path.getsize(filepath)
        if filename == "model.bin" and file_size < 100000000:
            print(f"{filename} is corrupted/incomplete. Redownloading...")
        else:
            print(f"{filename} already exists and looks complete. Skipping.")
            continue
        
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
print("\nDownload Complete! The model is saved in ./vegam-model")
