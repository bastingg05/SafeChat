import os
import shutil
import subprocess

print("========================================")
print("INITIATING AI BRAIN EXPORT & UPDATE")
print("========================================")

model_dir = "./finetuned_model"
backup_dir = "./finetuned_model_backup"

if os.path.exists(model_dir):
    print("1. Creating backup of the current Brain...")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(model_dir, backup_dir)
    print("   [+] Backup secured in ./finetuned_model_backup")
else:
    print("1. No existing Brain found. Will train a fresh one.")

print("\n2. Retraining the Brain with the latest Memory (Dataset)...")
print("   (This may take 10-20 minutes depending on your hardware)")

try:
    # Run the training script directly
    subprocess.run(["python", "train_model.py"], check=True)
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    print("\n   [+] Brain successfully updated and exported to ./finetuned_model!")
except subprocess.CalledProcessError:
    print("\n   [!] ERROR: Brain update failed! Restoring from backup...")
    if os.path.exists(backup_dir):
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        shutil.copytree(backup_dir, model_dir)
        print("   [+] Restoration complete. Previous Brain is safe.")
    exit(1)

print("\n========================================")
print("BRAIN EXPORT COMPLETE.")
print("========================================")
