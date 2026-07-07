import subprocess
import sys

print("==================================================")
print("🧠 STARTING BRAIN EXPORT & AI RE-TRAINING 🧠")
print("==================================================")

try:
    print("\n[1/2] Updating dataset splits...")
    subprocess.run([sys.executable, "create_holdout.py"], check=True)
    
    print("\n[2/2] Baking new memory into AI weights (This may take a minute)...")
    subprocess.run([sys.executable, "train_model.py"], check=True)
    
    print("\n✅ Brain successfully exported! The AI is now permanently smarter.")
except Exception as e:
    print(f"\n❌ Error during Brain Export: {e}")
