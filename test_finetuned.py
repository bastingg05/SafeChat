import sys
from transformers import pipeline

print("Loading FINE-TUNED model... Please wait.")
# Load the fine-tuned model from the local directory instead of HuggingFace hub
local_model_dir = "./finetuned_model"
classifier = pipeline("text-classification", model=local_model_dir)
print("Fine-tuned model loaded successfully!")

while True:
    try:
        user_input = input("\nEnter text to test with the FINE-TUNED model (or type 'exit' to stop): ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        if not user_input:
            continue
            
        # Get all prediction scores
        raw_results = classifier(user_input, top_k=None)

        if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], list):
            results = raw_results[0]
        else:
            results = raw_results

        # Define the safe labels based on the dataset
        safe_labels = ['Not_offensive', 'Not_in_intended_language']

        # Calculate the total probability that the text is safe
        safe_score = sum(r['score'] for r in results if r['label'] in safe_labels)
        offensive_score = 1.0 - safe_score # The rest is offensive

        # Inverted Logic: If safe score is dominant (>50%), it's safe. Otherwise unsafe.
        is_safe = safe_score > 0.5
        confidence = (safe_score if is_safe else offensive_score) * 100

        top_result = max(results, key=lambda x: x['score'])
        raw_label = top_result['label']

        print(f"\n--- Results for: '{user_input}' ---")
        print(f"Top predicted raw label: {raw_label}")
        print(f"Aggregated Offensive Score: {offensive_score:.4f}")
        print(f"Aggregated Safe Score: {safe_score:.4f}")
        
        if not is_safe:
            print(f"Final Conclusion: [TOXIC / Offensive] ({confidence:.2f}%)")
        else:
            print(f"Final Conclusion: [SAFE / Not Offensive] ({confidence:.2f}%)")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
