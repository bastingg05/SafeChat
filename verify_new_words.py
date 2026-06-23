import os
import sys
from transformers import pipeline

# Adjust console encoding for Windows if necessary
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("Loading fine-tuned model from ./finetuned_model...")
try:
    classifier = pipeline("text-classification", model="./finetuned_model")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

test_cases = [
    # 1. പന്നി (Panni)
    {"text": "eda panni mone", "expected": "Offensive"},
    {"text": "kattu panni krishi motham nashippichu", "expected": "Safe"},
    {"text": "പോടാ പട്ടി മോനെ", "expected": "Offensive"},
    {"text": "കാട്ടു പന്നി കൃഷി മൊത്തം നശിപ്പിച്ചു", "expected": "Safe"},
    
    # 2. കോപ്പ് (Kopp/Koppe)
    {"text": "enthu koppa ithu", "expected": "Offensive"},
    {"text": "sadhanakoppukal ellam thayaaraanu", "expected": "Safe"},
    {"text": "എന്ത് കോപ്പ് ആണ് ഇത്", "expected": "Offensive"},
    {"text": "നല്ല സാധനക്കോപ്പുകൾ ഉണ്ടല്ലോ ഇവിടെ", "expected": "Safe"},
    
    # 3. മലര് (Malar/Malare)
    {"text": "poda malare", "expected": "Offensive"},
    {"text": "ambalathil malar vazhipadu nadatheenam", "expected": "Safe"},
    {"text": "പോടാ മലരേ", "expected": "Offensive"},
    {"text": "അമ്പലത്തിൽ മലർ വഴിപാട് നടത്തണം", "expected": "Safe"},
    
    # 4. കഴുത (Kazhutha)
    {"text": "nee oru kazhutha aanu", "expected": "Offensive"},
    {"text": "kazhutha chumadu chummakkunnathu kando", "expected": "Safe"},
    {"text": "നീ ഒരു കഴുത ആണ്", "expected": "Offensive"},
    {"text": "കഴുത ചുമട് ചുമക്കുന്നത് കണ്ടോ", "expected": "Safe"},
    
    # 5. ചെറ്റ (Chetta)
    {"text": "avan verum chetta aanu", "expected": "Offensive"},
    {"text": "veetinu purathekku chetta veli ketti", "expected": "Safe"},
    {"text": "അവൻ വെറും ചെറ്റ ആണ്", "expected": "Offensive"},
    {"text": "വീടിന് പുറത്തേക്ക് ചെറ്റ വേലി കെട്ടി", "expected": "Safe"},
    
    # 6. കുരിശ് (Kurish/Kurishe)
    {"text": "enikk ee kurish venda", "expected": "Offensive"},
    {"text": "achayan kurishu varachu prarthikkuvanu", "expected": "Safe"},
    {"text": "എനിക്ക് ഈ കുരിശ് വേണ്ട", "expected": "Offensive"},
    {"text": "അച്ചായൻ കുരിശ് വരച്ചു പ്രാർത്ഥിക്കുകയാണ്", "expected": "Safe"},
    
    # 7. പുല്ല് (Pullu)
    {"text": "poda pullu", "expected": "Offensive"},
    {"text": "njan parambil poyi pashuvingulla pullu parichu", "expected": "Safe"},
    {"text": "പോടാ പുല്ല്", "expected": "Offensive"},
    {"text": "ഞാൻ പറമ്പിൽ പോയി പശുവിനുള്ള പുല്ല് പറിച്ചു", "expected": "Safe"},

    # 8. തെണ്ടി (Thendi - specifically requested by user)
    {"text": "da thendi", "expected": "Offensive"},
    {"text": "ഡാ തെണ്ടി", "expected": "Offensive"}
]

print("\nRunning verification tests...")
print(f"{'Text':<55} | {'Expected':<10} | {'Predicted':<10} | {'Off. Score':<10} | {'Safe Score':<10} | {'Status':<10}")
print("-" * 115)

safe_labels = ['Not_offensive', 'Not_in_intended_language']
passed = 0

for case in test_cases:
    text = case["text"]
    raw_results = classifier(text, top_k=None)
    
    # Handle list formats
    if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], list):
        results = raw_results[0]
    else:
        results = raw_results
        
    safe_score = sum(r['score'] for r in results if r['label'] in safe_labels)
    offensive_score = 1.0 - safe_score
    
    is_safe = safe_score > 0.5
    predicted = "Safe" if is_safe else "Offensive"
    
    status = "PASS" if predicted == case["expected"] else "FAIL"
    if status == "PASS":
        passed += 1
        
    print(f"{text:<55} | {case['expected']:<10} | {predicted:<10} | {offensive_score:.4f}     | {safe_score:.4f}     | {status:<10}")

print("-" * 115)
print(f"Passed {passed}/{len(test_cases)} tests.")
