---
language: 
  - ml
  - en
tags:
  - hate-speech-detection
  - malayalam
  - manglish
  - muril
  - text-classification
license: apache-2.0
---

# SafeChat: Manglish & Malayalam Hate Speech Detection Model

### Model Description
This is a fine-tuned **MuRIL (Multilingual Representations for Indian Languages)** model designed specifically for real-time hate speech, cyberbullying, and profanity detection in **Manglish** (Malayalam written in English script) and **Native Malayalam** script. 

Standard NLP models struggle with the unique phonetic spelling variations and context-heavy nature of Manglish. This model has been aggressively fine-tuned on a custom dataset of 39,716 unique records to understand the deep semantic context of Malayalam internet slang. It goes beyond binary "Safe/Unsafe" classification by categorizing text into nuanced safety buckets.

### Intended Use
This model is intended to be used as the core inference engine for automated moderation systems, chat applications, and social media filters targeting Malayalam-speaking demographics.

### Model Details
* **Base Architecture:** `google/muril-base-cased`
* **Language(s):** Manglish, Malayalam, English
* **Fine-Tuning Dataset:** 40,000+ custom-engineered code-mixed records.
* **Epochs:** 1
* **Task:** Multi-class Sequence Classification

### Output Classes (5 Buckets)
The model classifies text into the following categories:
1. `Not_offensive`: Normal, safe conversation.
2. `Profanity`: General vulgarity and swear words without a specific target.
3. `Off_target_ind`: Hate speech or abuse targeted at a specific individual (Personal Attack).
4. `Off_target_group`: Hate speech targeted at a specific group (Religious, Political, Gender-based).
5. `Not_in_intended_language`: Text outside the domain of Malayalam/Manglish.

### Evaluation Metrics
On a strictly unseen holdout dataset, the model achieved:
* **Validation Accuracy:** > 94%
* **Precision:** 95.1%
* **Recall:** 92.8%
* **F1-Score:** 0.93

### Required Preprocessing Pipeline
**⚠️ CRITICAL:** This model was trained in tandem with a mathematical pre-processing firewall. For optimal results, production environments *must* implement the following before passing text to the model:
1. **Transliteration:** Convert native Malayalam script to ASCII Manglish (via `indic-transliteration`).
2. **Regex Disambiguation:** Neutralize harmless casual slang (e.g., mapping `da` to `friend`) and isolate semantic contexts (e.g., mapping `poori` + `curry` to `food` to prevent false positives).
3. **Cosine Similarity Spell-Check:** Use TF-IDF character n-grams to catch obfuscated spellings of slurs before classification.
