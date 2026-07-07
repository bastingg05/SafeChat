"""
preprocessing.py
----------------
Shared preprocessing pipeline used by both chat_server.py and evaluate scripts.
Any changes here automatically apply to ALL evaluation and inference code.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from indic_transliteration import sanscript

# ─────────────────────────────────────────────
# Slur vocabulary for cosine similarity matching
# ─────────────────────────────────────────────
BASE_SLURS = [
    "myr", "myre", "thendi", "poori", "thayoli", "panni", "punda", "pundi",
    "kundan", "kunna", "kundi", "patti", "kazhuveri", "nayinte",
    "kazhutha", "naari", "themaradi", "andi", "vaanam",
    "തെണ്ടി", "മൈര്", "പൂറി", "തായോളി", "കുണ്ടൻ", "പന്നി",
    "നാറി", "വെടി", "കഴുവേറി", "തെമ്മാടി", "പട്ടി", "കഴുത", "വാണം"
]

SAFE_EXCEPTIONS = {
    "kashuvandi", "cashew", "apple", "pooryum",
    "lab", "wifi", "homework", "project", "exam", "class", "photo"
}

_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
_vectorizer.fit(BASE_SLURS)
_base_vectors = _vectorizer.transform(BASE_SLURS)


def apply_cosine_similarity(text, threshold=0.85):
    words = text.split()
    processed_words = []
    for word in words:
        clean_word = word.lower()
        clean_word = re.sub(r'(.)\1{2,}', r'\1\1', clean_word)
        if clean_word in SAFE_EXCEPTIONS:
            processed_words.append(word)
            continue
        word_vector = _vectorizer.transform([clean_word])
        similarities = cosine_similarity(word_vector, _base_vectors)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        if best_score >= threshold:
            processed_words.append(BASE_SLURS[best_idx])
        else:
            processed_words.append(word)
    return " ".join(processed_words)


def preprocess_text(text):
    # Convert native Malayalam script to English letters (Manglish) using OPTITRANS
    text = sanscript.transliterate(text, sanscript.MALAYALAM, sanscript.OPTITRANS)
    
    # ── Smart Name Protection ──
    # If a word starts with a Capital letter, we check if its lowercase version is a known slur.
    # If it is a known slur (like "Thendi"), we do NOT protect it.
    # If it is not a known slur (like "Cinta"), we protect it by replacing it with 'friend'.
    words = text.split()
    new_words = []
    for w in words:
        if w and w[0].isupper() and w.isalpha():
            if w.lower() not in BASE_SLURS:
                new_words.append('friend')
                continue
        new_words.append(w)
    text = " ".join(new_words)

    text = text.lower()
    
    # ── Explicit Disambiguation Rules ──
    text = re.sub(r'\bchettan\b', 'brother', text)
    text = re.sub(r'\bchettaa\b', 'brother', text)
    text = re.sub(r'\bchetta\b', 'brother', text)
    text = re.sub(r'\bchette\b', 'thendi', text)
    text = re.sub(r'\bmaitanti\b', 'mythandi', text)
    text = re.sub(r'\beta\b', 'eda', text)

    # ── Neutralize casual slang bias ──
    text = re.sub(r'\bpoda\b', 'friend', text)
    text = re.sub(r'\bpodey\b', 'friend', text)
    text = re.sub(r'\bpodi\b', 'friend', text)

    # ── Protect "poo" (flower) and related words ──
    text = re.sub(r'\bpoo(?:kal)?\b', 'flower', text)
    text = re.sub(r'\blotus\b', 'flower', text)
    text = re.sub(r'\bthamara\b', 'flower', text)

    # ── Protect food poori ──
    text = re.sub(
        r'\bpoor(?:i|yum)\s+(?:and\s+)?(?:curry|bhaji|masala|undakan|kazhikan|kazhikam|ishtamano|chood)(?:um)?\b',
        'food', text
    )

    # ── Comparison structure: "X pole undu/aanu" ──
    # Keep the slur word but remove the comparison wrapper so cosine similarity can catch it
    text = re.sub(r'(\w+)\s+pole\s+(?:undu|aanu|irikkum|kaanunnu)', r'\1 insult', text)

    # ── Neutralize common false-positive school/tech words ──
    text = re.sub(r'\blab\b', 'laboratory', text)
    text = re.sub(r'\bwifi\b', 'internet', text)
    text = re.sub(r'\bhomework\b', 'assignment_work', text)
    text = re.sub(r'\bproject\b', 'coursework', text)

    # ── Globally neutralize movement verbs to prevent false alarms ──
    text = re.sub(r'\bpoyi\b', 'went', text)   # went / left
    text = re.sub(r'\baayee\b', 'arrived', text)  # came / arrived
    text = re.sub(r'\bvanna\b', 'came', text)     # came

    # ── Neutralize over-fitted pronouns/words ──
    text = re.sub(r'\bfriend(?:s)?\b', 'friend', text)
    text = re.sub(r'\bninte\b', 'friend', text)
    text = re.sub(r'\bninak(?:k)?(?:oke|ku|kku|um|e|)\b', 'friend', text)  # ninakoke, ninakku, ninak
    text = re.sub(r'\bninak(?:k)?\s+oke\b', 'friend', text) # ninak oke (with space)
    text = re.sub(r'\benthinte\b', 'friend', text)
    text = re.sub(r'\bda\b', 'friend', text)
    text = re.sub(r'\bmari\b', 'friend', text)
    text = re.sub(r'\bangot\b', 'friend', text)
    text = re.sub(r'\basugam\b', 'friend', text)
    text = re.sub(r'\bmaryada(?:k|kku)?\b', 'friend', text)

    # ── Cosine similarity catch-all ──
    text = apply_cosine_similarity(text)
    return text
