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

try:
    import jellyfish
except ImportError:
    jellyfish = None

# ─────────────────────────────────────────────
# Slur vocabulary for cosine similarity matching
# ─────────────────────────────────────────────
BASE_SLURS = [
    "myr", "myre", "thendi", "poori", "thayoli", "panni", "punda", "pundi",
    "kundan", "kunna", "kundi", "patti", "kazhuveri", "nayinte", "naya",
    "kazhutha", "naari", "themaradi", "andi", "vaanam",
    "polayadi", "polayadimwone", "naye", "maramakri", "mayir", "mayire",
    "poor", "koothi", "funda", "pary", "kaziveri", "andikkannan", "moron",
    "തെണ്ടി", "മൈര്", "പൂറി", "തായോളി", "കുണ്ടൻ", "പന്നി",
    "നാറി", "വെടി", "കഴുവേറി", "തെമ്മാടി", "പട്ടി", "കഴുത", "വാണം"
]

SAFE_EXCEPTIONS = {
    "kashuvandi", "cashew", "apple", "pooryum",
    "lab", "wifi", "homework", "project", "exam", "class", "photo"
}

# Precompute TF-IDF vectors for slurs
_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
_base_vectors = _vectorizer.fit_transform(BASE_SLURS)

# Precompute Phonetic Soundexes for slurs
if jellyfish:
    SLUR_SOUNDEXES = {jellyfish.soundex(s) for s in BASE_SLURS if s.isascii()}
else:
    SLUR_SOUNDEXES = set()

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
            # Suffix Stripper Fallback: Strip common Malayalam suffixes and re-evaluate
            suffixes = ["yod", "yode", "kku", "ude", "yude", "ne", "aye", "iye", "mar", "kal", "k"]
            stripped_match = False
            for suf in suffixes:
                if clean_word.endswith(suf) and len(clean_word) > len(suf) + 2:
                    root = clean_word[:-len(suf)]
                    # Check exact match first
                    if root in BASE_SLURS:
                        processed_words.append(root)
                        stripped_match = True
                        break
                    # If not exact, check cosine of the root
                    root_vec = _vectorizer.transform([root])
                    root_sims = cosine_similarity(root_vec, _base_vectors)[0]
                    root_best_idx = np.argmax(root_sims)
                    if root_sims[root_best_idx] >= threshold:
                        processed_words.append(BASE_SLURS[root_best_idx])
                        stripped_match = True
                        break
            if not stripped_match and jellyfish:
                # Phonetic Fallback: If math missed it closely, check if it sounds exactly like a slur
                sx = jellyfish.soundex(clean_word)
                if sx in SLUR_SOUNDEXES:
                    # Find the first base slur that matches this soundex and replace it
                    for s in BASE_SLURS:
                        if jellyfish.soundex(s) == sx:
                            processed_words.append(s)
                            stripped_match = True
                            break
                            
            if not stripped_match:
                processed_words.append(word)
    return " ".join(processed_words)


def preprocess_text(text):
    # Convert native Malayalam script to English letters (Manglish) using OPTITRANS
    text = sanscript.transliterate(text, sanscript.MALAYALAM, sanscript.OPTITRANS)
    
    text = text.lower()
    
    # ── Explicit Disambiguation Rules ──
    text = re.sub(r'\bchettan\b', 'brother', text)
    text = re.sub(r'\bchettaa\b', 'brother', text)
    text = re.sub(r'\bchetta\b', 'brother', text)
    text = re.sub(r'\bchette\b', 'thendi', text)
    text = re.sub(r'\bmaitanti\b', 'mythandi', text)
    text = re.sub(r'\beta\b', 'eda', text)
    text = re.sub(r'\bnjan\b', 'me', text)
    text = re.sub(r'\bkanune+\b', 'seeing', text)

    # ── Protect "eda / edaa" — casual Malayalam address term (like 'hey') ──
    # It is commonly used as a friendly address word, not an insult.
    text = re.sub(r'\beda+\b', 'friend', text)
    # ── Always protect "pattikutti" (puppy) ──
    # A puppy is almost never a slur, so we blanket-replace it without needing context words
    text = re.sub(r'\bpatti(?:kutti|kuttiye|kuttikku|kuttikal)\b', 'dog', text)

    # ── Protect "patti" when used as "dog" in location/neutral context ──
    # Handles Malayalam case suffixes: pattine, pattikku, pattiye, pattikal etc.
    # e.g. "patti avide kidakkunnu" / "pattine kannan" = safe
    PATTI_FORMS = r'\bpatti(?:ne|kku|ye|kal|ude|yude|yku|)\b'
    PATTI_SAFE_WORDS = r'(?:avide|ividde|athu|ithu|und|undu|kidappund|kidakkunnu|vannu|poyi|odum|nottu|kanikunee|kanikune|kanikkam|kanikkan|cute|kollila|kollam|sadanam|enthu|entha|evidey|kand|kanda|kandal|kanan|vishyam|pwoli|ahnalo|ahnnnn|ahn|aan|aanu)'
    text = re.sub(
        PATTI_FORMS + r'\s+' + PATTI_SAFE_WORDS,
        'dog', text
    )
    text = re.sub(
        PATTI_SAFE_WORDS + r'\s+' + PATTI_FORMS,
        'dog', text
    )
    # Prefix articles: oru/aa/ente/ninte/e/i + patti
    text = re.sub(
        r'(?:oru|aa|ente|ninte|e|i)\s+' + PATTI_FORMS,
        'dog', text
    )

    # ── Strip Intensifier Prefixes (para, perum, etc.) from Slurs ──
    # e.g., paranari -> nari, perummyre -> myre, danmayire -> mayire
    words = text.split()
    for i, w in enumerate(words):
        for prefix in ['para', 'perum', 'maha', 'verum', 'dan', 'da', 'eda', 'poda']:
            if w.startswith(prefix) and len(w) > len(prefix) + 2:
                root = w[len(prefix):]
                root_norm = re.sub(r'(.)\1+', r'\1', root)
                
                # Check if root is in BASE_SLURS directly, or via soundex/cosine
                if root in BASE_SLURS or root_norm in SLURS_NORMALIZED:
                    words[i] = root
                    break
                
                # Also run cosine similarity on the stripped root to be safe
                root_vec = _vectorizer.transform([root])
                root_sims = cosine_similarity(root_vec, _base_vectors)[0]
                if np.max(root_sims) >= 0.85:
                    words[i] = BASE_SLURS[np.argmax(root_sims)]
                    break
                    
    text = " ".join(words)


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

    # ── Explicit slur additions to prevent bypasses ──
    text = re.sub(r'\bulle\b', 'myre', text)      # Maps "ulle" directly to a strong known slur
    
    # Check if common prefixes are used immediately before a slur word, and if so, remove them
    prefixes_to_strip = {"da", "poda", "ninte", "ne"}
    words = text.split()
    for i in range(len(words) - 1):
        if words[i] in prefixes_to_strip and words[i+1] in BASE_SLURS:
            words[i] = ""
    text = " ".join([w for w in words if w])

    # ── Neutralize over-fitted pronouns/words ──
    text = re.sub(r'\bfriend(?:s)?\b', 'friend', text)
    text = re.sub(r'\bninte\b', 'friend', text)
    text = re.sub(r'\bninak(?:k)?(?:oke|ku|kku|um|e|)\b', 'friend', text)  # ninakoke, ninakku, ninak
    text = re.sub(r'\bninak(?:k)?\s+oke\b', 'friend', text) # ninak oke (with space)
    text = re.sub(r'\bninak(?:k)?\s+oke\b', 'friend', text) # ninak oke (with space)
    
    # Female pronouns (often tied to misogynistic bias in training data)
    text = re.sub(r'\baval\b', 'friend', text)
    text = re.sub(r'\bavalude\b', 'friend', text)
    text = re.sub(r'\bavalk\b', 'friend', text)
    text = re.sub(r'\bavalkk\b', 'friend', text)
    text = re.sub(r'\bavale\b', 'friend', text)
    text = re.sub(r'\bavalodu\b', 'friend', text)
    text = re.sub(r'\bival\b', 'friend', text)
    text = re.sub(r'\bivalude\b', 'friend', text)
    text = re.sub(r'\bivalk\b', 'friend', text)
    text = re.sub(r'\bivalkk\b', 'friend', text)
    text = re.sub(r'\bivale\b', 'friend', text)
    text = re.sub(r'\bivalodu\b', 'friend', text)
    
    # Group pronouns (often tied to communal bias)
    text = re.sub(r'\bavar\b', 'friend', text)
    text = re.sub(r'\bavarude\b', 'friend', text)
    text = re.sub(r'\bavark\b', 'friend', text)
    text = re.sub(r'\bavarkk\b', 'friend', text)
    text = re.sub(r'\bavare\b', 'friend', text)
    text = re.sub(r'\bavaroke\b', 'friend', text)
    
    # Male pronouns (often tied to targeted bullying bias)
    text = re.sub(r'\bavan\b', 'friend', text)
    text = re.sub(r'\bavante\b', 'friend', text)
    text = re.sub(r'\bavane\b', 'friend', text)
    text = re.sub(r'\bavank\b', 'friend', text)
    text = re.sub(r'\bivan\b', 'friend', text)
    text = re.sub(r'\bivante\b', 'friend', text)
    text = re.sub(r'\bivane\b', 'friend', text)
    
    # Other over-fitted words
    text = re.sub(r'\benthinte\b', 'friend', text)
    text = re.sub(r'\bda\b', 'friend', text)
    text = re.sub(r'\bmari\b', 'friend', text)
    text = re.sub(r'\bangot\b', 'friend', text)
    text = re.sub(r'\basugam\b', 'friend', text)
    text = re.sub(r'\bmaryada(?:k|kku)?\b', 'friend', text)

    # ── Neutralize affectionate terms (prevent false positives) ──
    text = re.sub(r'\bkunje\b', 'friend', text)
    text = re.sub(r'\bkunju\b', 'friend', text)
    text = re.sub(r'\bmone\b', 'friend', text)
    text = re.sub(r'\bmuthe\b', 'friend', text)
    text = re.sub(r'\bkutta\b', 'friend', text)
    text = re.sub(r'\bmakane\b', 'friend', text)

    # ── Cosine similarity catch-all ──
    text = apply_cosine_similarity(text)
    return text
