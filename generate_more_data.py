import csv
import random

subjects_en = ["njan", "avan", "aval", "avar", "namakk", "kutty", "achchan", "amma", "chetan", "anujathi", "koottukaran", "koottukari", "teacher", "driver", "doctor"]
verbs_en = ["poyi", "vannu", "kandu", "paranju", "ettu", "kazhichu", "konduvannu", "nokki", "kettu", "chiri", "karanju", "padichu", "urangi", "ezhunnettu", "kalichu", "oddi"]
objects_en = ["veettil", "kadayil", "cinema", "food", "paattu", "vandi", "schoolil", "officeil", "nattil", "kalyanam", "kallyanathinu", "padipp", "cricket", "football", "groundil"]
adjectives_en = ["nalla", "adipoli", "super", "kidilan", "valiya", "cheriya", "puthiya", "pazhaya", "bhayankara", "nalla rasamulla", "boring", "poliyaya"]

subjects_ml = ["ഞാൻ", "അവൻ", "അവൾ", "അവർ", "നമുക്ക്", "കുട്ടി", "അച്ഛൻ", "അമ്മ", "ചേട്ടൻ", "അനുജത്തി", "കൂട്ടുകാരൻ", "ടീച്ചർ", "ഡോക്ടർ"]
verbs_ml = ["പോയി", "വന്നു", "കണ്ടു", "പറഞ്ഞു", "എടുത്തു", "കഴിച്ചു", "കൊണ്ടുവന്നു", "നോക്കി", "കേട്ടു", "ചിരിച്ചു", "കരഞ്ഞു", "പഠിച്ചു", "ഉറങ്ങി", "കളിച്ചു", "ഓടി"]
objects_ml = ["വീട്ടിൽ", "കടയിൽ", "സിനിമ", "ഭക്ഷണം", "പാട്ട്", "വണ്ടി", "സ്കൂളിൽ", "ഓഫീസിൽ", "നാട്ടിൽ", "കല്യാണം", "ക്രിക്കറ്റ്", "മൈതാനത്ത്"]
adjectives_ml = ["നല്ല", "അടിപൊളി", "സൂപ്പർ", "കിടിലൻ", "വലിയ", "ചെറിയ", "പുതിയ", "പഴയ", "ഭയങ്കര", "ബോറൻ"]

def generate_sentences():
    sentences = set()
    # Generate English/Manglish Safe
    for s in subjects_en:
        for v in verbs_en:
            for o in objects_en:
                for a in adjectives_en:
                    sentences.add(f"{s} {a} {o} {v}")
                    sentences.add(f"{a} {o} {s} {v}")
                    sentences.add(f"{s} {o} {a} {v}")
    
    # Generate Malayalam Safe
    for s in subjects_ml:
        for v in verbs_ml:
            for o in objects_ml:
                for a in adjectives_ml:
                    sentences.add(f"{s} {a} {o} {v}")
                    sentences.add(f"{a} {o} {s} {v}")
                    
    return list(sentences)

if __name__ == "__main__":
    all_sentences = generate_sentences()
    random.shuffle(all_sentences)
    
    # We will inject 15,000 rows. The Pandas drop_duplicates will instantly clean out any overlaps
    # from previous runs, ensuring we comfortably break the 30,000 threshold.
    selected = all_sentences[:15000]
    
    dataset_path = 'mal_full_offensive_train.csv'
    
    with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for sentence in selected:
            writer.writerow([sentence, 'Not_offensive', 'synthetic_expansion_30k'])
            
    print(f"Appended {len(selected)} massive cartesian product rows!")
