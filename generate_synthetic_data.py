import csv
import random

# Mapping of bad words from Manglish (Latin script) to Native Malayalam script
bad_words_map = {
    'patti': 'പട്ടി',
    'myr': 'മൈര്',
    'myre': 'മൈരേ',
    'maire': 'മൈരേ',
    'poda': 'പോടാ',
    'thendi': 'തെണ്ടി',
    'kundan': 'കുണ്ടൻ',
    'thayoli': 'തായോളി',
    'poori': 'പൂറി',
    'naari': 'നാറി',
    'vedi': 'വെടി',
    'vedi poori': 'വെടിപ്പൂറി',
    'kopp': 'കോപ്പ്',
    'koppe': 'കോപ്പേ',
    'panni': 'പന്നി',
    'malar': 'മലര്',
    'malare': 'മലരേ',
    'kazhutha': 'കഴുത',
    'chetta': 'ചെറ്റ',
    'kurish': 'കുരിശ്',
    'kurishe': 'കുരിശേ',
    'pullu': 'പുല്ല്'
}

# Contextual templates for OFFENSIVE (Manglish)
offensive_templates_en = [
    "eda {word}",
    "nee oru {word} aanu",
    "{word} mone",
    "poda {word}",
    "naari {word}",
    "ninne njan kaanichu tharaam {word}",
    "ivan oru {word}",
    "enthu {word} aanu ithu",
    "ninte thala {word}",
    "poyi chathu kude {word}",
    "verum {word} abhinayam",
    "valiya {word} aavan nokkunnu",
    "{word} poyi oomb",
    "avan verum {word} aanu",
    "verum {word} tharam kanikkathe"
]

# Contextual templates for OFFENSIVE (Native Malayalam)
offensive_templates_ml = [
    "എടാ {word}",
    "നീ ഒരു {word} ആണ്",
    "{word} മോനെ",
    "പോടാ {word}",
    "നാറി {word}",
    "നിന്നെ ഞാൻ കാണിച്ചു തരാം {word}",
    "ഇവൻ ഒരു {word}",
    "എന്ത് {word} ആണ് ഇത്",
    "നിന്റെ തല {word}",
    "പോയി ചത്തു കൂടെ {word}",
    "വെറും {word} അഭിനയം",
    "വലിയ {word} ആവാൻ നോക്കുന്നു",
    "{word} പോയി ഊമ്പ്",
    "അവൻ വെറും {word} ആണ്",
    "വെറും {word}ത്തരം കാണിക്കാതെ"
]

# Word-specific offensive phrases (Manglish)
word_specific_offensive_en = {
    'kurish': [
        "enikk ee kurish venda",
        "ee kurish thalayil aayi",
        "enthina ee kurish varuthivakkunnathu",
        "ente thalayil ee kurish vechuthannu",
        "enthu kurish aanu ithu thalayil varuthiyathu"
    ],
    'kurishe': [
        "enikk ee kurishe venda",
        "ee kurishe thalayil aayi",
        "enthina ee kurishe varuthivakkunnathu",
        "ente thalayil ee kurishe vechuthannu"
    ],
    'chetta': [
        "avan verum chetta aanu",
        "ijjazhi chetta tharam kanikkuthu",
        "chetta sambhashanam parayathe poda",
        "verum chetta manobhavam",
        "chetta character aanu ivan"
    ],
    'pullu': [
        "enikk ee pullu venda",
        "ee pullu cinema aaru kaanana",
        "enthu pullu aanu ithu",
        "pullu, njan thottu",
        "ee pullu ivide ninnum poyikude"
    ],
    'kopp': [
        "enikk ee kopp venda",
        "enthu kopp aanu ithu",
        "ee kopp ivide ninnum poyikude"
    ],
    'koppe': [
        "enikk ee koppe venda",
        "enthu koppe aanu ithu",
        "ee koppe ivide ninnum poyikude"
    ]
}

# Word-specific offensive phrases (Native Malayalam)
word_specific_offensive_ml = {
    'കുരിശ്': [
        "എനിക്ക് ഈ കുരിശ് വേണ്ട",
        "ഈ കുരിശ് തലയിൽ ആയി",
        "എന്തിനാ ഈ കുരിശ് വരുത്തിവെക്കുന്നത്",
        "എന്റെ തലയിൽ ഈ കുരിശ് വെച്ചുതന്നു",
        "എന്ത് കുരിശ് ആണ് ഇത് തലയിൽ വരുത്തിയത്"
    ],
    'കുരിശേ': [
        "എനിക്ക് ഈ കുരിശേ വേണ്ട",
        "ഈ കുരിശേ തലയിൽ ആയി",
        "എന്തിനാ ഈ കുരിശേ വരുത്തിവെക്കുന്നത്",
        "എന്റെ തലയിൽ ഈ കുരിശേ വെച്ചുതന്നു"
    ],
    'ചെറ്റ': [
        "അവൻ വെറും ചെറ്റ ആണ്",
        "ഇജ്ജാതി ചെറ്റത്തരം കാണിക്കരുത്",
        "ചെറ്റ സംഭാഷണം പറയാതെ പോടാ",
        "വെറും ചെറ്റ മനോഭാവം",
        "ചെറ്റ ക്യാരക്ടർ ആണ് ഇവൻ"
    ],
    'പുല്ല്': [
        "എനിക്ക് ഈ പുല്ല് വേണ്ട",
        "ഈ പുല്ല് സിനിമ ആര് കാണാനാ",
        "എന്ത് പുല്ല് ആണ് ഇത്",
        "പുല്ല്, ഞാൻ തോറ്റു",
        "ഈ പുല്ല് ഇവിടെ നിന്നും പോയിക്കൂടെ"
    ],
    'കോപ്പ്': [
        "എനിക്ക് ഈ കോപ്പ് വേണ്ട",
        "എന്ത് കോപ്പ് ആണ് ഇത്",
        "ഈ കോപ്പ് ഇവിടെ നിന്നും പോയിക്കൂടെ"
    ],
    'കോപ്പേ': [
        "എനിക്ക് ഈ കോപ്പേ വേണ്ട",
        "എന്ത് കോപ്പേ ആണ് ഇത്",
        "ഈ കോപ്പേ ഇവിടെ നിന്നും പോയിക്കൂടെ"
    ]
}

# Contextual templates for SAFE (Manglish)
safe_templates_en = [
    "ee cinemaye patti enth parayunnu",
    "nalla abhiprayam aanu idhine patti",
    "poda aniya, nalla cinema aanu",
    "ivide varan pattiya cinema",
    "njan onnum parayunilla patti",
    "adhine patti kooduthal ariyilla",
    "athine patti valiya dharana illa",
    "nalla patti (dog) aanu",
    "poda namukk cinema kaanam",
    "enikk poori curry venam",
    "poori and curry super combinations aanu",
    "nalla choodu poori um curryum kazhichu",
    "ravile poori bhaji aanu undakkiyathu",
    "enikk poori kazhikkanam",
    "aa hotelile poori um curry um kollaam",
    "ammachi poori undakkukayanu",
    "ambalathil vedi vazhipadu nadathi",
    "vishuvinu vedi pottikkanam",
    "utsavathinu vedi kettiyathu kandu",
    "nalla shabdamulla vedi aayirunnu",
    "vedi vazhipadu neruthuka",
    "vishu kani kaanan ulla koppukal thayyaraakki",
    "nalla sadhanakoppukal undallo ivide",
    "kattu panni krishi motham nashippichu",
    "parambil panni keri marichittu poyi",
    "avide panni valarthal kendram undu",
    "poojakkulla malarum poovum tharaamo",
    "ambalathil malar vazhipadu nadatheenam",
    "kazhutha chumadu chummakkunnathu kando",
    "palliyile kurishu varakkal chadaangu thudangi",
    "achayan kurishu varachu prarthikkuvanu",
    "veetinu purathekku chetta veli ketti",
    "chettapure varoo cheriya pura aanu",
    "njan parambil poyi pashuvingulla pullu parichu",
    "ee pullu medukal kaanan kidilam aanu",
    # --- Extra safe sentences for ambiguous words ---
    # pullu = grass (literal)
    "pashu pullu thinnuu",
    "aatu pullu meayunnu",
    "pullu medu kanaan poakaam",
    "parambil pullu kond varunnu",
    "maithanam niraye pullu undu",
    "pacha pullu koyyaedukkunnu",
    "aa pullu medu valare sundaramaanu",
    "krishi bhoomiyil pullu valarnnu",
    "pashu pullu thinnu santhoshamaayi",
    "aadukal pullu thedukunnu",
    # panni = pig / farm animal (literal)
    "panni valarthal labhakaram aanu",
    "panni krishi cheyyunna karshakane kandu",
    "aa panni kuttikalo valare munthiri aanu",
    "panni irakki thinnittundu",
    "panni valarthuka oru nalla thozhilaanu",
    "ee panni farm valare valuthaanu",
    # poda = go (casual direction, not insult)
    "poda school poakam",
    "poda angott poakam nammal",
    "poda nee eni vaarumo",
    "poda, nalla cinema aanu itthu",
    # poori = Indian bread (food)
    "subhash ravile poori thinnittund",
    "ammachi undakkunna poori super aanu",
    "ee poori bhaji combinations kidilam",
    "market il poori kittum",
    # vedi = firecracker (festival)
    "vedi kettiyappol ella perum santhoshichu",
    "vedi pottikkunna shabdam dooarthe ninn kettu",
    "utsavathinu vedi veriche",
    # kopp = vessel / utensil
    "ammachi kopp kazhukunnu",
    "aa kopp eduthu veykkuu",
    "paalukal vekkan kopp venam",
    # patti = dog
    "ente patti valare cute aanu",
    "aa patti kuttikku paalu kodukkanam",
    "nalla patti thada tharavottundu",
    "ninte patti kunju evideya",
    "ente vettil oru patti undu",
    "patti valare nalla mrighamaanu",
    "patti kurakkunnundu",
    "aa patti odippoyi",
    "pattikk paalu kodukku",
    "nalla oru patti kutti aanu",
    "ninte patti kunju ethrayaayi",
    # kazhutha = donkey (animal)
    "kazhutha chumadu vahikkunnu",
    "kazhuthaye nokku ellavarum chirichchu",
    "aa kazhutha valare manassanu",
    # chetta = thatched hut / fence (object)
    "chetta veli kettiyathu nokkaan nallathanu",
    "parambil chetta pura undakkaanam",
    "cheriya chetta kude ivide kaanam",
    # malar = flower / puffed rice (ritual)
    "ambalathil malar vazhipadu cheyyanam",
    "vishuvinulla malar nannaayi phalichu",
    "malar vazhipadu ella ambalathilum und",
    # kurish = cross (religious symbol)
    "palliyile kurish varakkal chadanganu",
    "kurish varachu prarthanakal nadathi",
    "ee kurish valare paavanamaanu",
    "kristhyani aacharangalil kurish prasakthamaanu"
]

# Contextual templates for SAFE (Native Malayalam)
safe_templates_ml = [
    "ഈ സിനിമയെ പറ്റി എന്ത് പറയുന്നു",
    "നല്ല അഭിപ്രായം ആണ് ഇതിനെ പറ്റി",
    "പോടാ അനിയാ, നല്ല സിനിമ ആണ്",
    "ഇവിടെ വരാൻ പറ്റിയ സിനിമ",
    "ഞാൻ ഒന്നും പറയുന്നില്ല പറ്റി",
    "അതിനെ പറ്റി കൂടുതൽ അറിയില്ല",
    "അതിനെ പറ്റി വലിയ ധാരണ ഇല്ല",
    "നല്ല പട്ടി (നായ) ആണ്",
    "പോടാ നമുക്ക് സിനിമ കാണാം",
    "എനിക്ക് പൂരി കറി വേണം",
    "പൂരിയും കറിയും സൂപ്പർ കോമ്പിനേഷൻ ആണ്",
    "നല്ല ചൂട് പൂരിയും കറിയും കഴിച്ചു",
    "രാവിലെ പൂരി ഭാജി ആണ് ഉണ്ടാക്കിയത്",
    "എനിക്ക് പൂരി കഴിക്കാൻ വേണം",
    "ആ ഹോട്ടലിലെ പൂരിയും കറിയും കൊള്ളാം",
    "അമ്മച്ചി പൂരി ഉണ്ടാക്കുകയാണ്",
    "അമ്പലത്തിൽ വെടി വഴിപാട് നടത്തി",
    "വിഷുവിന് വെടി പൊട്ടിക്കണം",
    "ഉത്സവത്തിന് വെടി കെട്ടിയത് കണ്ടു",
    "നല്ല ശബ്ദമുള്ള വെടി ആയിരുന്നു",
    "വെടി വഴിപാട് നിർത്തുക",
    "വിഷു കണി കാണാൻ ഉള്ള കോപ്പുകൾ തയ്യാറാക്കി",
    "നല്ല സാധനക്കോപ്പുകൾ ഉണ്ടല്ലോ ഇവിടെ",
    "കാട്ടു പന്നി കൃഷി മൊത്തം നശിപ്പിച്ചു",
    "പറമ്പിൽ പന്നി കേറി മറിച്ചിട്ടു പോയി",
    "അവിടെ പന്നി വളർത്തൽ കേന്ദ്രം ഉണ്ട്",
    "പൂജക്കുള്ള മലരും പൂവും തരാമോ",
    "അമ്പലത്തിൽ മലർ വഴിപാട് നടത്തണം",
    "കഴുത ചുമട് ചുമക്കുന്നത് കണ്ടോ",
    "പള്ളിയിലെ കുരിശ് വരക്കൽ ചടങ്ങ് തുടങ്ങി",
    "അച്ചായൻ കുരിശ് വരച്ചു പ്രാർത്ഥിക്കുകയാണ്",
    "വീടിന് പുറത്തേക്ക് ചെറ്റ വേലി കെട്ടി",
    "ചെറ്റപ്പുര വരൂ ചെറിയ പുര ആണ്",
    "ഞാൻ പറമ്പിൽ പോയി പശുവിനുള്ള പുല്ല് പറിച്ചു",
    "ഈ പുല്ല് മേടുകൾ കാണാൻ കിടിലം ആണ്",
    # --- Extra safe sentences for ambiguous words ---
    # പുല്ല് = grass (literal)
    "പശു പുല്ല് തിന്നു",
    "ആട് പുല്ല് മേയുന്നു",
    "പുല്ല് മേട്ടിൽ കളിക്കാൻ പോകാം",
    "പറമ്പിൽ പോയി പുല്ല് കൊണ്ടു വന്നു",
    "മൈതാനത്തിൽ പുല്ല് ഉണ്ട്",
    "പശുവിന് പുല്ല് വെട്ടിക്കൊടുത്തു",
    "ആ പുല്ല് മേട് വളരെ പച്ചയാണ്",
    "കൃഷിഭൂമിയിൽ പുല്ല് വളർന്നു",
    "പശു പുല്ല് തിന്ന് സന്തോഷമായി",
    "ആടുകൾ പുല്ല് തേടുന്നു",
    "പുല്ല് കൊണ്ടു വന്ന് പശുവിനു കൊടുത്തു",
    "മനോഹരമായ പുല്ല് നിറഞ്ഞ വയൽ",
    # പന്നി = pig / farm animal (literal)
    "പന്നി വളർത്തൽ ലാഭകരമാണ്",
    "പന്നി കൃഷി ചെയ്യുന്ന കർഷകനെ കണ്ടു",
    "ആ പന്നി കുട്ടികൾ വളരെ മുന്തിരിയാണ്",
    "പന്നി ഇറക്കി തിന്നിട്ടുണ്ട്",
    "പന്നി വളർത്തൽ ഒരു നല്ല തൊഴിലാണ്",
    "ഈ പന്നി ഫാം വളരെ വലുതാണ്",
    # പോടാ = go (casual, not an insult when directional)
    "പോടാ സ്കൂളിൽ പോകാം",
    "പോടാ അങ്ങോട്ട് പോകാം നമ്മൾ",
    "പോടാ, നല്ല സിനിമ ആണ് ഇത്",
    # പൂരി = Indian bread (food)
    "സുഭാഷ് രാവിലെ പൂരി തിന്നിട്ടുണ്ട്",
    "അമ്മ ഉണ്ടാക്കുന്ന പൂരി സൂപ്പർ ആണ്",
    "ഈ പൂരി ഭാജി കോമ്പിനേഷൻ കിടിലം",
    "മാർക്കറ്റിൽ പൂരി കിട്ടും",
    # വെടി = firecracker (festival)
    "വെടി കെട്ടിയപ്പോൾ എല്ലാവരും സന്തോഷിച്ചു",
    "വെടി പൊട്ടിക്കുന്ന ശബ്ദം ദൂരത്ത് നിന്ന് കേട്ടു",
    "ഉത്സവത്തിന് വെടി വെച്ചു",
    # കോപ്പ് = vessel / utensil
    "അമ്മ കോപ്പ് കഴുകുന്നു",
    "ആ കോപ്പ് എടുത്ത് വെക്കൂ",
    "പാൽ വെക്കാൻ കോപ്പ് വേണം",
    # പട്ടി = dog
    "എന്റെ പട്ടി വളരെ cute ആണ്",
    "ആ പട്ടി കുട്ടിക്ക് പാൽ കൊടുക്കണം",
    "നല്ല പട്ടി തടവൊട്ടുന്നുണ്ട്",
    "നിന്റെ പട്ടി കുഞ്ഞു എവിടെയാ",
    "എന്റെ വീട്ടിൽ ഒരു പട്ടി ഉണ്ട്",
    "പട്ടി വളരെ നല്ല മൃഗമാണ്",
    "പട്ടി കുരക്കുന്നുണ്ട്",
    "ആ പട്ടി ഓടിപ്പോയി",
    "പട്ടിക്ക് പാൽ കൊടുക്കൂ",
    "നല്ല ഒരു പട്ടി കുട്ടി ആണ്",
    "നിന്റെ പട്ടി കുഞ്ഞു എത്രയായി",
    # കഴുത = donkey (animal)
    "കഴുത ചുമട് വഹിക്കുന്നു",
    "കഴുതയെ നോക്കി എല്ലാവരും ചിരിച്ചു",
    "ആ കഴുത വളരെ മണസ്സുള്ളതാണ്",
    # ചെറ്റ = thatched hut / fence (object)
    "ചെറ്റ വേലി കെട്ടിയത് നോക്കാൻ നല്ലതാണ്",
    "പറമ്പിൽ ചെറ്റ പുര ഉണ്ടാക്കണം",
    "ചെറിയ ചെറ്റ കൂട് ഇവിടെ കാണും",
    # മലർ = flower / puffed rice (ritual)
    "അമ്പലത്തിൽ മലർ വഴിപാട് ചെയ്യണം",
    "വിഷുവിനുള്ള മലർ നന്നായി ഫലിച്ചു",
    "മലർ വഴിപാട് എല്ലാ അമ്പലത്തിലും ഉണ്ട്",
    # കുരിശ് = cross (religious symbol)
    "പള്ളിയിലെ കുരിശ് വരക്കൽ ചടങ്ങ് ആണ്",
    "കുരിശ് വരച്ചു പ്രാർത്ഥനകൾ നടത്തി",
    "ഈ കുരിശ് വളരെ പാവനമാണ്",
    "ക്രിസ്ത്യൻ ആചാരങ്ങളിൽ കുരിശ് പ്രസക്തമാണ്"
]

def generate_data():
    synthetic_rows = []
    
    # Generate Offensive rows (300 Manglish + 300 Native per bad word)
    for word_en, word_ml in bad_words_map.items():
        # 1. Manglish Generation
        for _ in range(300):
            templates_pool = offensive_templates_en.copy()
            if word_en in word_specific_offensive_en:
                templates_pool.extend(word_specific_offensive_en[word_en])
            
            template = random.choice(templates_pool)
            if "{word}" in template:
                text = template.format(word=word_en)
            else:
                text = template
            synthetic_rows.append([text, 'Offensive_Untargetede', ''])
            
        # 2. Native Malayalam Generation
        for _ in range(300):
            templates_pool = offensive_templates_ml.copy()
            if word_ml in word_specific_offensive_ml:
                templates_pool.extend(word_specific_offensive_ml[word_ml])
            
            template = random.choice(templates_pool)
            if "{word}" in template:
                text = template.format(word=word_ml)
            else:
                text = template
            synthetic_rows.append([text, 'Offensive_Untargetede', ''])
            
    # Generate Safe rows (1000 Manglish + 1000 Native)
    for _ in range(1000):
        text = random.choice(safe_templates_en)
        synthetic_rows.append([text, 'Not_offensive', ''])
        
    for _ in range(1000):
        text = random.choice(safe_templates_ml)
        synthetic_rows.append([text, 'Not_offensive', ''])
        
    return synthetic_rows

if __name__ == "__main__":
    new_data = generate_data()
    print(f"Generated {len(new_data)} synthetic rows.")
    
    # Append to the existing CSV
    dataset_path = 'mal_full_offensive_train.csv'
    
    with open(dataset_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in new_data:
            writer.writerow(row)
            
    print(f"Successfully appended {len(new_data)} rows to {dataset_path}!")
