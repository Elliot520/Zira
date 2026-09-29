"""Hinglish for Kokoro TTS (2026-09-28). Zira writes Hindi in Latin letters ("aaj weather kaafi achha hai"). Kokoro's
English G2P reads those Hindi words as English ("bataunga" -> "bæTɔŋɡə"), and its Hindi G2P (espeak-ng `hi`) only
pronounces Hindi properly from Devanagari. So, per chunk of speech:

- a chunk that is really Hinglish (see is_hinglish) is spoken with the Hindi G2P, after its Hindi words are written
  in Devanagari: known words from the word list below, other clearly-not-English words by a simple spelling rule.
  English words ("weather", "model", "Android") stay exactly as they are - espeak-ng reads Latin words as English;
- everything else is plain English and goes to the English G2P untouched.

Nothing is sent anywhere; this is pure local text processing. is_english (a word -> bool check) comes from the
Kokoro worker, which has misaki's English dictionary (~180k words).
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Common Hindi words as Zira writes them in Latin letters -> Devanagari.
WORDS: dict[str, str] = {
    # pronouns and possessives
    "main": "मैं", "mai": "मैं", "mein": "में", "me": "में", "hum": "हम", "ham": "हम", "tum": "तुम", "tu": "तू",
    "aap": "आप", "ap": "आप", "woh": "वो", "wo": "वो", "vo": "वो", "yeh": "ये", "ye": "ये", "yah": "यह", "voh": "वह",
    "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "tera": "तेरा", "teri": "तेरी", "tere": "तेरे",
    "tumhara": "तुम्हारा", "tumhari": "तुम्हारी", "tumhare": "तुम्हारे", "aapka": "आपका", "aapki": "आपकी",
    "aapke": "आपके", "hamara": "हमारा", "hamari": "हमारी", "hamare": "हमारे", "uska": "उसका", "uski": "उसकी",
    "uske": "उसके", "iska": "इसका", "iski": "इसकी", "iske": "इसके", "unka": "उनका", "unki": "उनकी", "unke": "उनके",
    "mujhe": "मुझे", "mujhse": "मुझसे", "tumhe": "तुम्हें", "tumhein": "तुम्हें", "aapko": "आपको", "unhe": "उन्हें",
    "hume": "हमें", "humein": "हमें", "isko": "इसको", "usko": "उसको", "kisi": "किसी", "koi": "कोई", "kuch": "कुछ",
    "sab": "सब", "sabhi": "सभी", "sabse": "सबसे", "sabka": "सबका", "khud": "ख़ुद", "apna": "अपना", "apni": "अपनी",
    "apne": "अपने",
    # to be, to do, helpers
    "hai": "है", "hain": "हैं", "hoon": "हूँ", "hu": "हूँ", "hun": "हूँ", "ho": "हो", "tha": "था", "thi": "थी",
    "the": "थे", "thay": "थे", "hoga": "होगा", "hogi": "होगी", "honge": "होंगे", "hota": "होता", "hoti": "होती",
    "hote": "होते", "raha": "रहा", "rahi": "रही", "rahe": "रहे", "rha": "रहा", "rhi": "रही", "karta": "करता",
    "karti": "करती", "karte": "करते", "karna": "करना", "karo": "करो", "kar": "कर", "kiya": "किया", "kiye": "किए",
    "karunga": "करूँगा", "karungi": "करूँगी", "karenge": "करेंगे", "sakta": "सकता", "sakti": "सकती", "sakte": "सकते",
    "chahiye": "चाहिए", "chahta": "चाहता", "chahti": "चाहती", "lagta": "लगता", "lagti": "लगती", "laga": "लगा",
    "gaya": "गया", "gayi": "गई", "gaye": "गए", "diya": "दिया", "liya": "लिया", "lena": "लेना", "dena": "देना",
    "milega": "मिलेगा", "milta": "मिलता", "mila": "मिला", "dekho": "देखो", "dekh": "देख", "dekha": "देखा",
    "dekhna": "देखना", "suno": "सुनो", "bolo": "बोलो", "batao": "बताओ", "bataunga": "बताऊँगा",
    "bataungi": "बताऊँगी", "bataya": "बताया", "samjha": "समझा", "samjho": "समझो", "samajh": "समझ",
    "samjhao": "समझाओ", "chalo": "चलो", "chal": "चल", "jao": "जाओ", "jaana": "जाना", "jana": "जाना",
    "aana": "आना", "aao": "आओ", "aaya": "आया", "aayi": "आई", "rakh": "रख", "rakho": "रखो", "bana": "बना",
    "banao": "बनाओ", "banaya": "बनाया", "banati": "बनाती", "banata": "बनाता", "sochta": "सोचता", "socho": "सोचो",
    # small words
    "ki": "की", "ke": "के", "ka": "का", "ko": "को", "se": "से", "par": "पर", "pe": "पे", "tak": "तक", "na": "ना",
    "nahi": "नहीं", "nahin": "नहीं", "nhi": "नहीं", "mat": "मत", "haan": "हाँ", "han": "हाँ", "ji": "जी",
    "kya": "क्या", "kyun": "क्यों", "kyon": "क्यों", "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी",
    "kahan": "कहाँ", "kab": "कब", "kaun": "कौन", "kitna": "कितना", "kitni": "कितनी", "kitne": "कितने",
    "jab": "जब", "tab": "तब", "ab": "अब", "abhi": "अभी", "bhi": "भी", "hi": "ही", "to": "तो", "toh": "तो",
    "aur": "और", "ya": "या", "lekin": "लेकिन", "magar": "मगर", "kyunki": "क्योंकि", "kyonki": "क्योंकि",
    "agar": "अगर", "bas": "बस", "fir": "फिर", "phir": "फिर", "isliye": "इसलिए", "jaise": "जैसे", "waise": "वैसे",
    "aisa": "ऐसा", "aisi": "ऐसी", "aise": "ऐसे", "waisa": "वैसा", "wala": "वाला", "wali": "वाली", "wale": "वाले",
    "saath": "साथ", "baad": "बाद", "pehle": "पहले", "pahle": "पहले", "andar": "अंदर", "bahar": "बाहर",
    "upar": "ऊपर", "neeche": "नीचे", "hamesha": "हमेशा", "kabhi": "कभी", "sirf": "सिर्फ़", "bilkul": "बिल्कुल",
    "zaroor": "ज़रूर", "jaroor": "ज़रूर", "shayad": "शायद", "sach": "सच", "arre": "अरे", "yaar": "यार",
    # describing words
    "bahut": "बहुत", "bohot": "बहुत", "bahot": "बहुत", "zyada": "ज़्यादा", "jyada": "ज़्यादा", "kam": "कम",
    "thoda": "थोड़ा", "thodi": "थोड़ी", "achha": "अच्छा", "accha": "अच्छा", "acha": "अच्छा", "achhi": "अच्छी",
    "acchi": "अच्छी", "achhe": "अच्छे", "bura": "बुरा", "bada": "बड़ा", "badi": "बड़ी", "bade": "बड़े",
    "chhota": "छोटा", "chhoti": "छोटी", "naya": "नया", "nayi": "नई", "purana": "पुराना", "sahi": "सही",
    "galat": "ग़लत", "theek": "ठीक", "thik": "ठीक", "kaafi": "काफ़ी", "kafi": "काफ़ी", "saare": "सारे",
    "sara": "सारा", "saari": "सारी", "pura": "पूरा", "poora": "पूरा",
    # things and time
    "aaj": "आज", "kal": "कल", "parso": "परसों", "din": "दिन", "raat": "रात", "subah": "सुबह", "shaam": "शाम",
    "waqt": "वक़्त", "samay": "समय", "baat": "बात", "baatein": "बातें", "kaam": "काम", "ghar": "घर",
    "log": "लोग", "dost": "दोस्त", "pyaar": "प्यार", "dil": "दिल", "zindagi": "ज़िंदगी", "duniya": "दुनिया",
    "paani": "पानी", "khana": "खाना", "mausam": "मौसम", "matlab": "मतलब", "pata": "पता", "bhai": "भाई",
    "namaste": "नमस्ते", "shukriya": "शुक्रिया", "dhanyavaad": "धन्यवाद", "aadmi": "आदमी", "aurat": "औरत",
    "bachcha": "बच्चा", "bachche": "बच्चे", "ladka": "लड़का", "ladki": "लड़की", "beta": "बेटा", "maa": "माँ",
    "ek": "एक", "do": "दो", "teen": "तीन", "char": "चार", "paanch": "पांच", "das": "दस", "sau": "सौ",
    "hazaar": "हज़ार", "lakh": "लाख", "crore": "करोड़", "rupaye": "रुपये", "rupay": "रुपये", "sawal": "सवाल",
    "jawab": "जवाब", "madad": "मदद", "kahani": "कहानी", "gaana": "गाना", "tasveer": "तस्वीर",
}

# Also everyday English words: they only count as Hindi when the rest of the chunk already is Hindi.
AMBIGUOUS = {"main", "me", "the", "to", "do", "use", "par", "hi", "log", "din", "mat", "beta", "tab", "han",
             "char", "das", "ho", "ye", "ya", "kal", "bas", "sab", "ki", "na", "ka", "pe"}

# A word keeps its digits and inner dots/apostrophes together ("Qwen3.5", "don't"), so a model or version name is
# never cut up and half-transliterated.
_WORD = re.compile(r"[A-Za-z0-9]+(?:[.'][A-Za-z0-9]+)*|[^A-Za-z0-9]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def is_hinglish(text: str) -> bool:
    """A chunk is spoken as Hinglish when it has Devanagari, or at least two clearly-Hindi words (one is enough in
    a short chunk). "Main task is done" stays English: "main" alone is also English."""
    if _DEVANAGARI.search(text):
        return True
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", text)]
    strong = sum(1 for w in words if w in WORDS and w not in AMBIGUOUS)
    return strong >= 2 or (strong == 1 and len(words) <= 4)


def to_hinglish_speech(text: str, is_english: Callable[[str], bool]) -> str:
    """Hindi words -> Devanagari; English words, numbers, names with digits and punctuation unchanged."""
    out = []
    for token in _WORD.findall(text):
        if not token[0].isalpha() or any(ch.isdigit() or ch == "." for ch in token):
            out.append(token)  # punctuation, numbers, and names like "Qwen3.5" or "GPT-4o" stay as written
            continue
        low = token.lower()
        if low in WORDS:
            out.append(WORDS[low])
        elif is_english(low) or len(token) <= 2 or token.isupper():
            out.append(token)  # an English word, or an acronym like "API"
        else:
            out.append(transliterate(low))  # e.g. "widgets" is English above; "khidki" lands here
    return "".join(out)


# --- a simple romanised-Hindi spelling rule, for words not in WORDS ---
_CONSONANTS = [
    ("chh", "छ"), ("ksh", "क्ष"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"), ("dh", "ध"),
    ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("f", "फ़"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"),
    ("w", "व"), ("s", "स"), ("h", "ह"), ("z", "ज़"), ("q", "क़"), ("x", "क्स"),
]
_VOWELS = [  # (latin, independent letter, sign after a consonant)
    ("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("ee", "ई", "ी"), ("ii", "ई", "ी"), ("oo", "ऊ", "ू"),
    ("ou", "औ", "ौ"), ("a", "अ", ""), ("i", "इ", "ि"), ("u", "उ", "ु"), ("e", "ए", "े"), ("o", "ओ", "ो"),
]
_HALANT = "्"


def transliterate(word: str) -> str:
    """kahani -> कहानी, khidki -> खिडकी: good enough for espeak-ng's Hindi voice, not a spelling checker."""
    out: list[str] = []
    i, after_consonant = 0, False
    while i < len(word):
        for latin, letter in _CONSONANTS:
            if word.startswith(latin, i):
                nxt = i + len(latin)
                if after_consonant:
                    if out[-1] in ("न", "म") and letter not in ("न", "म", "य", "र", "ल", "व", "ह"):
                        out[-1] = "ं"  # a nasal before a consonant: anusvara (bataunga)
                    else:
                        out.append(_HALANT)
                out.append(letter)
                i, after_consonant = nxt, True
                break
        else:
            for latin, letter, sign in _VOWELS:
                if word.startswith(latin, i):
                    end = i + len(latin) >= len(word)
                    if after_consonant:
                        if latin == "a" and end:
                            sign = "ा"  # final -a is long in Hinglish spelling: bada, achha
                        elif latin == "i" and end:
                            sign = "ी"  # banati, gayi
                        out.append(sign)
                    else:
                        out.append(letter)
                    i, after_consonant = i + len(latin), False
                    break
            else:
                out.append(word[i])
                i, after_consonant = i + 1, False
    return "".join(out)
