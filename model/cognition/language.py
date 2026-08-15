"""Language and script identification for everything the mind takes in.

Two honest claims, kept separate.

**Script coverage is total.** The tokenizer is byte-level BPE, so every UTF-8
string encodes - no alphabet is architecturally excluded. The table below maps
Unicode ranges to writing systems across the world's scripts, so text in Tamil,
Ge'ez, Cherokee, Khmer, or Hangul is recognised as such rather than dropped into
a Latin-shaped default.

**Named-language resolution is partial and says so.** Script alone settles many
languages (Thai, Korean, Georgian, Greek...). Where one script carries many
languages - Latin, Cyrillic, Arabic, Devanagari, Han - short function-word
profiles separate the common cases and anything unresolved returns the script's
default with low confidence, or ``und``. It never silently guesses "English".

What this does *not* do is make a model fluent in a language it was never
trained on. FramerAI trains from scratch on local data, so competence follows the
corpus. What the mind gets from this module is the ability to notice which
language it is in, remember per-language experience, answer in the language it
was addressed in, and let curiosity pull toward the languages it is worst at -
which is the part that is architecture rather than data.
"""

import re
import unicodedata
from dataclasses import dataclass

# (first, last, script, default language code). Ordered widest-use first; the
# scan takes the script holding the most characters, so mixed text resolves to
# its dominant writing system rather than its first character.
SCRIPT_RANGES: tuple[tuple[int, int, str, str], ...] = (
    (0x0041, 0x024F, "Latin", "und-Latn"),
    (0x1E00, 0x1EFF, "Latin", "und-Latn"),
    (0x0370, 0x03FF, "Greek", "el"),
    (0x1F00, 0x1FFF, "Greek", "el"),
    (0x0400, 0x052F, "Cyrillic", "und-Cyrl"),
    (0x0530, 0x058F, "Armenian", "hy"),
    (0x0590, 0x05FF, "Hebrew", "he"),
    (0x0600, 0x06FF, "Arabic", "und-Arab"),
    (0x0750, 0x077F, "Arabic", "und-Arab"),
    (0x08A0, 0x08FF, "Arabic", "und-Arab"),
    (0x0700, 0x074F, "Syriac", "syr"),
    (0x0780, 0x07BF, "Thaana", "dv"),
    (0x07C0, 0x07FF, "NKo", "nqo"),
    (0x0900, 0x097F, "Devanagari", "und-Deva"),
    (0x0980, 0x09FF, "Bengali", "bn"),
    (0x0A00, 0x0A7F, "Gurmukhi", "pa"),
    (0x0A80, 0x0AFF, "Gujarati", "gu"),
    (0x0B00, 0x0B7F, "Odia", "or"),
    (0x0B80, 0x0BFF, "Tamil", "ta"),
    (0x0C00, 0x0C7F, "Telugu", "te"),
    (0x0C80, 0x0CFF, "Kannada", "kn"),
    (0x0D00, 0x0D7F, "Malayalam", "ml"),
    (0x0D80, 0x0DFF, "Sinhala", "si"),
    (0x0E00, 0x0E7F, "Thai", "th"),
    (0x0E80, 0x0EFF, "Lao", "lo"),
    (0x0F00, 0x0FFF, "Tibetan", "bo"),
    (0x1000, 0x109F, "Myanmar", "my"),
    (0x10A0, 0x10FF, "Georgian", "ka"),
    (0x1200, 0x137F, "Ethiopic", "am"),
    (0x13A0, 0x13FF, "Cherokee", "chr"),
    (0x1400, 0x167F, "Canadian Aboriginal", "iu"),
    (0x1680, 0x169F, "Ogham", "sga"),
    (0x16A0, 0x16FF, "Runic", "non"),
    (0x1700, 0x171F, "Tagalog", "tl"),
    (0x1780, 0x17FF, "Khmer", "km"),
    (0x1800, 0x18AF, "Mongolian", "mn"),
    (0x1B00, 0x1B7F, "Balinese", "ban"),
    (0x1C80, 0x1C8F, "Cyrillic", "und-Cyrl"),
    (0x2C80, 0x2CFF, "Coptic", "cop"),
    (0x2D30, 0x2D7F, "Tifinagh", "ber"),
    (0x3040, 0x309F, "Hiragana", "ja"),
    (0x30A0, 0x30FF, "Katakana", "ja"),
    (0x3130, 0x318F, "Hangul", "ko"),
    (0x1100, 0x11FF, "Hangul", "ko"),
    (0x4E00, 0x9FFF, "Han", "zh"),
    (0x3400, 0x4DBF, "Han", "zh"),
    (0xA000, 0xA48F, "Yi", "ii"),
    (0xA980, 0xA9DF, "Javanese", "jv"),
    (0xAC00, 0xD7AF, "Hangul", "ko"),
    (0x10900, 0x1091F, "Phoenician", "phn"),
    (0x1E900, 0x1E95F, "Adlam", "ff"),
    (0x13000, 0x1342F, "Egyptian Hieroglyphs", "egy"),
)

# Function words are the cheapest reliable signal: they are frequent, short, and
# rarely borrowed. Enough to separate the languages that share a script.
_PROFILES: dict[str, frozenset[str]] = {
    "en": frozenset("the of and to in is that it for was with as on are you this be have not".split()),
    "es": frozenset("de la que el en y los se del las por un para con no una su es".split()),
    "fr": frozenset("le de un et les des en du une que dans qui pour pas sur est au".split()),
    "de": frozenset("der die und den von zu das mit sich des auf für ist nicht ein eine dem".split()),
    "pt": frozenset("de que não uma dos com para por como mais mas ao das mesmo mim mas".split()),
    "it": frozenset("che di non per una sono con come sul nel gli gliela alla gli gli però".split()),
    "nl": frozenset("het een van en dat zijn niet met voor aan op ook maar heeft worden".split()),
    "sv": frozenset("och att det som med för inte har den till av var men detta".split()),
    "da": frozenset("og det ikke til at med for men den han hun var som af".split()),
    "no": frozenset("og det ikke til at med for men den han hun var som av".split()),
    "fi": frozenset("ja on ei että se hän niin kuin mutta jos kun ovat oli".split()),
    "pl": frozenset("nie się jest że nas tak jak dla przez oraz który tego jego jej".split()),
    "cs": frozenset("není jsem jsou které nebo aby jako tak ale když pro toho jeho".split()),
    "ro": frozenset("este care nu pentru sunt care din mai fost când dar acest".split()),
    "tr": frozenset("bir ve bu için ile daha çok olarak gibi ama kadar sonra".split()),
    "id": frozenset("yang dan di untuk dengan tidak ini itu dari akan pada dalam".split()),
    "vi": frozenset("của và các là được trong người những cho không với khi".split()),
    "sw": frozenset("kwa katika wa ya na si kama lakini hii hiyo yake wake".split()),
    "tl": frozenset("ang mga sa ng na at ay hindi ito para may kung".split()),
    "hu": frozenset("hogy nem egy volt csak meg már mint ezt majd amikor".split()),
    "ca": frozenset("que amb això però quan per una les dels són molt".split()),
    "eu": frozenset("eta bat hau baina zen dira nola dela gehiago beste".split()),
    "ru": frozenset("что как это для его она они был при или уже так все".split()),
    "uk": frozenset("що для його вона вони був при або вже так все які".split()),
    "bg": frozenset("това като която които той тя ние вече само защото".split()),
    "sr": frozenset("који што ово као али када због него сваки такође".split()),
    "mk": frozenset("што како ова кога затоа само нешто добро многу".split()),
    "ar": frozenset("في من على أن إلى هذا التي كان مع هو لا ما عن".split()),
    "fa": frozenset("این که برای است های خود یک شده کرد آن ولی چون".split()),
    "ur": frozenset("کے میں سے ہے کی کو نے پر اور یہ ہیں تھا".split()),
    "hi": frozenset("के में की है और से को नहीं यह कि पर एक हैं".split()),
    "mr": frozenset("आहे आणि यांनी होते त्यांनी मात्र असून केली".split()),
    "ne": frozenset("छन् गरेको भएको हुन् त्यसैले तर पनि लागि".split()),
}

LANGUAGE_NAMES: dict[str, str] = {
    "am": "Amharic", "ar": "Arabic", "ban": "Balinese", "ber": "Tamazight", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "ca": "Catalan", "chr": "Cherokee", "cop": "Coptic",
    "cs": "Czech", "da": "Danish", "de": "German", "dv": "Dhivehi", "egy": "Egyptian",
    "el": "Greek", "en": "English", "es": "Spanish", "eu": "Basque", "fa": "Persian",
    "ff": "Fulani", "fi": "Finnish", "fr": "French", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "ii": "Yi",
    "it": "Italian", "iu": "Inuktitut", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian",
    "km": "Khmer", "kn": "Kannada", "ko": "Korean", "lo": "Lao", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "my": "Burmese", "ne": "Nepali",
    "no": "Norwegian", "non": "Old Norse", "nqo": "N'Ko", "or": "Odia", "pa": "Punjabi",
    "phn": "Phoenician", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sga": "Old Irish", "si": "Sinhala", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "syr": "Syriac", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Tagalog",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
    "und": "unidentified", "und-Latn": "unidentified (Latin script)",
    "und-Cyrl": "unidentified (Cyrillic script)", "und-Arab": "unidentified (Arabic script)",
    "und-Deva": "unidentified (Devanagari script)",
}

# Which profiles are worth testing once the script is known. Anything not listed
# is settled by its script alone.
_SCRIPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Latin": ("en", "es", "fr", "de", "pt", "it", "nl", "sv", "da", "no", "fi", "pl",
              "cs", "ro", "tr", "id", "vi", "sw", "tl", "hu", "ca", "eu"),
    "Cyrillic": ("ru", "uk", "bg", "sr", "mk"),
    "Arabic": ("ar", "fa", "ur"),
    "Devanagari": ("hi", "mr", "ne"),
}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class Language:
    """What the mind concluded about the language of an input."""

    code: str = "und"
    script: str = "Unknown"
    confidence: float = 0.0

    @property
    def name(self) -> str:
        return LANGUAGE_NAMES.get(self.code, self.code)

    @property
    def identified(self) -> bool:
        return self.code != "und" and not self.code.startswith("und-")

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name,
            "script": self.script, "confidence": round(self.confidence, 3),
        }


def detect_script(text: str) -> tuple[str, str, float]:
    """Dominant writing system: ``(script, default_code, share_of_letters)``.

    Han text containing kana is Japanese, which is the one script overlap common
    enough to be worth special-casing.
    """
    counts: dict[str, int] = {}
    defaults: dict[str, str] = {}
    letters = 0

    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        point = ord(char)
        for first, last, script, code in SCRIPT_RANGES:
            if first <= point <= last:
                counts[script] = counts.get(script, 0) + 1
                defaults[script] = code
                break
        else:
            name = unicodedata.name(char, "")
            script = name.split(" ")[0].title() if name else "Unknown"
            counts[script] = counts.get(script, 0) + 1
            defaults.setdefault(script, "und")

    if not counts:
        return "Unknown", "und", 0.0

    if counts.get("Han") and (counts.get("Hiragana") or counts.get("Katakana")):
        share = (counts.get("Han", 0) + counts.get("Hiragana", 0)
                 + counts.get("Katakana", 0)) / letters
        return "Japanese", "ja", share

    script = max(counts, key=lambda s: counts[s])
    return script, defaults.get(script, "und"), counts[script] / letters


def detect_language(text: str) -> Language:
    """Identify the language of a string, or admit that it did not.

    Confidence is deliberately conservative: a script-only result reports the
    script's share of the letters and a code marked ``und-`` so callers can tell
    "I know this is Cyrillic" from "I know this is Ukrainian".
    """
    if not text or not text.strip():
        return Language()

    script, default, share = detect_script(text)
    if script == "Unknown":
        return Language(script=script)

    candidates = _SCRIPT_CANDIDATES.get(script)
    if not candidates:
        return Language(code=default, script=script, confidence=round(share, 3))

    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return Language(code=default, script=script, confidence=round(share * 0.5, 3))

    scored = sorted(
        ((sum(w in _PROFILES[code] for w in words) / len(words), code) for code in candidates),
        reverse=True,
    )
    best_score, best_code = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if best_score == 0.0:
        # The script is certain, the language is not. Say exactly that.
        return Language(code=default, script=script, confidence=round(share * 0.4, 3))

    # Margin over the runner-up, so "the two profiles tie" reads as low
    # confidence, and an evidence factor, so one matching function word is not
    # treated as the same evidence as a whole sentence of them.
    margin = (best_score - runner_up) / best_score
    evidence = 0.5 + 0.5 * min(1.0, len(words) / 8)
    confidence = min(1.0, share * (0.45 + 0.55 * margin) + min(0.35, best_score * 2)) * evidence
    return Language(code=best_code, script=script, confidence=round(confidence, 3))


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


# Every profile's function words pooled together. Used for topic labelling: a
# word that is grammar in *some* language is a poor name for a subject, and a
# mind that only knows English stopwords labels Spanish input "porque".
_FUNCTION_WORDS = frozenset().union(*_PROFILES.values())

# Writing systems that do not separate words with spaces, where splitting on
# whitespace yields one enormous "word" instead of a usable label.
NON_SPACING_SCRIPTS = frozenset(
    {"Han", "Hiragana", "Katakana", "Japanese", "Thai", "Khmer", "Lao", "Myanmar", "Tibetan"}
)


def is_function_word(word: str) -> bool:
    """True if this is grammar rather than subject matter, in any known profile."""
    return word.lower() in _FUNCTION_WORDS
