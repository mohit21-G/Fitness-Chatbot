"""
Food Search Engine
Implements the full matching pipeline:

  User input
    → normalize()
    → exact name match
    → exact alias match
    → variant strip + re-match
    → fuzzy name match
    → fuzzy alias match
    → clarification (low-confidence)

Each step returns a SearchResult with a match_type and confidence score
so the API caller can decide how to present the result.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from rapidfuzz import fuzz, process as rf_process

from food_repository import FoodRepositorySync


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum fuzzy score (0-100) to accept as a confident match
FUZZY_CONFIDENT_THRESHOLD = 85

# Below this score → return as low-confidence / needs clarification
FUZZY_LOW_CONFIDENCE_THRESHOLD = 55

# How many fuzzy candidates to surface in a low-confidence result
MAX_CLARIFICATION_CANDIDATES = 5

# Preparation / cooking-method keywords that can appear in user input
# but may not be part of the canonical food_name.
# Order matters: more specific phrases first.
VARIANT_KEYWORDS: list[str] = [
    # oils / fats
    "with ghee", "in ghee", "ghee",
    "with butter", "in butter",
    "with oil", "in oil",
    # cooking methods
    "deep fried", "pan fried", "stir fried", "fried",
    "boiled", "steamed", "baked", "grilled", "roasted",
    "raw", "fresh",
    # extras
    "with sugar", "without sugar",
    "with salt", "without salt",
    "masala", "spicy", "plain",
    # portion descriptors (strip these too)
    "piece", "pieces", "slice", "slices",
    "bowl", "bowls", "cup", "cups", "plate", "plates", "serving", "servings",
    "glass", "glasses", "katori", "katoris", "bottle", "bottles", "packet", "packets",
    "small", "medium", "large",
    "half", "full",
]

# Regex built once at import time
_VARIANT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in VARIANT_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Native-script (Devanagari / Gujarati) -> Roman transliteration
# ---------------------------------------------------------------------------
# Applied to raw text BEFORE Unicode NFKD normalisation (NFKD strips the
# combining vowel signs / matras, which would otherwise destroy the syllables).
# The Roman output is intentionally shaped to feed the Latin TRANSLITERATION_MAP
# below, e.g.  રોટલી -> "rotli" -> (map) -> "roti".  This is deterministic and
# does not touch the model, DB, or nutrition schema.
#
# Devanagari block: U+0900–U+097F   |   Gujarati block: U+0A80–U+0AFF
# Both scripts share the same abugida layout, so we map by phonetic value.

# Independent vowels
_INDIC_VOWELS = {
    # Devanagari
    "अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "अं": "an",
    # Gujarati
    "અ": "a", "આ": "aa", "ઇ": "i", "ઈ": "i", "ઉ": "u", "ઊ": "u",
    "ઋ": "ri", "એ": "e", "ઐ": "ai", "ઓ": "o", "ઔ": "au",
}

# Consonants (inherent "a" is added contextually, see transliterate below)
_INDIC_CONSONANTS = {
    # Devanagari
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # Gujarati
    "ક": "k", "ખ": "kh", "ગ": "g", "ઘ": "gh", "ઙ": "ng",
    "ચ": "ch", "છ": "chh", "જ": "j", "ઝ": "jh", "ઞ": "ny",
    "ટ": "t", "ઠ": "th", "ડ": "d", "ઢ": "dh", "ણ": "n",
    "ત": "t", "થ": "th", "દ": "d", "ધ": "dh", "ન": "n",
    "પ": "p", "ફ": "ph", "બ": "b", "ભ": "bh", "મ": "m",
    "ય": "y", "ર": "r", "લ": "l", "વ": "v", "ળ": "l",
    "શ": "sh", "ષ": "sh", "સ": "s", "હ": "h",
}

# Dependent vowel signs (matras) — attach to the preceding consonant,
# replacing its inherent "a".
_INDIC_MATRAS = {
    # Devanagari
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    # Gujarati
    "ા": "a", "િ": "i", "ી": "i", "ુ": "u", "ૂ": "u",
    "ૃ": "ri", "ે": "e", "ૈ": "ai", "ો": "o", "ૌ": "au",
}

# Virama / halant — suppresses the inherent vowel of the preceding consonant.
_INDIC_VIRAMA = {"\u094D", "\u0ACD"}   # Devanagari, Gujarati

# Anusvara / chandrabindu / nukta / avagraha — treat nasal as "n", drop the rest.
_INDIC_NASAL = {"\u0902", "\u0A82", "\u0901", "\u0A81"}   # anusvara / candrabindu
_INDIC_IGNORE = {"\u093C", "\u0ABC", "\u094D"}            # nukta (handled), virama


def _is_indic_char(c: str) -> bool:
    return ("\u0900" <= c <= "\u097F") or ("\u0A80" <= c <= "\u0AFF")


def transliterate_indic(text: str) -> str:
    """
    Deterministic syllabic Devanagari/Gujarati -> Roman transliteration.

    Produces a phonetic Roman form (with inherent-vowel handling) that is then
    canonicalised by TRANSLITERATION_MAP.  Non-Indic characters pass through
    unchanged, so mixed sentences are safe.
    """
    if not any(_is_indic_char(c) for c in text):
        return text

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in _INDIC_CONSONANTS:
            base = _INDIC_CONSONANTS[c]
            # Look ahead: matra, virama, or nasal?
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in _INDIC_VIRAMA:
                # Consonant cluster: no vowel, skip virama
                out.append(base)
                i += 2
                continue
            if nxt in _INDIC_MATRAS:
                out.append(base + _INDIC_MATRAS[nxt])
                i += 2
                continue
            # No matra following. Apply schwa deletion: a word-final consonant
            # (end of string, whitespace, or a non-Indic char follows) drops its
            # inherent "a" — matching how these words are Romanised in practice
            # (દાળ -> "dal", पनीर -> "panir", not "dala"/"panira").
            if nxt == "" or nxt.isspace() or not _is_indic_char(nxt):
                out.append(base)
            else:
                out.append(base + "a")
            i += 1
            continue
        if c in _INDIC_VOWELS:
            out.append(_INDIC_VOWELS[c])
            i += 1
            continue
        if c in _INDIC_MATRAS:
            # Orphan matra (no preceding consonant) -> its vowel
            out.append(_INDIC_MATRAS[c])
            i += 1
            continue
        if c in _INDIC_NASAL:
            out.append("n")
            i += 1
            continue
        if c in _INDIC_IGNORE:
            i += 1
            continue
        # Non-Indic char: pass through
        out.append(c)
        i += 1
    return "".join(out)


# Common Indian transliteration noise & typos (user types → canonical)
TRANSLITERATION_MAP: dict[str, str] = {
    # bhakri / bhakhri
    "bhkhari": "bhakri",
    "bhkhri": "bhakri",
    "bhkari": "bhakri",
    "bhkri": "bhakri",
    "bhakhri": "bhakri",
    "bhakharis": "bhakri",
    "bhakhari": "bhakri",
    "bhakhars": "bhakri",
    "bhakhar": "bhakri",
    "bhakara": "bhakri",
    "bhakari": "bhakri",
    # milk / dudh
    "dudh": "milk",
    "doodh": "milk",
    "dud": "milk",
    "dood": "milk",
    "douh": "milk",
    # samosa / samossa
    "samossa": "samosa",
    "samoosa": "samosa",
    "samasa": "samosa",
    "smosa": "samosa",
    # roti variants
    "chappati": "chapati",
    "rotli": "roti",        # Gujarati
    "roatli": "roti",
    "rotoli": "roti",
    "rotali": "roti",       # from Gujarati રોટલી transliteration (medial schwa)
    # rotlo / rotla (Gujarati thick millet flatbread — distinct from wheat roti!)
    "rotalo": "rotlo",      # from Gujarati રોટલો transliteration (medial schwa)
    "rotala": "rotlo",      # from Gujarati રોટલા transliteration
    "rotla": "rotlo",       # plural / dialect form normalized to rotlo
    "bajari": "bajri",      # from Gujarati બાજરી transliteration
    "bajara": "bajra",      # from Gujarati બાજરા transliteration
    "bajarino": "bajri no", # Gujarati compound બાજરીનો
    "bajrano": "bajra no",  # Gujarati compound બાજરાનો
    # daal / dal
    "daal": "dal",
    "dhal": "dal",
    "daahl": "dal",
    "dahl": "dal",
    # puri
    "poori": "puri",
    "pooris": "puri",
    # aloo / potato
    "alu": "aloo",
    "alloo": "aloo",
    "bataka": "aloo",       # Gujarati → potato
    "bateka": "aloo",       # Gujarati alternate spelling → potato
    "bateki": "aloo",       # Gujarati variant
    "batata": "aloo",
    "potato": "aloo",
    # paneer
    "paner": "paneer",
    "panir": "paneer",
    # idli
    "idly": "idli",
    "idlee": "idli",
    # dosa
    "dosai": "dosa",
    "thosai": "dosa",
    "dossai": "dosa",
    # biryani & chicken
    "chiken": "chicken",
    "chikan": "chicken",
    "chikn": "chicken",
    "chckn": "chicken",
    "checken": "chicken",
    "biriyani": "biryani",
    "biriyaani": "biryani",
    "briyani": "biryani",
    # pizza variants
    "margarita": "margherita",
    "margerita": "margherita",
    "margrita": "margherita",
    "piza": "pizza",
    "pizzas": "pizza",
    # papad
    "papadum": "papad",
    "pappadam": "papad",
    "papadam": "papad",
    # chai / tea
    "chay": "chai",
    "cha": "chai",
    "chaa": "chai",
    "chha": "chai",
    "chaha": "chai",
    # curd / yogurt
    "yoghurt": "yogurt",
    # rice / chawal
    "chawal": "rice",
    "chaval": "rice",
    "bhat": "rice",
    "bhaat": "rice",
    # buttermilk — map to the canonical local food_name "chhas"
    "chhaas": "chhas",
    "chaas": "chhas",
    "chas": "chhas",
    "chhach": "chhas",
    "chach": "chhas",
    "buttermilk": "chhas",
    "buttermlik": "chhas",
    "butter milk": "chhas",
    "chhaash": "chhas",
    "mattha": "chhas",
    # Additional buttermilk/chaas variants
    "chass":  "chhas",
    "chhass": "chhas",
    "takra":  "chhas",      # Gujarati/Sanskrit name for buttermilk
    # Water — pani/paani are the most common Indian names
    "pani":   "water",
    "paani":  "water",
    "neer":   "water",      # Tamil/Telugu name for water
    # Sugarcane juice
    "ganne ka ras":  "sugarcane juice",
    "ganna juice":   "sugarcane juice",
    "ganne":         "sugarcane juice",
    "ganna ras":     "sugarcane juice",
    # Lemon-based drinks
    "limbu pani":   "lemonade",   # Gujarati lemon water
    "nimbu pani":   "lemonade",   # Hindi lemon water
    "nimbu paani":  "lemonade",
    "shikanji":     "lemonade",   # spiced lemonade — closest common match
    # Turmeric / almond enriched milk
    "haldi doodh":  "turmeric milk",
    "haldi milk":   "turmeric milk",
    "badam doodh":  "almond milk",
    "badam milk":   "almond milk",
    # Cola / soft drink generic terms
    "cola":         "soda",       # generic cola → soda (DB category match)
    "cold drink":   "soda",
    "soft drink":   "soda",
    # Coconut water Gujarati/Hindi variants
    "nariyal pani":  "coconut water",
    "naariyal pani": "coconut water",
    "nariyal paani": "coconut water",
    # Gujarati snacks & dishes
    "gathia": "gathiya",
    "gatiya": "gathiya",
    "khakra": "khakhra",
    "khakhara": "khakhra",
    "fafada": "fafda",
    # gundi — deep-fried Gujarati wheat/besan snack (many spellings)
    "gunde": "gundi",
    "gundiya": "gundi",
    "gundy": "gundi",
    "goondi": "gundi",
    "goodi": "gundi",
    # chakli / chakri — spiral fried snack
    "chakri": "chakli",
    "chakkli": "chakli",
    "chakale": "chakli",
    "murukku": "chakli",
    "muruku": "chakli",
    # moong dal in Gujarati = "mag" or "mug"
    "mag": "moong dal",
    "mug": "moong dal",
    "mung": "moong dal",
    # sabji / subji / shak — Gujarati/Hinglish for vegetable dish
    # These are stripped as method words in compound-fallback; the map entry
    # here ensures they normalise consistently before cleaning.
    "subji": "sabji",
    "subzi": "sabzi",
    "shak": "sabji",        # Gujarati shak = sabji/vegetable dish
    "shaak": "sabji",       # alternate spelling
    # Gujarati preparation words — strip to a neutral form so the head food
    # can be resolved.  bafela/bafeli/bafelu = boiled; phanagavela = soaked/sprouted.
    # Mapping them to their English equivalent lets the compound-fallback strip
    # them as method words and match the bare food name.
    "bafela":       "boiled",
    "bafeli":       "boiled",
    "bafelu":       "boiled",
    "bafelo":       "boiled",
    "phanagavela":  "sprouted",
    "phanagaveli":  "sprouted",
    "phanagavelu":  "sprouted",
    "phangavela":   "sprouted",  # common short form
    "phangaveli":   "sprouted",
    "bhajavela":    "fried",    # Gujarati fried
    "bhajeli":      "fried",
    "shekela":      "roasted",  # Gujarati roasted
    "sekela":       "roasted",
    "dhokala": "dhokla",
    "dhoklaa": "dhokla",
    "undhyu": "undhiyu",
    "ringna": "eggplant",   # eggplant IS in DB; brinjal is not
    "ringan": "eggplant",
    "dudhi": "bottle gourd",
    "tindola": "ivy gourd",
    "karela": "bitter gourd",   # bitter gourd → sabji resolver will handle it
    "methi": "fenugreek",
    "palak": "spinach",
    "panipuri": "pani puri",
    "paanipuri": "pani puri",
    "thepala": "thepla",
    "theplaa": "thepla",
    "bhendi": "bhindi",
    "chaye": "chai",
    "coffe": "coffee",
    "koffee": "coffee",
    "sandwitch": "sandwich",
    "noodels": "noodles",
    "nudles": "noodles",
    "keechdi": "khichdi",
    "kichdi": "khichdi",
    "daliyaa": "daliya",
    "frnch": "french",
    "kaaju": "kaju",
    "kaajuu": "kaju",
    "kheerr": "kheer",
    "gulaab": "gulab",
    "jalabi": "jalebi",
    "raajma": "rajma",
    "saambar": "sambar",
    "paalak": "palak",
    "panir": "paneer",
    "panear": "paneer",    # common English misspelling
    "paner": "paneer",
    "lasssi": "lassi",
    "pizzza": "pizza",
    "burgur": "burger",
    "khamann": "khaman",
    "kakadi": "cucumber",
    "makkai": "corn",
    "makai": "corn",
    "baigan": "baingan",
    "soop": "soup",
    "orrange": "orange",
    "tarbuj": "watermelon",
    "papeeta": "papaya",
    "steemed": "steamed",
    "wada": "vada",
    "paav": "pav",
    # chutney variants (Gujarati/Hindi spellings)
    "chattni": "chutney",
    "chatni": "chutney",
    "chutni": "chutney",
    "chatney": "chutney",
    "chutnee": "chutney",
    "chatpata": "chutney",
    # sambar variants
    "sambhar": "sambar",
    "sambaar": "sambar",
    "saambar": "sambar",
    "sambhaar": "sambar",
    # coconut / nariyal
    "nariyal": "coconut",
    "naryal": "coconut",
    "nariyel": "coconut",
    "khopra": "coconut",
    "copra": "coconut",
    # other common regional food words
    "kadhi": "kadhi",
    "raita": "raita",
    "raitu": "raita",
    "achar": "pickle",
    "achaar": "pickle",
    "aachar": "pickle",
    "lehsun": "garlic",
    "adrak": "ginger",
    "dhaniya": "coriander",
    "kothmir": "coriander",
    "kothimbir": "coriander",
    "pudina": "mint",
    "imli": "tamarind",
    "gur": "jaggery",
    "gud": "jaggery",
    # ── Fix: map transliteration targets to actual DB food names ────────────
    # ringna/ringan were mapped to "brinjal" which has no DB food;
    # eggplant IS in DB — use it as the canonical target.
    "ringna":  "eggplant",
    "ringan":  "eggplant",
    "vangi":   "eggplant",
    "baigan":  "eggplant",
    "begun":   "eggplant",
    # bottle gourd / lauki
    "lauki":   "bottle gourd",   # sabji resolver handles "bottle gourd nu shak"
    "doodhi":  "bottle gourd",
    # ivy gourd / tindora
    "tindora": "ivy gourd",
    "tindli":  "ivy gourd",
    "kundru":  "ivy gourd",
    # ridge gourd / turiya
    "turiya":  "ridge gourd",
    "turia":   "ridge gourd",
    "gilki":   "ridge gourd",
    "torai":   "ridge gourd",
    # yam / suran
    "suran":   "yam",
    # sweet potato
    "ratalu":      "sweet potato",
    "sakarkand":   "sweet potato",
    "shakarkand":  "sweet potato",
    # peanuts / groundnut — point to "peanut" (added to DB)
    "singdana":    "peanut",
    "shengdana":   "peanut",
    "mungfali":    "peanut",
    "mungphali":   "peanut",
    "moongfali":   "peanut",
    "moongphali":  "peanut",
    # garlic / lasan
    "lasan":   "garlic",
    "lasun":   "garlic",
    # other Indian vegetables
    "turai":   "ridge gourd",
    "parwal":  "pointed gourd",
    "kunduri": "ivy gourd",
    # fish / meat common Hindi/Bengali
    "machli":  "fish",
    "maachh":  "fish",
    "murga":   "chicken",
    "maans":   "mutton",
    # spices (may not resolve to standalone food — that's ok)
    "jeera":   "cumin",
    "dhania":  "coriander",
    "haldi":   "turmeric",
    # daliya — standalone means the Indian porridge
    "dalia":   "daliya",
    "daaliya": "daliya",
    # moong dal joined
    "moongdal": "moong dal",
    "mungdal":  "moong dal",
}

# ---------------------------------------------------------------------------
# Joined-word boundary table
# ---------------------------------------------------------------------------
# When users type food names without spaces (common on mobile / Hinglish),
# we try to split at known word boundaries BEFORE the transliteration map runs.
# Each entry maps a joined token → the spaced version.
# Only add entries where the split produces words that appear in the
# TRANSLITERATION_MAP or as DB food names / aliases.
# KEEP THIS SHORT AND GENERIC — do NOT add individual food combinations here;
# add only the head/suffix words that can combine with many partners.
_JOINED_FOOD_SPLITS: dict[str, str] = {
    # Chana combinations
    "singchana":     "sing chana",
    "bhunachana":    "bhuna chana",
    # Dal combinations
    "moongdal":      "moong dal",
    "mungdal":       "moong dal",
    "chhadal":       "chha dal",
    # Paratha / roti combinations
    "paneerparatha":  "paneer paratha",
    "alooparatha":    "aloo paratha",
    "gobhiparatha":   "gobhi paratha",
    "methiparatha":   "methi paratha",
    "maidaparatha":   "maida paratha",
    "alooroti":       "aloo roti",
    # Rice combinations
    "rajmachawal":    "rajma chawal",
    "dalichawal":     "dal chawal",
    "kaddichawal":    "kaddi chawal",
    # Puri combinations
    "aloopuri":       "aloo puri",
    # Vada / vadapav
    "vadapav":        "vada pav",
    "wadapav":        "wada pav",
    # Pani puri
    "panipuri":       "pani puri",   # already in map but add here too
    "sevpuri":        "sev puri",
    "dahipuri":       "dahi puri",
    # Butter / ghee prefix
    "butterroti":     "butter roti",
    "gheechapati":    "ghee chapati",
}


# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class MatchType(str, Enum):
    EXACT_NAME    = "exact_name"
    EXACT_ALIAS   = "exact_alias"
    VARIANT_STRIP = "variant_strip"   # variant keyword removed → re-matched
    FUZZY_NAME    = "fuzzy_name"
    FUZZY_ALIAS   = "fuzzy_alias"
    LOW_CONFIDENCE = "low_confidence"
    NO_MATCH       = "no_match"


@dataclass
class SearchResult:
    """Returned by FoodSearchEngine.search()"""
    query_original: str
    query_normalized: str
    match_type: MatchType
    confidence: float                        # 0.0 – 1.0
    food: Optional[dict] = None

    # Populated only for VARIANT_STRIP matches
    variant_detected: Optional[str] = None  # e.g. "ghee"
    stripped_query: Optional[str] = None    # query after removing variant word

    # Populated only for LOW_CONFIDENCE / NO_MATCH
    candidates: list[dict] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Canonical normalisation applied to every user query AND to every
    candidate before scoring.

    Steps:
      0a. Joined-word split (e.g. "singchana" → "sing chana")
      0b. Native-script (Devanagari / Gujarati) -> Roman
      1. Unicode NFKD → strip accents
      2. Lowercase
      3. Clean non-alphanumeric noise characters
      4. Collapse whitespace
      5. Apply transliteration map (word-level)
      6. Strip eating verbs
    """
    # 0a. Joined-word split — resolve no-space compound food words BEFORE
    #     everything else so the rest of the pipeline sees spaced tokens.
    text_lower = (text or "").strip().lower()
    if text_lower in _JOINED_FOOD_SPLITS:
        text = _JOINED_FOOD_SPLITS[text_lower]

    # 0b. Native-script (Devanagari / Gujarati) -> Roman FIRST.
    #    Must run before NFKD, which strips the combining vowel signs (matras)
    #    that carry the syllable's vowel. Non-Indic text passes through unchanged.
    text = transliterate_indic(text)

    # 1. Unicode normalise + strip combining characters
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # 2. Lowercase
    text = text.lower()

    # 3. Clean punctuation / non-alphanumeric symbols (e.g. n=me -> n me)
    text = re.sub(r"[^\w\s]", " ", text)

    # 4. Collapse whitespace / trim
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Transliteration map — whole-word replacements
    for wrong, right in TRANSLITERATION_MAP.items():
        # Use word boundaries where possible
        pattern = r"\b" + re.escape(wrong.strip()) + r"\b"
        text = re.sub(pattern, right, text)

    # 6. Strip sentence-ending eating/drinking verbs that are never part of a
    #    food name (e.g. "gundi khadha" → "gundi", "dal khadhi" → "dal").
    #    Only strip at word boundaries; multi-word food names survive.
    _EAT_VERBS = (
        "khadha", "khadhi", "khadhu", "khadho", "khaya", "khayi", "khaye", "khai",
        "lidha", "lidhi", "lidhu", "lidho",
        "jamya", "jamyu",
        "pidha", "pidhi", "pidhu",
        "piya", "piyi",
        "ate", "eaten", "had", "drank",
    )
    for v in _EAT_VERBS:
        text = re.sub(r"\b" + v + r"\b", " ", text)

    # Final collapse after replacements
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_variants(text: str) -> tuple[str, list[str]]:
    """
    Remove cooking-method / portion keywords from `text`.
    Returns (stripped_text, list_of_removed_keywords).
    """
    found: list[str] = []

    def _replacer(m: re.Match) -> str:
        found.append(m.group(0).lower())
        return " "

    stripped = _VARIANT_PATTERN.sub(_replacer, text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped, found


# ---------------------------------------------------------------------------
# Search Engine
# ---------------------------------------------------------------------------

class FoodSearchEngine:
    """
    Stateless per-request search engine.
    Instantiate with a FoodRepository, then call .search().
    """

    def __init__(self, repo: FoodRepositorySync):
        self.repo = repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, raw_query: str) -> SearchResult:
        """
        Run the full matching pipeline and return a single SearchResult.

        Pipeline:
          1. Normalise
          2. Exact name match
          3. Exact alias match
          4. Variant strip → re-run steps 2-3
          5. Fuzzy name match  (score ≥ FUZZY_CONFIDENT_THRESHOLD)
          6. Fuzzy alias match (score ≥ FUZZY_CONFIDENT_THRESHOLD)
          7. Low-confidence candidates  (score ≥ FUZZY_LOW_CONFIDENCE_THRESHOLD)
          8. No match
        """
        q_norm = normalize(raw_query)

        # Helper to check distinct food noun compatibility (never substitute popcorn for leg piece, or kheer for thepla!)
        def is_compatible(q_str: str, f_doc: dict) -> bool:
            if not f_doc:
                return False
            q_norm_words = set(re.findall(r"\w+", normalize(q_str)))
            q_raw_words = set(re.findall(r"\w+", q_str.lower()))
            q_words = q_norm_words.union(q_raw_words)

            fn = f_doc.get("food_name", "").lower()
            f_words = set(re.findall(r"\w+", normalize(fn)))
            f_raw_words = set(re.findall(r"\w+", fn))
            f_all = f_words.union(f_raw_words)

            distinct_terms = {
                "leg", "drumstick", "breast", "wings", "thigh", "lolipop", "lollipop", "popcorn", "nuggets",
                "burger", "pizza", "sandwich", "roll", "frankie", "momos", "dumplings",
                "biryani", "pulao", "khichdi", "curry", "gravy", "soup",
                "roti", "rotlo", "paratha", "bhakri", "naan", "dosa", "idli", "khakhra", "fafda",
                "thepla", "kheer", "halwa", "raita", "salad", "puri", "kulcha", "bhatura", "chilla", "pancake",
                "shake", "juice", "tea", "coffee", "lassi", "chhas", "cake", "cookie", "biscuit",
                "burfi", "barfi", "ladoo", "mithai"
            }
            q_distinct = q_words.intersection(distinct_terms)
            if q_distinct and not q_distinct.intersection(f_all):
                return False

            # Dish-family incompatibility: a flatbread is never a dessert/soup/raita
            bread_terms = {"thepla", "roti", "rotlo", "paratha", "bhakri", "naan", "kulcha", "puri", "chilla"}
            dessert_terms = {"kheer", "halwa", "burfi", "barfi", "ladoo", "sweet", "mithai", "cake", "pudding"}
            liquid_terms = {"soup", "shorba", "raita", "juice", "shake"}

            if q_words.intersection(bread_terms) and f_all.intersection(dessert_terms.union(liquid_terms)):
                return False
            if q_words.intersection(dessert_terms) and f_all.intersection(bread_terms):
                return False

            heavy_types = {"sandwich", "burger", "pizza", "momos", "roll", "frankie", "taco", "burrito"}
            for ht in heavy_types:
                if ht in f_all and ht not in q_words:
                    return False
            return True

        # ── Step 0: Direct match on raw/cleaned query (preserves Indian names like methi thepla, dudhi thepla) ──
        q_raw_clean = re.sub(r"[^\w\s]", " ", raw_query.lower())
        q_raw_clean = re.sub(r"\s+", " ", q_raw_clean).strip()
        if q_raw_clean and q_raw_clean != q_norm:
            f_direct = self.repo.get_by_exact_name(q_raw_clean)
            if f_direct and is_compatible(raw_query, f_direct):
                return SearchResult(
                    query_original=raw_query,
                    query_normalized=q_norm,
                    match_type=MatchType.EXACT_NAME,
                    confidence=1.0,
                    food=f_direct,
                )
            f_direct_alias = self.repo.get_by_exact_alias(q_raw_clean)
            if f_direct_alias and is_compatible(raw_query, f_direct_alias):
                return SearchResult(
                    query_original=raw_query,
                    query_normalized=q_norm,
                    match_type=MatchType.EXACT_ALIAS,
                    confidence=1.0,
                    food=f_direct_alias,
                )

        # ── Step 1: Exact name ──────────────────────────────────────────
        food = self.repo.get_by_exact_name(q_norm)
        if food and is_compatible(raw_query, food):
            return SearchResult(
                query_original=raw_query,
                query_normalized=q_norm,
                match_type=MatchType.EXACT_NAME,
                confidence=1.0,
                food=food,
            )

        # ── Step 2: Exact alias ─────────────────────────────────────────
        food = self.repo.get_by_exact_alias(q_norm)
        if food and is_compatible(raw_query, food):
            return SearchResult(
                query_original=raw_query,
                query_normalized=q_norm,
                match_type=MatchType.EXACT_ALIAS,
                confidence=1.0,
                food=food,
            )

        # ── Step 3b: Compound food resolution: [modifier] + [base dish] ──
        # If the base dish exists in the curated DB and matches the head word of query,
        # resolve to the compound food while preserving dish family integrity.
        tokens = q_raw_clean.split()
        if len(tokens) >= 2:
            base_cand = tokens[-1]
            if base_cand in ("thepla", "paratha", "roti", "rotlo", "bhakri", "dosa", "khichdi", "rice"):
                base_food = self.repo.get_by_exact_name(base_cand)
                if not base_food:
                    base_food = self.repo.get_by_exact_alias(base_cand)
                if base_food and is_compatible(raw_query, base_food):
                    compound_name = q_raw_clean
                    compound_display = " ".join(t.capitalize() for t in tokens)
                    compound_food = dict(base_food)
                    compound_food["food_name"] = compound_name
                    compound_food["food_name_display"] = compound_display
                    return SearchResult(
                        query_original=raw_query,
                        query_normalized=q_norm,
                        match_type=MatchType.EXACT_NAME,
                        confidence=0.95,
                        food=compound_food,
                    )

        # ── Step 3c: Generic <prep>+<food> and <food>+water/pani resolver ──
        # Runs BEFORE variant strip so preparation words produce correctly-named
        # foods ("Boiled Whole Moong", "Sprouted Moong", "Moong Water") rather
        # than being silently stripped to the unlabelled base food.
        # Rules: base food must be in DB; nutrition scaled by prep multiplier.
        prep_result = self._resolve_prep_compound(raw_query, q_norm, is_compatible)
        if prep_result:
            return prep_result

        # ── Step 3: Variant strip ───────────────────────────────────────
        stripped, removed_variants = strip_variants(q_norm)
        if removed_variants and stripped:
            # Try exact matches on the stripped query
            food = self.repo.get_by_exact_name(stripped)
            if not food:
                food = self.repo.get_by_exact_alias(stripped)
            if food and is_compatible(raw_query, food):
                return SearchResult(
                    query_original=raw_query,
                    query_normalized=q_norm,
                    match_type=MatchType.VARIANT_STRIP,
                    confidence=0.95,
                    food=food,
                    variant_detected=", ".join(removed_variants),
                    stripped_query=stripped,
                )

            # If exact didn't work after stripping, try fuzzy on stripped
            result = self._fuzzy_search(raw_query, q_norm, stripped)
            if result and result.match_type in (
                MatchType.FUZZY_NAME, MatchType.FUZZY_ALIAS
            ):
                result.variant_detected = ", ".join(removed_variants)
                result.stripped_query = stripped
                return result

        # ── Step 4: Fuzzy on original normalised query ──────────────────
        result = self._fuzzy_search(raw_query, q_norm, q_norm)
        if result:
            return result

        # ── Step 5: No match ────────────────────────────────────────────
        return SearchResult(
            query_original=raw_query,
            query_normalized=q_norm,
            match_type=MatchType.NO_MATCH,
            confidence=0.0,
            message=f"No food found for '{raw_query}'. "
                    "Try a different spelling or a more specific name.",
        )

    def search_multi(self, raw_query: str, limit: int = 10) -> list[SearchResult]:
        """
        Return up to `limit` ranked candidates for a query.
        Useful for a "did you mean?" list rather than a single best match.
        """
        q_norm = normalize(raw_query)
        stripped, _ = strip_variants(q_norm)
        search_term = stripped if stripped else q_norm

        name_candidates = self._ranked_fuzzy_names(search_term, limit=limit * 2)
        alias_candidates = self._ranked_fuzzy_aliases(search_term, limit=limit * 2)

        # Merge, deduplicate by food_id, keep highest score
        seen: dict[str, float] = {}
        for food_id, score in name_candidates + alias_candidates:
            if food_id not in seen or score > seen[food_id]:
                seen[food_id] = score

        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:limit]
        foods = self.repo.get_foods_by_ids([fid for fid, _ in ranked])
        score_map = dict(ranked)

        results = []
        for food in foods:
            score = score_map.get(food["food_id"], 0.0)
            results.append(
                SearchResult(
                    query_original=raw_query,
                    query_normalized=q_norm,
                    match_type=MatchType.FUZZY_NAME,
                    confidence=round(score / 100, 2),
                    food=food,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    # Generic preparation compound resolver
    # ------------------------------------------------------------------

    # Preparation words after normalize() → (English label, calorie multiplier).
    # multiplier is applied to the base food's calories/protein/carbs/fat.
    # boiled/soaked legumes are close to raw; water is very dilute.
    _PREP_MAP: dict[str, tuple[str, float]] = {
        "boiled":   ("Boiled",    1.0),
        "soaked":   ("Soaked",    0.9),
        "sprouted": ("Sprouted",  0.85),
        "steamed":  ("Steamed",   1.0),
        "roasted":  ("Roasted",   1.05),
        "fried":    ("Fried",     1.4),
    }

    # Suffixes that mean "water/broth of <food>" — parsed AFTER normalization.
    _WATER_SUFFIXES = frozenset({"nu pani", "ka pani", "ki pani", "water", "pani"})

    # Suffixes that mean "vegetable dish / sabji of <food>" — normalized forms of
    # Gujarati 'nu shak', 'ni shak', Hindi 'ki sabji', 'ka saag', etc.
    # 'sabji' is what 'shak'/'shaak' normalizes to via TRANSLITERATION_MAP.
    # These are NOT prep words — they indicate a cooked vegetable/dal preparation.
    _SABJI_SUFFIXES = frozenset({
        "nu sabji", "ni sabji", "na sabji",    # Gujarati genitive + sabji
        "ka sabji", "ki sabji", "ke sabji",    # Hindi genitive + sabji
        "sabji",                               # bare (after stripping nu/ni)
        "nu shaak", "ni shaak",                # raw Gujarati (pre-normalize)
        "nu shak",  "ni shak",                 # common short form
    })

    def _resolve_prep_compound(
        self,
        raw_query: str,
        q_norm: str,
        is_compatible,
    ) -> Optional["SearchResult"]:
        """
        Generic resolver for two patterns:

        Pattern A — preparation + food:
          "boiled whole moong"  →  Boiled Whole Moong
          "sprouted moong dal"  →  Sprouted Moong Dal
          (prep word is the FIRST token after normalization)

        Pattern B — food + water/pani suffix:
          "moong dal nu pani"   →  Moong Water
          "chana nu pani"       →  Chana Water
          (last 1–2 tokens are a water suffix)

        Pattern C — prep + food + water suffix (combined):
          "boiled chana nu pani" → Boiled Chana Water

        Rules:
          • Base food must resolve to a verified DB record.
          • Nutrition is scaled by preparation multiplier (water = 0.05 of base).
          • display_name is "<Prep> <Base>" or "<Base> Water".
          • Never fires if the full normalized query already matched (steps 1-2).
        """
        tokens = q_norm.split()
        if len(tokens) < 2:
            return None

        # ── Detect sabji/shak suffix (Pattern D) ───────────────────────
        # "<food> nu shak" / "<food> ki sabji" → "<Food> Sabji"
        # Checked BEFORE water so "mag nu sabji" doesn't mis-fire as water.
        is_sabji = False
        base_tokens = tokens
        for suf_len in (2, 1):
            suf = " ".join(tokens[-suf_len:])
            if suf in self._SABJI_SUFFIXES:
                is_sabji = True
                base_tokens = tokens[:-suf_len]
                break
        # Also match bare trailing 'sabji' after stripping particles
        if not is_sabji and tokens[-1] == "sabji" and len(tokens) >= 2:
            is_sabji = True
            base_tokens = tokens[:-1]

        # ── Detect water/pani suffix (Pattern B or C) ──────────────────
        is_water = False
        if not is_sabji:
            for suf_len in (2, 1):           # try 2-word suffix first ("nu pani")
                suf = " ".join(tokens[-suf_len:])
                if suf in self._WATER_SUFFIXES:
                    is_water = True
                    base_tokens = tokens[:-suf_len]
                    break

        if not base_tokens:
            return None

        # ── Detect leading prep word (Pattern A or C) ──────────────────
        prep_label = None
        cal_mult   = 1.0
        food_tokens = base_tokens
        if base_tokens[0] in self._PREP_MAP:
            prep_label, cal_mult = self._PREP_MAP[base_tokens[0]]
            food_tokens = base_tokens[1:]

        if not food_tokens:
            return None

        # At least one token must remain that isn't just stop words
        _stops = {"dal", "nu", "ni", "na", "ka", "ki", "ke"}
        content = [t for t in food_tokens if t not in _stops]
        if not content:
            return None

        # Must have at least a prep OR water OR sabji suffix to justify this resolver
        if not is_water and not is_sabji and prep_label is None:
            return None

        # ── Resolve base food from DB ───────────────────────────────────
        base_q  = " ".join(food_tokens)
        base_q2 = " ".join(content)    # without connective particles

        base_food = None
        _unverified_fallback = None
        for candidate in [base_q, base_q2]:
            if not candidate:
                continue
            f = self.repo.get_by_exact_name(candidate)
            if not f:
                f = self.repo.get_by_exact_alias(candidate)
            if f and is_compatible(candidate, f):
                if f.get("is_verified"):
                    base_food = f
                    break
                elif _unverified_fallback is None:
                    _unverified_fallback = f

        # Extended lookup: try token subsets and reversed order.
        # Prefers verified curated DB foods over USDA/unverified entries so
        # nutritional profiles are more representative.
        if not base_food:
            extras = list(content)                               # individual tokens
            if len(content) >= 2:
                extras.append(" ".join(reversed(content)))      # "dal moong" from ["moong","dal"]
            for candidate in extras:
                if not candidate:
                    continue
                f = self.repo.get_by_exact_name(candidate)
                if not f:
                    f = self.repo.get_by_exact_alias(candidate)
                if f and is_compatible(candidate, f):
                    # Prefer verified foods; keep looking if this one is unverified
                    if f.get("is_verified"):
                        base_food = f
                        break
                    elif _unverified_fallback is None:
                        _unverified_fallback = f   # hold, keep looking for verified

        # Use best verified food, fall back to unverified only if nothing else found
        if not base_food:
            base_food = _unverified_fallback

        if not base_food:
            return None

        # ── Build synthetic food doc ────────────────────────────────────
        combined = dict(base_food)
        base_disp = base_food.get("food_name_display") or base_food.get("food_name", "")

        if is_sabji:
            # Cooked vegetable/dal sabji — moderate calorie density (~60-80 kcal/100g).
            # Use the base food's protein and fiber profile but scale calories to
            # represent a typical Indian home-cooked preparation with tempering.
            # A sabji serving is ~150g (1 katori).
            sabji_serving_g = 150.0
            # Scale base food macros to 100g equivalent first, then to 150g serving.
            base_srv = base_food.get("serving_size_g") or 100.0
            scale    = 100.0 / base_srv          # per-100g factors from base

            base_cal_100g = base_food.get("calories_per_100g") or (
                base_food.get("calories_kcal", 0) * scale)
            # Cooked sabji calorie range: 50-90 kcal/100g (add oil/tempering ~20 kcal)
            sabji_cal_100g = min(max(base_cal_100g * 0.8 + 20, 50), 90)

            display  = f"{base_disp} Sabji"
            name     = f"{base_food['food_name']} sabji"
            combined["food_name"]          = name
            combined["food_name_display"]  = display
            combined["serving_size_g"]     = sabji_serving_g
            combined["calories_kcal"]      = round(sabji_cal_100g * sabji_serving_g / 100, 1)
            combined["calories_per_100g"]  = round(sabji_cal_100g, 1)
            combined["protein_g"]          = round(base_food.get("protein_g", 0) * scale * sabji_serving_g / 100, 1)
            combined["carbs_g"]            = round(base_food.get("carbs_g",   0) * scale * sabji_serving_g / 100, 1)
            combined["fat_g"]              = round(base_food.get("fat_g",     0) * scale * sabji_serving_g / 100 + 3.0, 1)  # +3g for tempering oil
            combined["fiber_g"]            = round(base_food.get("fiber_g",   0) * scale * sabji_serving_g / 100, 1)
            variant_tag = "sabji"

        elif is_water:
            # Water is a very dilute broth — ~5% of base macros per 100 ml
            water_mult = 0.05
            display    = f"{prep_label + ' ' if prep_label else ''}{base_disp} Water"
            name       = f"{prep_label.lower() + ' ' if prep_label else ''}{base_food['food_name']} water"
            combined["calories_kcal"]    = round(base_food.get("calories_kcal",    0) * water_mult, 1)
            combined["calories_per_100g"]= round(base_food.get("calories_per_100g",0) * water_mult, 1)
            combined["protein_g"]        = round(base_food.get("protein_g",        0) * water_mult, 1)
            combined["carbs_g"]          = round(base_food.get("carbs_g",          0) * water_mult, 1)
            combined["fat_g"]            = round(base_food.get("fat_g",            0) * water_mult, 1)
            combined["fiber_g"]          = round(base_food.get("fiber_g",          0) * water_mult, 1)
            variant_tag = "water"
            combined["food_name"]         = name
            combined["food_name_display"] = display
        else:
            display = f"{prep_label} {base_disp}"
            name    = f"{prep_label.lower()} {base_food['food_name']}"
            combined["calories_kcal"]    = round(base_food.get("calories_kcal",    0) * cal_mult, 1)
            combined["calories_per_100g"]= round(base_food.get("calories_per_100g",0) * cal_mult, 1)
            combined["protein_g"]        = round(base_food.get("protein_g",        0) * cal_mult, 1)
            combined["carbs_g"]          = round(base_food.get("carbs_g",          0) * cal_mult, 1)
            combined["fat_g"]            = round(base_food.get("fat_g",            0) * cal_mult, 1)
            variant_tag = prep_label.lower()
            combined["food_name"]         = name
            combined["food_name_display"] = display
        # Preserve original food_id so calorie lookups work
        # (food_id stays the base food's id — nutrition is scaled above)

        return SearchResult(
            query_original=raw_query,
            query_normalized=q_norm,
            match_type=MatchType.VARIANT_STRIP,
            confidence=0.95,
            food=combined,
            variant_detected=variant_tag,
            stripped_query=" ".join(food_tokens),
        )

    # ------------------------------------------------------------------

    def _fuzzy_search(
        self,
        raw_query: str,
        q_norm: str,
        search_term: str,
    ) -> Optional[SearchResult]:
        """
        Run fuzzy matching against names then aliases.
        Returns a SearchResult if confident enough, otherwise None
        (caller may then build a LOW_CONFIDENCE result).
        """
        q_words = set(re.findall(r"\w+", q_norm))
        distinct_terms = {
            "leg", "drumstick", "breast", "wings", "thigh", "lolipop", "lollipop", "popcorn", "nuggets",
            "burger", "pizza", "sandwich", "roll", "frankie", "momos", "dumplings",
            "biryani", "pulao", "khichdi", "curry", "gravy", "soup",
            "roti", "rotlo", "paratha", "bhakri", "naan", "dosa", "idli", "khakhra", "fafda"
        }
        q_distinct = q_words.intersection(distinct_terms)

        heavy_types = {"sandwich", "burger", "pizza", "momos", "roll", "frankie", "taco", "burrito"}

        # Token-overlap guard: a fuzzy match must share a real word stem with the
        # query. token_sort_ratio can score an unrelated multi-word candidate very
        # highly off a single partial token (e.g. query "aje bhakri" scoring 91
        # against "aj amarillo yellow chile pepper"). Requiring a genuine token
        # overlap keeps fuzzy matching useful for typos while rejecting
        # semantically unrelated results — generic, no per-food rules.
        _search_words = {w for w in re.findall(r"\w+", (search_term or "").lower()) if len(w) >= 3}
        _query_words = {w for w in q_words if len(w) >= 3} | _search_words

        def _shares_token(f_doc: dict) -> bool:
            if not _query_words:
                return True  # nothing meaningful to compare — don't block
            f_words = {w for w in re.findall(r"\w+", (f_doc.get("food_name") or "").lower()) if len(w) >= 3}
            if not f_words:
                return True
            for qw in _query_words:
                for fw in f_words:
                    # exact token, or one is a prefix of the other (handles
                    # transliteration endings: bhakri/bhakhri, chai/cha)
                    if qw == fw:
                        return True
                    shorter, longer = (qw, fw) if len(qw) <= len(fw) else (fw, qw)
                    if len(shorter) >= 3 and longer.startswith(shorter):
                        return True
            return False

        def _tightest(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
            """
            Re-rank equally-scoring hits so the most specific (fewest extra
            words) candidate wins. Prevents a verbose branded entry such as
            "silk chai, soymilk" from beating the plain canonical "chai".
            Ordering is stable: score first, then fewer extra tokens.
            """
            def _extra_tokens(fid: str) -> int:
                doc = self.repo.get_by_id(fid) or {}
                return len(re.findall(r"\w+", (doc.get("food_name") or "")))
            return sorted(hits, key=lambda h: (-h[1], _extra_tokens(h[0])))

        # -- Fuzzy on food names --
        name_hits = self._ranked_fuzzy_names(search_term, limit=5)
        for best_id, best_score in _tightest(name_hits):
            if best_score >= FUZZY_CONFIDENT_THRESHOLD:
                food = self.repo.get_by_id(best_id)
                if food:
                    f_words = set(re.findall(r"\w+", normalize(food.get("food_name", ""))))
                    if q_distinct and not q_distinct.intersection(f_words):
                        continue
                    if any(ht in f_words and ht not in q_words for ht in heavy_types):
                        continue
                    if not _shares_token(food):
                        continue
                    return SearchResult(
                        query_original=raw_query,
                        query_normalized=q_norm,
                        match_type=MatchType.FUZZY_NAME,
                        confidence=round(best_score / 100, 2),
                        food=food,
                    )

        # -- Fuzzy on aliases --
        alias_hits = self._ranked_fuzzy_aliases(search_term, limit=5)
        for best_id, best_score in _tightest(alias_hits):
            if best_score >= FUZZY_CONFIDENT_THRESHOLD:
                food = self.repo.get_by_id(best_id)
                if food:
                    f_words = set(re.findall(r"\w+", normalize(food.get("food_name", ""))))
                    if q_distinct and not q_distinct.intersection(f_words):
                        continue
                    if any(ht in f_words and ht not in q_words for ht in heavy_types):
                        continue
                    if not _shares_token(food):
                        continue
                    return SearchResult(
                        query_original=raw_query,
                        query_normalized=q_norm,
                        match_type=MatchType.FUZZY_ALIAS,
                        confidence=round(best_score / 100, 2),
                        food=food,
                    )

        # -- Low-confidence: gather top candidates --
        all_hits: dict[str, float] = {}
        for food_id, score in name_hits + alias_hits:
            if score >= FUZZY_LOW_CONFIDENCE_THRESHOLD:
                if food_id not in all_hits or score > all_hits[food_id]:
                    all_hits[food_id] = score

        if all_hits:
            ranked = sorted(all_hits.items(), key=lambda x: x[1], reverse=True)
            ranked = ranked[:MAX_CLARIFICATION_CANDIDATES]
            foods = self.repo.get_foods_by_ids([fid for fid, _ in ranked])
            score_map = dict(ranked)

            # Drop semantically unrelated candidates so a clarification prompt
            # never offers something like "Amarillo Yellow Chile Pepper".
            related = [f for f in foods if _shares_token(f)]
            if related:
                foods = related

            candidates = [
                {
                    "food_id": f["food_id"],
                    "food_name": f["food_name"],
                    "food_name_display": f["food_name_display"],
                    "confidence": round(score_map[f["food_id"]] / 100, 2),
                    "calories_kcal": f["calories_kcal"],
                    "serving_size_g": f["serving_size_g"],
                }
                for f in foods
            ]
            return SearchResult(
                query_original=raw_query,
                query_normalized=q_norm,
                match_type=MatchType.LOW_CONFIDENCE,
                confidence=round(ranked[0][1] / 100, 2),
                candidates=candidates,
                message=(
                    f"Did you mean one of these? "
                    f"Input '{raw_query}' is ambiguous."
                ),
            )

        return None  # caller will produce NO_MATCH

    def _ranked_fuzzy_names(
        self, query: str, limit: int = 10
    ) -> list[tuple[str, float]]:
        """
        Score all food names with rapidfuzz and return top-`limit`
        as [(food_id, score), ...] sorted descending.
        """
        candidates = self.repo.get_all_names_with_ids()
        if not candidates:
            return []

        # Build {name: food_id} map for lookup after scoring
        name_to_id: dict[str, str] = {}
        for name, fid in candidates:
            # keep last if duplicate name (shouldn't happen but be safe)
            name_to_id[name] = fid

        names = list(name_to_id.keys())

        # Use token_sort_ratio: prevents generic single-word foods from over-matching multi-word queries
        hits = rf_process.extract(
            query,
            names,
            scorer=fuzz.token_sort_ratio,
            limit=limit,
        )
        # hits: list of (match_string, score, index)
        return [(name_to_id[h[0]], h[1]) for h in hits]

    def _ranked_fuzzy_aliases(
        self, query: str, limit: int = 10
    ) -> list[tuple[str, float]]:
        """
        Score all aliases with rapidfuzz and return top-`limit`
        as [(food_id, score), ...] sorted descending.
        Deduplicates: if multiple aliases map to the same food_id,
        keeps the highest score.
        """
        candidates = self.repo.get_all_aliases_with_ids()
        if not candidates:
            return []

        alias_to_id: dict[str, str] = {}
        for alias, fid in candidates:
            alias_to_id[alias] = fid

        aliases = list(alias_to_id.keys())

        hits = rf_process.extract(
            query,
            aliases,
            scorer=fuzz.token_sort_ratio,
            limit=limit * 3,  # over-fetch to handle dedup
        )

        # Deduplicate: best score per food_id
        id_to_best: dict[str, float] = {}
        for match_str, score, _ in hits:
            fid = alias_to_id[match_str]
            if fid not in id_to_best or score > id_to_best[fid]:
                id_to_best[fid] = score

        ranked = sorted(id_to_best.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]
