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
    "chaha": "chai",
    # curd / yogurt
    "yoghurt": "yogurt",
    # rice / chawal
    "chawal": "rice",
    "chaval": "rice",
    "bhat": "rice",
    "bhaat": "rice",
    # buttermilk — map to the canonical local food_name "chhas" (a real DB
    # entry) rather than the generic word "buttermilk", which can collide with a
    # branded packaged product learned from an external API.
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
    "phanagavela":  "soaked",
    "phanagaveli":  "soaked",
    "phanagavelu":  "soaked",
    "phangavela":   "soaked",   # common short form
    "phangaveli":   "soaked",
    "bhajavela":    "fried",    # Gujarati fried
    "bhajeli":      "fried",
    "shekela":      "roasted",  # Gujarati roasted
    "sekela":       "roasted",
    "dhokala": "dhokla",
    "dhoklaa": "dhokla",
    "undhyu": "undhiyu",
    "ringna": "brinjal",
    "ringan": "brinjal",
    "dudhi": "bottle gourd",
    "tindola": "ivy gourd",
    "karela": "bitter gourd",
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
      1. Unicode NFKD → strip accents
      2. Lowercase
      3. Clean non-alphanumeric noise characters (convert punctuation like '=' to spaces)
      4. Collapse whitespace
      5. Apply transliteration map (word-level)
    """
    # 0. Native-script (Devanagari / Gujarati) -> Roman FIRST.
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
