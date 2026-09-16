"""
Quantity Parser
Parses user food-quantity input into a structured (amount, unit) pair.

Supported input patterns:
  "150g"           → (150.0, "g")
  "2 bowls"        → (2.0, "bowl")
  "1.5 cups"       → (1.5, "cup")
  "3 pieces"       → (3.0, "piece")
  "200 ml"         → (200.0, "ml")
  "1 serving"      → (1.0, "serving")
  "half bowl"      → (0.5, "bowl")
  "2"              → (2.0, "serving")  (default unit)
  "one plate"      → (1.0, "plate")
  ""               → (1.0, "serving")  (default fallback)

Fractional words: half, quarter, one-and-a-half, double, triple
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuantity:
    """Parsed quantity from user input."""
    amount: float
    unit: str           # normalised: g | ml | piece | bowl | cup | plate | serving | katori | glass | spoon | slice
    raw_input: str
    confidence: float   # 0.0–1.0 (how confident we are about the parse)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Word → number mapping (supports Hindi/English common words)
WORD_NUMBERS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1,
    # Hindi numerals (transliterated)
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5,
    "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
}

# Fractional words
FRACTION_WORDS: dict[str, float] = {
    "half": 0.5,
    "quarter": 0.25,
    "three-quarter": 0.75,
    "three quarter": 0.75,
    "one and a half": 1.5,
    "one-and-a-half": 1.5,
    "double": 2.0,
    "triple": 3.0,
    "dedh": 1.5,      # Hindi: 1.5
    "dhai": 2.5,      # Hindi: 2.5
    "aadha": 0.5,     # Hindi: half
    "paav": 0.25,     # Hindi: quarter
}

# Unit aliases → canonical unit
UNIT_ALIASES: dict[str, str] = {
    # grams
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "grm": "g", "grms": "g",
    # millilitres
    "ml": "ml", "mls": "ml", "millilitre": "ml", "millilitres": "ml",
    "milliliter": "ml", "milliliters": "ml",
    # pieces
    "piece": "piece", "pieces": "piece", "pcs": "piece", "pc": "piece",
    "nos": "piece", "no": "piece", "unit": "piece", "units": "piece",
    # bowl
    "bowl": "bowl", "bowls": "bowl", "katori": "katori", "katoris": "katori",
    # cup
    "cup": "cup", "cups": "cup",
    # plate
    "plate": "plate", "plates": "plate",
    # glass
    "glass": "glass", "glasses": "glass",
    # serving
    "serving": "serving", "servings": "serving", "serve": "serving",
    # spoon
    "tbsp": "tablespoon", "tablespoon": "tablespoon", "tablespoons": "tablespoon",
    "tsp": "teaspoon", "teaspoon": "teaspoon", "teaspoons": "teaspoon",
    "spoon": "tablespoon", "spoons": "tablespoon",
    # slice
    "slice": "slice", "slices": "slice",
    # roti/chapatti specific
    "roti": "piece", "rotis": "piece",
    "chapatti": "piece", "chapattis": "piece",
    "paratha": "piece", "parathas": "piece",
    "idli": "piece", "idlis": "piece",
    "dosa": "piece", "dosas": "piece",
    "puri": "piece", "puris": "piece",
    "vada": "piece", "vadas": "piece",
    # Flatbreads & regional items
    "rotlo": "rotlo", "rotla": "rotlo",
    "bhakri": "piece", "bhakhri": "piece",
    "thepla": "piece", "theplas": "piece",
    # Countable foods
    "samosa": "piece", "samosas": "piece",
    "egg": "piece", "eggs": "piece",
    "banana": "piece", "bananas": "piece",
    "apple": "piece", "apples": "piece",
    "date": "piece", "dates": "piece",
    "biscuit": "piece", "biscuits": "piece",
    "cookie": "piece", "cookies": "piece",
    "ladoo": "piece", "ladoos": "piece", "laddu": "piece", "laddus": "piece",
}

# Default unit when only a number is given
DEFAULT_UNIT = "serving"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Regex patterns (ordered by specificity)
# Pattern 1: number directly followed by unit (e.g., "150g", "200ml")
_PAT_NUM_UNIT_NOSPACE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(g|gm|gms|gram|grams|ml|mls)\b",
    re.IGNORECASE,
)

# Pattern 2: number + space + unit word (e.g., "2 bowls", "1.5 cups")
_PAT_NUM_SPACE_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s+([a-z]+)",
    re.IGNORECASE,
)

# Pattern 3: fraction word + unit (e.g., "half bowl", "quarter plate")
_PAT_FRACTION_UNIT = re.compile(
    r"(half|quarter|double|triple|aadha|paav|dedh|dhai)\s+([a-z]+)",
    re.IGNORECASE,
)

# Pattern 4: word number + unit (e.g., "two bowls", "one serving", "ek plate")
_PAT_WORD_NUM_UNIT = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|ek|do|teen|char|paanch)\b\s+([a-z]+)",
    re.IGNORECASE,
)

# Pattern 5: bare number (e.g., "2", "1.5")
_PAT_BARE_NUMBER = re.compile(r"^(\d+(?:\.\d+)?)$")

# Pattern 6: "one and a half" style
_PAT_COMPOUND_FRACTION = re.compile(
    r"(one and a half|one-and-a-half|three quarter|three-quarter)\s*([a-z]*)",
    re.IGNORECASE,
)


def parse_quantity(raw: str) -> ParsedQuantity:
    """
    Parse a user quantity string into (amount, unit).

    Returns a ParsedQuantity with confidence:
      1.0 — unambiguous parse (explicit number + recognized unit)
      0.8 — inferred (e.g., bare number → assumes 'serving')
      0.5 — partial parse or heuristic fallback
    """
    text = raw.strip().lower()

    if not text:
        return ParsedQuantity(
            amount=1.0, unit=DEFAULT_UNIT, raw_input=raw, confidence=0.5
        )

    # ── Compound fractions first (e.g., "one and a half bowl") ──────────
    m = _PAT_COMPOUND_FRACTION.search(text)
    if m:
        amount = FRACTION_WORDS.get(m.group(1).lower(), 1.0)
        unit_raw = m.group(2).strip() if m.group(2) else ""
        unit = _resolve_unit(unit_raw)
        return ParsedQuantity(amount=amount, unit=unit, raw_input=raw, confidence=1.0)

    # ── Pattern 1: "150g", "200ml" ─────────────────────────────────────
    m = _PAT_NUM_UNIT_NOSPACE.search(text)
    if m:
        amount = float(m.group(1))
        unit = _resolve_unit(m.group(2))
        return ParsedQuantity(amount=amount, unit=unit, raw_input=raw, confidence=1.0)

    # ── Pattern 3: fraction word + unit ────────────────────────────────
    m = _PAT_FRACTION_UNIT.search(text)
    if m:
        amount = FRACTION_WORDS.get(m.group(1).lower(), 1.0)
        unit = _resolve_unit(m.group(2))
        return ParsedQuantity(amount=amount, unit=unit, raw_input=raw, confidence=1.0)

    # ── Pattern 4: word number + unit ──────────────────────────────────
    m = _PAT_WORD_NUM_UNIT.search(text)
    if m:
        amount = WORD_NUMBERS.get(m.group(1).lower(), 1.0)
        unit = _resolve_unit(m.group(2))
        return ParsedQuantity(amount=amount, unit=unit, raw_input=raw, confidence=1.0)

    # ── Check for known dish/brand names containing numbers ─────────────
    KNOWN_NUMBERED_DISHES = {"cheezy 7", "cheezy-7", "7 cheese", "7-cheese", "7up", "7 up", "5 star", "3 in 1"}
    if any(d in text for d in KNOWN_NUMBERED_DISHES):
        return ParsedQuantity(amount=1.0, unit=DEFAULT_UNIT, raw_input=raw, confidence=0.8)

    # ── Pattern 2: "2 bowls", "1.5 cups" ──────────────────────────────
    m = _PAT_NUM_SPACE_UNIT.search(text)
    if m:
        unit_candidate = m.group(2).strip().lower()
        if unit_candidate in UNIT_ALIASES:
            amount = float(m.group(1))
            unit = _resolve_unit(unit_candidate)
            return ParsedQuantity(amount=amount, unit=unit, raw_input=raw, confidence=1.0)

    # ── Pattern 5: bare number anywhere in text (e.g., "2", "2 ghee bhakri") ────
    m_any_num = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if m_any_num:
        amount = float(m_any_num.group(1))
        after_text = text[m_any_num.end():].strip()
        words = after_text.split()
        unit = DEFAULT_UNIT
        found_unit = False
        for w in words:
            cw = re.sub(r"[^\w]", "", w.lower())
            if cw in UNIT_ALIASES:
                unit = _resolve_unit(cw)
                found_unit = True
                break
        return ParsedQuantity(
            amount=amount, unit=unit, raw_input=raw, confidence=0.9 if (found_unit or not words) else 0.8
        )

    # ── Fallback: check if it's just a unit word (implies 1 × unit) ────
    unit = _resolve_unit(text)
    if unit != DEFAULT_UNIT:
        return ParsedQuantity(amount=1.0, unit=unit, raw_input=raw, confidence=0.8)

    # ── Unrecognized → no explicit quantity found ─────────────────────
    return ParsedQuantity(
        amount=None, unit=None, raw_input=raw, confidence=0.0
    )


def _resolve_unit(raw_unit: str) -> str:
    """Map a raw unit string to its canonical form."""
    raw_unit = raw_unit.strip().lower()
    if not raw_unit:
        return DEFAULT_UNIT
    return UNIT_ALIASES.get(raw_unit, DEFAULT_UNIT)
