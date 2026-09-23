"""
Exercise Calorie Calculator
Fuzzy exercise search + natural input parsing + min/avg/max calorie output.

Pipeline:
  User input ("jogging 30 minutes")
    → parse exercise name + duration/amount
    → fuzzy match exercise in DB
    → calculate calories (min / avg / max)
    → return result with confidence

Supports:
  - Minutes:  "30 minutes", "45 min"
  - Reps:     "50 reps", "3 sets of 15"
  - Rounds:   "10 rounds"
  - Laps:     "20 laps"
  - Sessions: "1 session"
  - Default:  uses exercise's typical_duration_min if no amount given
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz, process as rf_process


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reference body weight the stored `calories_per_unit_*` rates are baked at.
# Master-DB (CSV) exercise rates follow the standard ~70 kg reference used by
# the MET → kcal formula (kcal/min = MET × 3.5 × weight_kg / 200). Burn is
# rescaled from this reference to the user's actual weight in
# ExerciseCalculator.calculate(). Kept in sync with
# external_exercise_api.DEFAULT_WEIGHT_KG.
REFERENCE_WEIGHT_KG = 70.0

# Unit aliases → canonical
EXERCISE_UNIT_ALIASES: dict[str, str] = {
    "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
    "m": "minutes",
    "rep": "reps", "reps": "reps", "repetition": "reps", "repetitions": "reps",
    "round": "rounds", "rounds": "rounds",
    "lap": "laps", "laps": "laps",
    "jump": "jumps", "jumps": "jumps", "skip": "jumps", "skips": "jumps",
    "session": "session", "sessions": "session",
    "set": "sets", "sets": "sets",
    "hr": "minutes", "hour": "minutes", "hours": "minutes",
    "houre": "minutes", "houres": "minutes", "hourse": "minutes",
    "ghanta": "minutes", "ghante": "minutes", "kalak": "minutes",
    "hold": "minutes",
    # distance
    "km": "km", "kms": "km", "kilometer": "km", "kilometers": "km",
    "kilometre": "km", "kilometres": "km",
    "metre": "meters", "metres": "meters", "meter": "meters", "meters": "meters",
    "step": "steps", "steps": "steps",
}


def normalize_exercise_unit(raw_unit: Optional[str]) -> str:
    """
    Sanitize a measurement_unit string into a clean canonical plural unit.

    The exercise dataset contains malformed per-unit strings like
    "1 rep", "1 min", "1 jump", "1 min hold", "1 round (1 min)",
    "1 rep (both legs)", "1 lap (25m)". These embed a count and extra text.
    This strips leading counts / parenthetical notes and maps the core word
    to a canonical unit so the UI shows "20 reps", never "20 1 rep".
    """
    if not raw_unit:
        return "reps"
    u = str(raw_unit).lower().strip()
    # Drop parenthetical qualifiers e.g. "1 round (1 min)" → "1 round"
    u = re.sub(r"\(.*?\)", " ", u)
    # Drop leading count e.g. "1 rep" → "rep", "1 min hold" → "min hold"
    u = re.sub(r"^\s*\d+(?:\.\d+)?\s*", "", u)
    u = re.sub(r"\s+", " ", u).strip()
    if not u:
        return "reps"
    # Map the first meaningful token to a canonical unit.
    first = u.split()[0]
    canonical = EXERCISE_UNIT_ALIASES.get(first)
    if canonical:
        return canonical
    # Whole-string alias (e.g. "min hold" → not in map, first token "min" is)
    canonical = EXERCISE_UNIT_ALIASES.get(u)
    if canonical:
        return canonical
    # Fallback: pluralise a bare singular, else return as-is.
    if u.endswith("s"):
        return u
    return u + "s"

# Word numbers — English + Hindi/Gujarati Romanised
WORD_NUMBERS: dict[str, float] = {
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    "twenty-five": 25, "thirty": 30,
    "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60,
    "half": 0.5,
    # Hindi (Romanised) — covers voice/STT input that wasn't normalised
    "ek": 1, "do": 2, "teen": 3, "char": 4,
    "paanch": 5, "panch": 5, "chhe": 6, "saat": 7, "aath": 8,
    "nau": 9, "das": 10,
    "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14,
    "pandrah": 15, "pandraha": 15, "pandara": 15,
    "solah": 16, "satrah": 17, "atharah": 18, "unnis": 19,
    "bees": 20, "pachees": 25, "pachis": 25,
    "tees": 30, "chaalees": 40, "pachaas": 50,
    # Gujarati (Romanised)
    "tran": 3, "nav": 9,
    "agiyar": 11, "ter": 13, "chaud": 14,
    "pandar": 15, "sol": 16, "ognis": 19, "vis": 20,
    "ekvis": 21, "chovis": 24, "pachvis": 25,
    "trish": 30, "chaalis": 40, "pachas": 50,
}

FUZZY_THRESHOLD = 70  # minimum score to accept a match


# ---------------------------------------------------------------------------
# Parsed input
# ---------------------------------------------------------------------------

@dataclass
class ParsedExerciseInput:
    """Parsed from user's natural-language exercise input."""
    exercise_query: str      # the exercise name part
    amount: Optional[float]  # parsed number (None = use default)
    unit: Optional[str]      # parsed unit (None = use exercise's native unit)
    raw_input: str


@dataclass
class ExerciseCalcResult:
    """Result of exercise calorie calculation."""
    exercise_id: str
    exercise_name: str
    exercise_name_display: str
    category: str

    # Input
    amount: float
    unit: str              # actual unit used (exercise's native unit)
    input_amount: float    # what user asked for
    input_unit: str        # what user specified

    # Calories
    calories_min: float
    calories_avg: float
    calories_max: float

    # Metadata
    search_confidence: float
    typical_duration_min: Optional[float]
    calculation_note: str


@dataclass
class WorkoutExerciseItem:
    """Single exercise item within a structured workout routine."""
    muscle_group: str
    exercise_name: str
    exercise_count: int = 1
    sets: Optional[int] = None
    reps: Optional[int] = None
    weight_kg: Optional[float] = None
    duration_min: Optional[float] = None


@dataclass
class ParsedWorkoutRoutine:
    """Structured workout routine containing multiple exercises/muscle groups."""
    items: list[WorkoutExerciseItem]
    total_exercises: int
    total_sets: int
    total_reps: int
    duration_min: Optional[float]
    raw_input: str

    @property
    def summary_text(self) -> str:
        parts = []
        for it in self.items:
            part = f"{it.muscle_group.title()} ({it.exercise_count} exercises"
            if it.sets and it.reps:
                part += f" × {it.sets} sets × {it.reps} reps)"
            elif it.reps:
                part += f" × {it.reps} reps)"
            elif it.sets:
                part += f" × {it.sets} sets)"
            else:
                part += ")"
            parts.append(part)
        return ", ".join(parts)


@dataclass
class RoutineCalcResult:
    """Calorie and display result for a structured workout routine."""
    routine: ParsedWorkoutRoutine
    calories_min: float
    calories_avg: float
    calories_max: float
    summary_text: str
    duration_min: Optional[float]
    total_reps: int


# ---------------------------------------------------------------------------
# Input parser
# ---------------------------------------------------------------------------

# Pattern: "3 sets of 15" → total reps = 3 × 15 = 45
_PAT_SETS = re.compile(
    r"(\d+)\s*(?:sets?)\s*(?:of|x|×)\s*(\d+)",
    re.IGNORECASE,
)

# Pattern: number + unit at end or start (incl. distance km / metres)
_PAT_NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|reps?|rounds?|laps?|jumps?|skips?|sessions?|sets?|hr|hours?|houres?|hourse|ghanta|ghante|kalak|"
    r"km|kms|kilometer|kilometers|kilometre|kilometres|metre|metres|meter|meters|steps?)\b",
    re.IGNORECASE,
)

# Pattern: "for X minutes/reps" at end
_PAT_FOR_DURATION = re.compile(
    r"\bfor\s+(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|reps?|rounds?|laps?|hours?|houres?|hourse|hrs?|ghanta|ghante|kalak|km|kms|kilometers?)\b",
    re.IGNORECASE,
)

# Pattern: word number + unit (English + Hindi/Gujarati number words)
_WORD_NUMS_RE = (
    # English
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"twenty-five|thirty|forty|forty-five|fifty|sixty|half|"
    # Hindi (Romanised)
    r"ek|do|teen|char|paanch|panch|chhe|saat|aath|nau|das|"
    r"gyarah|barah|terah|chaudah|pandrah|pandraha|pandara|"
    r"solah|satrah|atharah|unnis|bees|pachees|pachis|tees|"
    r"chaalees|pachaas|"
    # Gujarati (Romanised)
    r"be|tran|nav|agiyar|bar|ter|chaud|pandar|sol|ognis|vis|"
    r"ekvis|chovis|pachvis|trish|chaalis|pachas"
)
_PAT_WORD_NUM_UNIT = re.compile(
    r"\b(" + _WORD_NUMS_RE + r")\s+"
    r"(min(?:ute)?s?|reps?|rounds?|laps?|hours?|houres?|hourse|sessions?)\b",
    re.IGNORECASE,
)


def parse_duration_minutes(text: str) -> Optional[float]:
    """
    Parse a natural language duration string into total minutes.
    Handles:
      - Hours: "1 hour", "1 houre", "1.5 hours", "2 hrs", "0.5 hr", "2.5 ghanta", "1 kalak"
      - Minutes: "30 min", "45 mins", "45 minutes", "15 minute"
      - Seconds: "90 seconds", "90 sec" -> 1.5 min
      - Colloquial / fractions: "half an hour", "half hour", "aadha ghanta", "aadho kalak", "dedh ghanta"
      - Word numbers: "one hour", "ek ghanta", "two hours", etc.
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip().lower()

    # Pre-checks for colloquial terms
    if re.search(r"\b(?:half\s+(?:an\s+)?hour|aadha\s+ghanta|aadho\s+kalak|adho\s+kalak|adha\s+ghanta)\b", raw):
        return 30.0
    if re.search(r"\b(?:dedh|dhedh)\s+ghanta\b", raw):
        return 90.0
    if re.search(r"\b(?:dhai|adhai)\s+ghanta\b", raw):
        return 150.0

    # Hours pattern: numeric + hour unit (including typos like houre, houres, hourse)
    hr_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:hours?|houres?|hourse|hrs?|ghanta|ghante|kalak)\b",
        raw, re.I
    )
    if hr_match:
        return float(hr_match.group(1)) * 60.0

    # Minutes pattern: numeric + minute unit
    min_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:mins?|minutes?)\b",
        raw, re.I
    )
    if min_match:
        return float(min_match.group(1))

    # Seconds pattern: numeric + second unit
    sec_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:sec(?:ond)?s?)\b",
        raw, re.I
    )
    if sec_match:
        return round(float(sec_match.group(1)) / 60.0, 2)

    # Word numbers for hours: "one hour", "ek ghanta", "two hours", etc.
    word_hr_match = re.search(
        r"\b(one|two|three|four|five|ek|do|teen|char)\s+(?:hours?|houres?|hourse|hrs?|ghanta|ghante|kalak)\b",
        raw, re.I
    )
    if word_hr_match:
        w_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ek": 1, "do": 2, "teen": 3, "char": 4}
        return w_map[word_hr_match.group(1).lower()] * 60.0

    # Word numbers for minutes: "thirty minutes", "twenty mins", "pandrah minute", etc.
    word_min_match = re.search(
        r"\b(" + _WORD_NUMS_RE + r")\s+(?:mins?|minutes?)\b",
        raw, re.I
    )
    if word_min_match:
        val = WORD_NUMBERS.get(word_min_match.group(1).lower())
        if val is not None:
            return float(val)

    # Bare number check: e.g. "30", "45", "60"
    bare_match = re.fullmatch(r"^\s*(\d+(?:\.\d+)?)\s*$", raw)
    if bare_match:
        return float(bare_match.group(1))

    return None


def parse_exercise_input(raw: str) -> ParsedExerciseInput:
    """
    Parse natural language exercise input into exercise name + amount + unit.

    Examples:
      "jogging 30 minutes"    → query="jogging", amount=30, unit="minutes"
      "50 push-ups"           → query="push-ups", amount=50, unit="reps"
      "yoga"                  → query="yoga", amount=None, unit=None
      "3 sets of 12 squats"   → query="squats", amount=36, unit="reps"
      "cycling for 45 min"    → query="cycling", amount=45, unit="minutes"
      "pandrah pushups"       → query="pushups", amount=15, unit=None
    """
    text = raw.strip()
    if not text:
        return ParsedExerciseInput("", None, None, raw)

    # Pre-process: replace any Indian number words with digits so all patterns
    # below work on plain numbers. This handles typed input that bypassed STT
    # normalisation (e.g. "pandrah pushups" typed directly).
    for word, digit in sorted(WORD_NUMBERS.items(), key=lambda x: -len(x[0])):
        text = re.sub(
            r"\b" + re.escape(word) + r"\b",
            str(int(digit)) if digit == int(digit) else str(digit),
            text,
            flags=re.IGNORECASE,
        )

    # Strip noise/pronoun/verb words up front so leading context like
    # "aaj maine ..." or "today i did ..." never blocks number/name extraction.
    # (Digits and unit words are preserved — only noise tokens are removed.)
    _kept = [t for t in re.split(r"\s+", text.strip())
             if t and t.lower().strip(",.-") not in _EXERCISE_NOISE_WORDS]
    text = " ".join(_kept).strip() or text.strip()

    amount: Optional[float] = None
    unit: Optional[str] = None
    exercise_part = text

    # Try "3 sets of 15" pattern first
    m = _PAT_SETS.search(text)
    if m:
        sets_num = float(m.group(1))
        reps_num = float(m.group(2))
        amount = sets_num * reps_num
        unit = "reps"
        exercise_part = _PAT_SETS.sub("", text).strip()
        return ParsedExerciseInput(
            exercise_query=_clean_exercise_name(exercise_part),
            amount=amount, unit=unit, raw_input=raw,
        )

    # Try "for X minutes/reps" pattern
    m = _PAT_FOR_DURATION.search(text)
    if m:
        amount = float(m.group(1))
        unit = _resolve_exercise_unit(m.group(2))
        if unit == "minutes" and any(h in m.group(2).lower() for h in ("hour", "houre", "hr", "ghanta", "kalak")):
            amount *= 60
        exercise_part = _PAT_FOR_DURATION.sub("", text).strip()
        return ParsedExerciseInput(
            exercise_query=_clean_exercise_name(exercise_part),
            amount=amount, unit=unit, raw_input=raw,
        )

    # Try word number + unit ("thirty minutes")
    m = _PAT_WORD_NUM_UNIT.search(text)
    if m:
        amount = WORD_NUMBERS.get(m.group(1).lower(), 1.0)
        unit = _resolve_exercise_unit(m.group(2))
        if unit == "minutes" and any(h in m.group(2).lower() for h in ("hour", "houre", "hr", "ghanta", "kalak")):
            amount *= 60
        exercise_part = _PAT_WORD_NUM_UNIT.sub("", text).strip()
        return ParsedExerciseInput(
            exercise_query=_clean_exercise_name(exercise_part),
            amount=amount, unit=unit, raw_input=raw,
        )

    # Try number + unit anywhere
    m = _PAT_NUM_UNIT.search(text)
    if m:
        amount = float(m.group(1))
        unit = _resolve_exercise_unit(m.group(2))
        if unit == "minutes" and any(h in m.group(2).lower() for h in ("hour", "houre", "hr", "ghanta", "kalak")):
            amount *= 60
        exercise_part = _PAT_NUM_UNIT.sub("", text).strip()
        return ParsedExerciseInput(
            exercise_query=_clean_exercise_name(exercise_part),
            amount=amount, unit=unit, raw_input=raw,
        )

    # Try bare number at start or end (assume it's amount in exercise's native unit)
    bare = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", text)
    if bare:
        amount = float(bare.group(1))
        exercise_part = bare.group(2)
        return ParsedExerciseInput(
            exercise_query=_clean_exercise_name(exercise_part),
            amount=amount, unit=None, raw_input=raw,
        )

    bare_end = re.match(r"^(.+?)\s+(\d+(?:\.\d+)?)$", text)
    if bare_end:
        amount = float(bare_end.group(2))
        exercise_part = bare_end.group(1)
        return ParsedExerciseInput(
            exercise_query=_clean_exercise_name(exercise_part),
            amount=amount, unit=None, raw_input=raw,
        )

    # No amount found — just the exercise name
    return ParsedExerciseInput(
        exercise_query=_clean_exercise_name(text),
        amount=None, unit=None, raw_input=raw,
    )


def parse_workout_routine(text: str) -> Optional[ParsedWorkoutRoutine]:
    """
    Parse structured workout routines with muscle groups, exercise counts, sets, reps.
    Examples:
      - 'Chest: 4 exercises × 12 reps Shoulder: 2 exercises × 12 reps Triceps: 2 exercises × 12 reps'
      - '4 chest exercises 12 reps, 2 shoulder exercises 12 reps'
      - 'Chest 4 exercises 12 reps each'
      - 'Back 3 exercises x 10 reps, Biceps 2 exercises x 12 reps for 45 minutes'
      - '3 sets of 12 reps bench press, 4 sets of 10 reps shoulder press'
    
    IMPORTANT: Never infer workout duration from reps or exercise count.
    Only extracts duration_min when an explicit duration phrase is present (e.g. 'for 45 minutes').
    """
    if not text or not text.strip():
        return None
    raw = text.strip()

    # 1. Check for overall duration in the string (explicit only!)
    dur_min = None
    dur_match = re.search(
        r"\b(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:mins?|minutes?|hrs?|hours?|houres?|hourse|ghanta|ghante|kalak)\b",
        raw, re.I
    )
    if dur_match:
        val = float(dur_match.group(1))
        unit_str = dur_match.group(0).lower()
        if any(h in unit_str for h in ["hr", "hour", "houre", "ghanta", "kalak"]):
            val *= 60.0
        dur_min = val

    # Clean sentence for item matching: remove overall duration phrase
    clean_text = dur_match.re.sub(" ", raw) if dur_match else raw
    # Normalize multiplication signs
    clean_text = re.sub(r"[×*]", " x ", clean_text)

    # Indic digit conversion (Gujarati & Devanagari numerals)
    _indic_digits = {
        "૧": "1", "૨": "2", "૩": "3", "૪": "4", "૫": "5",
        "૬": "6", "૭": "7", "૮": "8", "૯": "9", "૦": "0",
        "१": "1", "२": "2", "३": "3", "४": "4", "५": "5",
        "६": "6", "७": "7", "८": "8", "९": "9", "०": "0",
    }
    for ind_d, num_d in _indic_digits.items():
        clean_text = clean_text.replace(ind_d, num_d)

    # Indic script & transliteration normalization
    _indic_routine_map = {
        "ચેસ્ટ": "chest", "ચેસ": "chest", "ચેસ્ટની": "chest", "ચેસ્ટના": "chest",
        "શોલ્ડર": "shoulder", "શોલ્ડર્સ": "shoulder",
        "ટ્રાઇસેપ્સ": "triceps", "ટ્રાયસેપ્સ": "triceps",
        "બાઇસેપ્સ": "biceps", "બાયસેપ્સ": "biceps",
        "બેક": "back", "લેગ્સ": "legs", "લેગ": "legs", "એબ્સ": "abs",
        "કસરત": "exercise", "કસરતો": "exercises", "રેપ્સ": "reps", "રેપ": "reps",
        "સેટ્સ": "sets", "સેટ": "sets",
        "चेस्ट": "chest", "शोल्डर": "shoulder", "शोल्डर्स": "shoulder",
        "ट्राइसेप्स": "triceps", "बाइसेप्स": "biceps", "बैक": "back",
        "लेग्स": "legs", "एब्स": "abs", "कसरत": "exercise",
        "एक्सरसाइज": "exercise", "रेप्स": "reps", "रेप": "reps",
        "सेट्स": "sets", "सेट": "sets",
        # Common typos & variations
        "excercise": "exercise", "excercises": "exercises",
        "excersise": "exercise", "excersises": "exercises",
        "excerise": "exercise", "excerises": "exercises",
        "kasarat": "exercise", "kasrat": "exercise",
        "repetition": "reps", "repetitions": "reps",
        "rap": "reps", "raps": "reps",
        "pectorals": "chest", "pectoral": "chest", "pecs": "chest",
        "deltoids": "shoulder", "deltoid": "shoulder", "delts": "shoulder",
        "quadriceps": "quads",
        "કાફ્સ": "calves", "કાફ": "calves", "કાલ્વ્સ": "calves",
        "काफ्स": "calves", "काफ": "calves",
    }
    for wrong, right in sorted(_indic_routine_map.items(), key=lambda x: -len(x[0])):
        clean_text = re.sub(r"(?i)\b" + re.escape(wrong) + r"\b", right, clean_text)

    # Strip grouping characters like parentheses, brackets, colons so e.g. "(4 exercises x 12 reps)" matches cleanly
    clean_text = re.sub(r"[\(\)\[\]\{\}:;]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    items: list[WorkoutExerciseItem] = []

    # Muscle group vocabulary
    muscles_re = (
        r"chest|shoulders?|triceps?|biceps?|back|legs?|abs|core|glutes?|lats?|arms?|"
        r"push|pull|quads?|hamstrings?|calves?|traps?|forearms?|delts?|squats?|rhomboids?"
    )

    # Check for a trailing or global reps amount (e.g. "... karyo 12 reps")
    global_reps = None
    glob_m = re.search(r"(\d+)\s*reps?\b(?:\s+(?:each|mara|karyo|karya|kiya|kiye))?", clean_text, re.I)
    if glob_m:
        global_reps = int(glob_m.group(1))

    # Patterns to match muscle routines across word orders, connectives, and languages:
    # 1. Muscle first with count: 'Chest: 4 exercises x 12 reps', 'Chest na 4 exercise 12 reps'
    pat_muscle_first_with_count = re.compile(
        rf"\b(?P<m>{muscles_re})\b(?:\s+(?:workout|training|session))?(?:\s+(?:na|ni|no|ne|ka|ke|ki|wali|of))?\s*:?\s*"
        rf"(?P<c>\d+)\s*(?:exercises?|kasrat|types?|varieties|variations)\b"
        rf"(?:\s*(?:x|of|,|-|sathe|with)?\s*(?:(?P<s>\d+)\s*sets?(?:\s*(?:of|x)\s*)?)?(?P<r>\d+)?\s*(?:reps?|each)?)?",
        re.I
    )
    # 2. Muscle count first: '4 chest exercises 12 reps', '4 chest ki exercises'
    pat_muscle_count_first = re.compile(
        rf"\b(?P<c>\d+)\s+(?P<m>{muscles_re})(?:\s+(?:ki|ni|wali|ke|na))?\s+(?:exercises?|kasrat)\b"
        rf"(?:\s*(?:x|of|,|-)?\s*(?:(?P<s>\d+)\s*sets?(?:\s*(?:of|x)\s*)?)?(?P<r>\d+)?\s*(?:reps?|each)?)?",
        re.I
    )
    # 3. Count first: '4 exercises chest 12 reps', '4 exercises of chest 12 reps'
    pat_count_first = re.compile(
        rf"\b(?P<c>\d+)\s*(?:exercises?|kasrat)\s*(?:of\s+)?(?P<m>{muscles_re})\b"
        rf"(?:\s*(?:x|of|,|-)?\s*(?:(?P<s>\d+)\s*sets?(?:\s*(?:of|x)\s*)?)?(?P<r>\d+)?\s*(?:reps?|each)?)?",
        re.I
    )
    # 4. Muscle sets x reps: 'Chest 3 sets of 12 reps', 'Chest 3 sets x 12 reps'
    pat_muscle_sets_reps = re.compile(
        rf"\b(?P<m>{muscles_re})\b(?:\s+(?:workout|training|session))?(?:\s+(?:na|ni|no|ne|ka|ke|ki|wali|of))?\s*:?\s*"
        rf"(?P<s>\d+)\s*sets?(?:\s*(?:of|x)\s*)(?P<r>\d+)\s*(?:reps?|each)?",
        re.I
    )
    # 5. Muscle first count only: 'Chest na 4 exercise', 'Shoulder 2 kasrat'
    pat_muscle_first_count_only = re.compile(
        rf"\b(?P<m>{muscles_re})\b(?:\s+(?:na|ni|ka|ke))?\s+(?P<c>\d+)\s+(?:exercises?|kasrat)\b",
        re.I
    )
    # 6. Muscle bare reps: 'Chest 12 reps', 'Chest 3 x 12 reps'
    pat_muscle_bare_reps = re.compile(
        rf"\b(?P<m>{muscles_re})\b(?:\s+(?:workout|training|session))?(?:\s+(?:na|ni|no|ne|ka|ke|ki|wali|of))?\s*:?\s*"
        rf"(?:(?P<s>\d+)\s*(?:x)\s*)?(?P<r>\d+)\s*reps?\b",
        re.I
    )

    matched_spans: list[tuple[int, int]] = []
    def _overlaps(s: int, e: int) -> bool:
        return any(max(s, ms) < min(e, me) for ms, me in matched_spans)

    patterns = [
        (pat_muscle_first_with_count, False),
        (pat_muscle_count_first, False),
        (pat_count_first, False),
        (pat_muscle_sets_reps, True),
        (pat_muscle_first_count_only, False),
        (pat_muscle_bare_reps, True),
    ]

    for pat, is_sets_or_bare in patterns:
        for m in pat.finditer(clean_text):
            if _overlaps(m.start(), m.end()):
                continue
            matched_spans.append((m.start(), m.end()))
            muscle = m.group("m").capitalize()
            if is_sets_or_bare:
                count = 1
                sets_val = int(m.group("s")) if m.groupdict().get("s") and m.group("s") else None
                reps_val = int(m.group("r")) if m.groupdict().get("r") and m.group("r") else global_reps
            else:
                c_str = m.groupdict().get("c")
                count = int(c_str) if c_str else 1
                sets_val = int(m.group("s")) if m.groupdict().get("s") and m.group("s") else None
                reps_val = int(m.group("r")) if m.groupdict().get("r") and m.group("r") else global_reps
            items.append(WorkoutExerciseItem(
                muscle_group=muscle.lower(),
                exercise_name=f"{muscle} exercises" if count > 1 else f"{muscle} exercise",
                exercise_count=count,
                sets=sets_val,
                reps=reps_val,
            ))

    if items:
        # Sort items by their order of appearance in the user input
        items.sort(key=lambda it: clean_text.lower().find(it.muscle_group.lower()))

    if not items:
        # Pattern B: Specific exercise names with sets and reps:
        # '3 sets of 12 reps bench press, 4 sets of 10 reps shoulder press'
        pat_b = re.compile(
            r"(\d+)\s*sets?(?:\s*(?:of|x)\s*)?(\d+)\s*reps?\s+([a-zA-Z\s\-]+?)(?:,|and|sathe|$)",
            re.I
        )
        for m in pat_b.finditer(clean_text):
            s_val = int(m.group(1))
            r_val = int(m.group(2))
            ex_name = m.group(3).strip()
            items.append(WorkoutExerciseItem(
                muscle_group="general",
                exercise_name=ex_name,
                exercise_count=1,
                sets=s_val,
                reps=r_val,
            ))

    if not items:
        return None

    tot_ex = sum(it.exercise_count for it in items)
    tot_sets = sum((it.sets or 1) * it.exercise_count for it in items)
    tot_reps = sum(
        (it.reps or 0) * (it.sets or 1) * it.exercise_count
        for it in items
    )

    return ParsedWorkoutRoutine(
        items=items,
        total_exercises=tot_ex,
        total_sets=tot_sets,
        total_reps=tot_reps,
        duration_min=dur_min,
        raw_input=raw,
    )


def _resolve_exercise_unit(raw: str) -> str:
    return EXERCISE_UNIT_ALIASES.get(raw.lower().strip(), raw.lower().strip())


# Noise words that are never part of an exercise name — removed anywhere in the
# string (English + Hindi/Gujarati/Punjabi context words & past-tense verbs).
_EXERCISE_NOISE_WORDS = {
    # English filler / pronouns / temporal
    "i", "we", "you", "my", "me", "did", "do", "done", "doing", "have", "had",
    "today", "yesterday", "morning", "evening", "night", "for", "of", "the",
    "a", "an", "was", "were", "am", "just", "some", "and",
    "play", "played", "playing", "perform", "performed",
    # play (gu/hi) — "khelya", "ramyo"
    "khel", "khelya", "khelyu", "ramyo", "ramya", "ramyu",
    # Hindi / Gujarati temporal & pronoun context
    "aaj", "aaje", "kal", "kale", "subah", "savare", "sanje", "raat",
    "maine", "mene", "mein", "main", "hu", "hoon", "aj",
    # Hindi / Gujarati / Punjabi past-tense "did/performed" verbs
    "kiya", "kiye", "kia", "kari", "karyu", "karya", "kar", "kara", "kare",
    "marya", "maryu", "maryo", "mari", "karela", "kareli", "karelu",
    "kita", "kiti", "kite", "lagavya", "lagavyu", "lagaya",
    # particles
    "ne", "ka", "ki", "ke", "nu", "ni", "na", "da", "di", "de", "te",
    # Gujarati/Hinglish connectors — "pn"/"pan"/"ane" = also/and/too
    # These appear in follow-up sentences like "pn me 6 set marya" (also did 6 sets)
    # and must NEVER be treated as exercise names.
    "pn", "pan", "ane", "pn me", "pan me", "ane me",
    "also", "too", "again", "more", "extra", "additional",
}


def _clean_exercise_name(text: str) -> str:
    """
    Reduce a phrase to just the exercise name by removing noise words
    (pronouns, temporal words, "did/performed" verbs, particles) ANYWHERE
    in the string — not only at the edges.

    Examples:
      "today i did  push-ups"   → "push-ups"
      "aaj maine kiya jogging"  → "jogging"
      "push-ups"                → "push-ups"
    """
    # Split into tokens, keep only non-noise words (preserve hyphenated names).
    tokens = re.split(r"\s+", text.strip())
    kept = [t for t in tokens if t and t.lower().strip(",.-") not in _EXERCISE_NOISE_WORDS]
    cleaned = " ".join(kept).strip().strip(",.-")
    # If everything was stripped (edge case), fall back to the original text.
    return cleaned if cleaned else text.strip().strip(",.-")


# ---------------------------------------------------------------------------
# Exercise Search (fuzzy)
# ---------------------------------------------------------------------------

class ExerciseSearcher:
    """Fuzzy search over the exercises collection (MongoDB, sync)."""

    def __init__(self, db):
        # Accepts a pymongo Database (sync) — uses the 'exercises' collection
        self.collection = db["exercises"]

    def search(self, query: str) -> tuple[Optional[dict], float]:
        """
        Return (exercise_dict, confidence_score) or (None, 0.0).
        Uses token_set_ratio for flexible matching.
        """
        query_lower = query.strip().lower()
        if not query_lower:
            return None, 0.0

        # Exercise typo, misspelling & transliteration map
        # Covers: common English misspellings, Indian Romanised variants,
        # and alternate spellings users type
        ex_typo_map = {
            # ── Running / Jogging ────────────────────────────────────────
            "runing":    "running",  "runnig":  "running",  "runn":    "running",
            "joging":    "jogging",  "joggin":  "jogging",  "jog":     "jogging",
            # ── Walking ──────────────────────────────────────────────────
            "walkng":    "walking",  "walkin":  "walking",  "waking":  "walking",
            "walkig":    "walking",
            # ── Cycling ──────────────────────────────────────────────────
            "cyclng":    "cycling",  "cyclin":  "cycling",  "ciclying":"cycling",
            "sycling":   "cycling",  "cicling": "cycling",  "cycle":   "cycling",
            "saikal":    "cycling",  "saikil":  "cycling",  # Gujarati/Hindi
            # ── Yoga ─────────────────────────────────────────────────────
            "yga":       "yoga",     "yog":     "yoga",     "yoag":    "yoga",
            "yoaga":     "yoga",     "yogaa":   "yoga",
            # ── Push-ups ─────────────────────────────────────────────────
            "puchups":   "pushups",  "pushup":  "pushups",
            "push up":   "pushups",  "push ups":"pushups",  "push-ups":"pushups",
            "psuhups":   "pushups",  "puhsups": "pushups",  "pushupps":"pushups",
            "pushsup":   "pushups",  "pushsups":"pushups",
            "pusups":    "pushups",  "puships": "pushups",
            "pushap":    "pushups",  "pushaps": "pushups",  # Hinglish: "push ap"
            "pus ups":   "pushups",  "pus-ups": "pushups",
            # ── Squats ───────────────────────────────────────────────────
            "squot":     "squats",   "squat":   "squats",   "squatt":  "squats",
            "squuts":    "squats",   "sqots":   "squats",   "sqats":   "squats",
            "skwats":    "squats",   "squatss": "squats",
            # ── Swimming ─────────────────────────────────────────────────
            "swiming":   "swimming", "swimig":  "swimming", "swiiming":"swimming",
            "swimm":     "swimming", "swmming": "swimming",
            # ── Plank ────────────────────────────────────────────────────
            "plak":      "plank",    "planks":  "plank",    "plnk":    "plank",
            "plaink":    "plank",
            # ── Crunches ─────────────────────────────────────────────────
            "crunch":    "crunches", "crunchs": "crunches", "crnches": "crunches",
            "cruches":   "crunches",
            # ── Sit-ups ──────────────────────────────────────────────────
            "situp":     "sit-ups",  "sit up":  "sit-ups",  "sit ups": "sit-ups",
            "situps":    "sit-ups",
            # ── Pull-ups ─────────────────────────────────────────────────
            "pullup":    "pull-ups", "pull up": "pull-ups", "pull ups":"pull-ups",
            "pullups":   "pull-ups",
            # ── Burpees ──────────────────────────────────────────────────
            "burpee":    "burpees",  "burpes":  "burpees",  "burpies": "burpees",
            "burppe":    "burpees",
            # ── Lunges ───────────────────────────────────────────────────
            "lunge":     "lunges",   "lungs":   "lunges",   "lunjes":  "lunges",
            # ── Deadlift ─────────────────────────────────────────────────
            "deadlift":  "deadlifts","deedlift":"deadlifts","detlift":  "deadlifts",
            # ── Gym / Workout ─────────────────────────────────────────────
            "workut":    "workout",  "workuot": "workout",  "work out":"workout",
            # ── Skipping / Jumping rope ───────────────────────────────────
            "skiping":   "skipping", "skippping":"skipping",
            "jumping rope":"skipping",
            # ── Stretching ────────────────────────────────────────────────
            "strechting":"stretching","streching":"stretching","stretchig":"stretching",
            # ── Zumba / Dance ─────────────────────────────────────────────
            "zumba":     "zumba",    "dumba":   "zumba",
        }
        for wrong, right in ex_typo_map.items():
            pattern = r"\b" + re.escape(wrong) + r"\b"
            query_lower = re.sub(pattern, right, query_lower)

        exact = self.collection.find_one({"exercise_name": query_lower}, {"_id": 0})
        if exact:
            return exact, 1.0

        # Fuzzy match — load all exercise names
        all_docs = list(self.collection.find({}, {"exercise_name": 1, "exercise_id": 1, "_id": 0}))
        if not all_docs:
            return None, 0.0

        name_to_id = {d["exercise_name"]: d["exercise_id"] for d in all_docs}
        names = list(name_to_id.keys())

        hits = rf_process.extract(
            query_lower, names,
            scorer=fuzz.token_set_ratio,
            limit=3,
        )

        if hits and hits[0][1] >= FUZZY_THRESHOLD:
            best_name, score, _ = hits[0]
            exercise = self.collection.find_one(
                {"exercise_id": name_to_id[best_name]}, {"_id": 0}
            )
            return exercise, round(score / 100, 2)

        return None, 0.0

    # ------------------------------------------------------------------
    # Caching of externally-resolved exercises into MongoDB
    # ------------------------------------------------------------------
    def get_by_exact_name(self, name: str) -> Optional[dict]:
        if not name:
            return None
        return self.collection.find_one(
            {"exercise_name": name.strip().lower()}, {"_id": 0}
        )

    def save_learned_exercise_sync(self, query: str, exercise_dict: dict) -> Optional[dict]:
        """
        Persist an externally-resolved (ExerciseDB / wger / Compendium / AI)
        exercise into the local `exercises` collection so repeated queries don't
        call the APIs again.

        Mirrors FoodRepositorySync.save_learned_food_sync:
          - Validates the calorie rate ( calories_per_unit_avg > 0 ).
          - Overwrite protection: never overwrites a verified (is_verified=True)
            master exercise — returns the existing doc instead.
          - Upserts by exercise_id and stores learned aliases on the doc.

        Returns the stored (or pre-existing) exercise document, or None if the
        input was invalid.
        """
        if not exercise_dict:
            return None

        name_clean = (exercise_dict.get("exercise_name") or query or "").strip().lower()
        if not name_clean:
            return None

        try:
            cal_avg = float(exercise_dict.get("calories_per_unit_avg", 0.0) or 0.0)
        except (TypeError, ValueError):
            cal_avg = 0.0
        if cal_avg <= 0:
            return None  # invalid live data — do not cache

        # Overwrite protection: keep trusted local/master data intact.
        existing = self.collection.find_one({"exercise_name": name_clean})
        if existing and existing.get("is_verified", False):
            return {k: v for k, v in existing.items() if k != "_id"}

        source = exercise_dict.get("source", "external")
        exercise_id = exercise_dict.get("exercise_id") or ("ex_learned_" + re.sub(r"[^\w]", "_", name_clean).strip("_"))

        # Merge query into aliases so future fuzzy lookups also resolve.
        aliases = set(exercise_dict.get("aliases") or [])
        if query:
            aliases.add(query.strip().lower())
        aliases.add(name_clean)

        doc = {
            "exercise_id": exercise_id,
            "exercise_name": name_clean,
            "exercise_name_display": exercise_dict.get("exercise_name_display", name_clean.title()),
            "category": exercise_dict.get("category", "Fitness"),
            "measurement_unit": normalize_exercise_unit(exercise_dict.get("measurement_unit")),
            "calories_per_unit_min": float(exercise_dict.get("calories_per_unit_min", cal_avg * 0.8) or cal_avg * 0.8),
            "calories_per_unit_avg": cal_avg,
            "calories_per_unit_max": float(exercise_dict.get("calories_per_unit_max", cal_avg * 1.2) or cal_avg * 1.2),
            "typical_duration_min": float(exercise_dict.get("typical_duration_min", 30.0) or 30.0),
            "met": exercise_dict.get("met"),
            "equipment": exercise_dict.get("equipment", ""),
            "muscle_group": exercise_dict.get("muscle_group", ""),
            "aliases": sorted(a for a in aliases if a),
            "data_source": source,
            "source": source,
            "is_verified": False,
            "is_learned": True,
        }
        self.collection.update_one(
            {"exercise_id": exercise_id}, {"$set": doc}, upsert=True
        )
        return {k: v for k, v in doc.items() if k != "_id"}


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class ExerciseCalculator:
    """
    Calculate calories burned for an exercise.

    Handles unit conversion:
      - If user gives minutes but exercise unit is reps → use session calories
      - If user gives reps but exercise unit is minutes → convert via typical_duration
      - Same unit → direct multiplication
    """

    def calculate(
        self,
        exercise: dict,
        amount: Optional[float] = None,
        unit: Optional[str] = None,
        weight_kg: Optional[float] = None,
    ) -> ExerciseCalcResult:
        """
        Calculate calories from a MongoDB exercise document.

        Calories burned scale with the user's ACTUAL body weight. The stored
        ``calories_per_unit_*`` rates are baked at a reference body weight
        (``REFERENCE_WEIGHT_KG`` = 70 kg for master-DB exercises, or the weight
        recorded in ``reference_weight_kg`` for externally-resolved ones). We
        rescale linearly — consistent with the MET formula
        ``kcal/min = MET × 3.5 × weight_kg / 200`` — so heavier users burn more
        and lighter users burn less for the same activity/duration.

        ``weight_kg=None`` means "use the reference weight" (factor 1.0), which
        preserves the previous behaviour for callers that don't pass a weight.
        """
        # Sanitize the stored measurement_unit ("1 rep" → "reps", "1 min" → "minutes")
        # so the displayed unit is always clean.
        native_unit = normalize_exercise_unit(exercise.get("measurement_unit"))
        input_amount = amount
        input_unit = normalize_exercise_unit(unit) if unit else native_unit

        # Default to typical duration/amount if none specified
        if amount is None:
            if native_unit == "session":
                amount = 1.0
            else:
                amount = exercise.get("typical_duration_min") or 30.0
            input_unit = native_unit

        effective_amount, effective_unit, note = self._convert_units(
            exercise, amount, input_unit, native_unit
        )

        # Body-weight scaling: rescale the baked per-unit rate from the weight it
        # was baked at (reference) to the user's actual weight.
        weight_factor = 1.0
        if weight_kg is not None:
            try:
                ref = float(exercise.get("reference_weight_kg") or REFERENCE_WEIGHT_KG)
                w = float(weight_kg)
                if ref > 0 and w > 0:
                    weight_factor = w / ref
            except (TypeError, ValueError):
                weight_factor = 1.0

        if effective_unit == "reps" and native_unit == "minutes":
            # Resistance/strength training rep rate: standard ~0.35 - 0.55 kcal/rep
            rep_rate_min = 0.35
            rep_rate_avg = 0.45
            rep_rate_max = 0.55
            cal_min = effective_amount * rep_rate_min * weight_factor
            cal_avg = effective_amount * rep_rate_avg * weight_factor
            cal_max = effective_amount * rep_rate_max * weight_factor
        else:
            cal_min = effective_amount * exercise["calories_per_unit_min"] * weight_factor
            cal_avg = effective_amount * exercise["calories_per_unit_avg"] * weight_factor
            cal_max = effective_amount * exercise["calories_per_unit_max"] * weight_factor

        return ExerciseCalcResult(
            exercise_id=exercise["exercise_id"],
            exercise_name=exercise["exercise_name"],
            exercise_name_display=exercise["exercise_name_display"],
            category=exercise["category"],
            amount=effective_amount,
            unit=effective_unit,
            input_amount=input_amount or effective_amount,
            input_unit=input_unit,
            calories_min=round(cal_min, 1),
            calories_avg=round(cal_avg, 1),
            calories_max=round(cal_max, 1),
            search_confidence=1.0,
            typical_duration_min=exercise.get("typical_duration_min"),
            calculation_note=note,
        )

    def _convert_units(
        self,
        exercise: dict,
        amount: float,
        input_unit: str,
        native_unit: str,
    ) -> tuple[float, str, str]:
        """
        Convert user's unit to exercise's native unit.
        Returns (effective_amount, effective_unit, note).
        """
        # Same unit or compatible
        if input_unit == native_unit:
            return amount, native_unit, f"{amount:.0f} {native_unit}"

        # Minutes → session-based exercise
        if input_unit == "minutes" and native_unit == "session":
            typical = exercise.get("typical_duration_min")
            if typical and typical > 0:
                ratio = amount / typical
                return ratio, "session", (
                    f"{amount:.0f} min = {ratio:.1f} sessions "
                    f"(typical session: {typical:.0f} min)"
                )
            return 1.0, "session", f"Assumed 1 session for {amount:.0f} min"

        # Minutes → reps-based exercise
        if input_unit == "minutes" and native_unit == "reps":
            typical = exercise.get("typical_duration_min")
            if typical and typical > 0:
                ratio = amount / typical
                cal_avg = (exercise.get("session_calories_avg") or 0) * ratio
                per_unit_avg = exercise["calories_per_unit_avg"]
                equiv_reps = (cal_avg / per_unit_avg) if per_unit_avg > 0 else amount
                return equiv_reps, "reps", (
                    f"{amount:.0f} min ~ {equiv_reps:.0f} reps "
                    f"(based on session: {typical:.0f} min)"
                )
            return amount, native_unit, f"Estimated {amount:.0f} reps from {amount:.0f} min"

        # Reps → minutes-based exercise (e.g. user gave reps for generic workout or strength exercise)
        # NEVER convert reps into minutes! Keep unit as reps and calculate via resistance rate.
        if input_unit == "reps" and native_unit == "minutes":
            return amount, "reps", f"{amount:.0f} reps (resistance training rate)"

        # Sets → reps (already handled in parser, but safety)
        if input_unit == "sets":
            return amount * 10, "reps", f"{amount:.0f} sets ~ {amount * 10:.0f} reps (assuming 10/set)"

        # Hours → minutes
        if input_unit == "hours":
            mins = amount * 60
            return mins, "minutes", f"{amount:.1f} hours = {mins:.0f} minutes"

        # Distance (km) → minutes for time-based exercises (running/walking/cycling).
        # Uses a simple average pace so calories can be estimated. Keeps the
        # user-facing unit as "km" for display via input_unit tracking.
        if input_unit == "km":
            name = (exercise.get("exercise_name") or "").lower()
            # minutes-per-km pace estimates
            if "cycl" in name:
                pace = 3.0     # ~20 km/h
            elif "walk" in name:
                pace = 12.0    # ~5 km/h
            else:               # running / jogging / default
                pace = 6.0     # ~10 km/h
            if native_unit == "minutes":
                mins = amount * pace
                return mins, "minutes", f"{amount:g} km ~ {mins:.0f} min (est. pace {pace:.0f} min/km)"
            # native is session/reps → approximate as a session fraction
            typical = exercise.get("typical_duration_min") or 30.0
            mins = amount * pace
            ratio = mins / typical if typical else 1.0
            return ratio, native_unit, f"{amount:g} km ~ {mins:.0f} min"

        # Fallback: use as-is with the native unit
        return amount, native_unit, f"{amount:.0f} {input_unit} (treated as {native_unit})"


def calculate_routine(
    routine: ParsedWorkoutRoutine,
    weight_kg: Optional[float] = None,
) -> RoutineCalcResult:
    """
    Calculate calories burned for a structured workout routine.
    If duration is provided, uses MET-based resistance training formula.
    If duration is NOT provided, computes directly from reps/sets without inventing minutes.
    """
    w = float(weight_kg) if weight_kg and float(weight_kg) > 0 else REFERENCE_WEIGHT_KG
    weight_factor = w / REFERENCE_WEIGHT_KG

    if routine.duration_min is not None and routine.duration_min > 0:
        # User explicitly stated a duration for the workout (e.g. 45 min)
        # Moderate-vigorous resistance training MET ~ 5.0
        dur = float(routine.duration_min)
        cal_avg = 5.0 * 3.5 * w / 200.0 * dur
        cal_min = cal_avg * 0.85
        cal_max = cal_avg * 1.15
    else:
        # Rep-based calculation for resistance routine
        # Weight training rate: ~0.45 kcal/rep average (at 70 kg)
        tot_reps = routine.total_reps if routine.total_reps > 0 else (routine.total_exercises * 12)
        cal_min = tot_reps * 0.35 * weight_factor
        cal_avg = tot_reps * 0.45 * weight_factor
        cal_max = tot_reps * 0.55 * weight_factor

    # Build user-friendly summary description
    # e.g., "Chest (4 exercises × 12 reps), Shoulder (2 exercises × 12 reps), Triceps (2 exercises × 12 reps)"
    item_descs = []
    for it in routine.items:
        m_name = it.muscle_group.capitalize()
        reps_str = f"{it.reps} reps" if it.reps else ""
        sets_str = f"{it.sets} sets" if it.sets else ""
        if it.exercise_count > 1:
            cnt_str = f"{it.exercise_count} exercises"
        else:
            cnt_str = it.exercise_name if it.exercise_name and it.exercise_name.lower() != f"{it.muscle_group} exercise" else "1 exercise"

        if reps_str and sets_str:
            detail = f"{cnt_str} × {sets_str} × {reps_str}"
        elif reps_str:
            detail = f"{cnt_str} × {reps_str}"
        elif sets_str:
            detail = f"{cnt_str} × {sets_str}"
        else:
            detail = cnt_str

        item_descs.append(f"{m_name} ({detail})")

    summary_text = ", ".join(item_descs)
    return RoutineCalcResult(
        routine=routine,
        calories_min=round(cal_min, 1),
        calories_avg=round(cal_avg, 1),
        calories_max=round(cal_max, 1),
        summary_text=summary_text,
        duration_min=routine.duration_min,
        total_reps=routine.total_reps,
    )
