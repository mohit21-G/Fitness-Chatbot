"""
generate_dataset.py  — v2
==========================
Generates 10 000 + high-quality ChatML training examples for the Fitness
Chatbot intent parser.

Strategy
--------
Rather than writing every sentence by hand, we use TEMPLATE × SLOT expansion:

  template = "I had {qty} {food} for {meal}"
  slots    = cross-product of qty_variants × food_list × meal_list

One template across 5 qty variants × 20 foods × 3 meals = 300 sentences from
a single line, each with a correct JSON answer synthesised automatically.

Categories covered
------------------
  1.  food_qty_en          food + quantity (English)
  2.  food_noqty_en        food, quantity missing
  3.  food_nomeal_en       food + qty, meal missing
  4.  food_gujarati        Roman-Gujarati sentences
  5.  food_hindi           Hindi / Hinglish
  6.  food_typo            Typos / phonetic spellings
  7.  multi_food           Multiple foods in one sentence
  8.  exercise_dur         Exercise + duration
  9.  exercise_reps        Exercise + reps / sets
 10.  exercise_dist        Exercise + distance
 11.  exercise_noamt       Exercise, amount missing
 12.  nutrition            Calorie / macro knowledge lookup
 13.  daily_log            Summary / query_meal reads
 14.  yesterday            Yesterday date variants
 15.  variants             Cooking method variants
 16.  profile              Profile / BMR / TDEE queries
 17.  greeting             Greetings
 18.  junk                 Off-topic / gibberish → clarification
 19.  ambiguous            Vague food → clarification
 20.  skip_meal            Skipped a meal
 21.  exercise_read        Reading exercise log
 22.  followup_context     Follow-up / contextual turns

Run
---
    cd "d:\\anques\\fitness chatbot"
    python training/generate_dataset.py

Outputs
-------
    training/dataset_train.jsonl   (80 % of deduplicated examples)
    training/dataset_eval.jsonl    (20 % held-out, no leakage)
    training/dataset_stats.txt     Category counts and duplicate stats
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (verbatim copy from backend/llm_service.py)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a fitness chatbot intent parser. Your ONLY job is to extract "
    "structured data from user messages about food logging, exercise logging, "
    "or nutrition queries.\n\n"
    "CORE PRINCIPLES:\n"
    "1. Respond ONLY with valid JSON. No extra text, no markdown.\n"
    "2. You do NOT calculate calories or nutrition. Just extract what the user said.\n"
    "3. Understand ANY language, script, and writing style.\n"
    "4. NEVER substitute the user's food/exercise with a similar or generic or "
    "unrelated one.\n"
    "5. Extract EVERY detail present: quantity, unit, reps, sets, distance, "
    "duration, intensity, time of day, and meal/workout context.\n"
    "6. If exactly one required detail is missing, still return the log intent "
    "and set missing_detail to that field name.\n"
    "7. Capture EVERY food the user mentions as its own entry in foods.\n\n"
    "OUTPUT FORMAT (JSON only):\n"
    '{"intent":"...","food_query":"...","quantity":"...","variant":"...",'
    '"meal_type":"...","foods":[...],"exercise_query":"...","exercise_input":"...",'
    '"reps":null,"sets":null,"distance":null,"distance_unit":null,'
    '"duration_min":null,"intensity":null,"time_of_day":null,'
    '"date":"today","missing_detail":null,"exact_term":"...","clarification_question":null}'
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _out(**kw) -> str:
    d: dict[str, Any] = {
        "intent":                kw.get("intent", "log_food"),
        "exact_term":            kw.get("exact_term"),
        "food_query":            kw.get("food_query"),
        "quantity":              kw.get("quantity"),
        "variant":               kw.get("variant"),
        "meal_type":             kw.get("meal_type"),
        "foods":                 kw.get("foods", []),
        "exercise_query":        kw.get("exercise_query"),
        "exercise_input":        kw.get("exercise_input"),
        "reps":                  kw.get("reps"),
        "sets":                  kw.get("sets"),
        "distance":              kw.get("distance"),
        "distance_unit":         kw.get("distance_unit"),
        "duration_min":          kw.get("duration_min"),
        "intensity":             kw.get("intensity"),
        "time_of_day":           kw.get("time_of_day"),
        "date":                  kw.get("date", "today"),
        "missing_detail":        kw.get("missing_detail"),
        "clarification_question":kw.get("clarification_question"),
    }
    return json.dumps(d, ensure_ascii=False)


def fi(food_query, quantity=None, variant=None, meal_type=None) -> dict:
    return {"food_query": food_query, "quantity": quantity,
            "variant": variant, "meal_type": meal_type}


def ex(user: str, cat: str, **kw) -> dict:
    return {
        "category": cat,
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": _out(**kw)},
        ],
    }


def _fp(s: str) -> str:
    """Fingerprint a user utterance for dedup."""
    return hashlib.md5(s.strip().lower().encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# SLOT BANKS
# ─────────────────────────────────────────────────────────────────────────────

MEALS = ["breakfast", "lunch", "dinner", "snack"]

MEAL_WORDS = {
    "breakfast": ["breakfast", "savare", "subah", "savar ma", "nashte me",
                  "nashmte me", "subah ke nashte me", "morning nashto"],
    "lunch":     ["lunch", "bapore", "dopahar", "dopahar me", "lunch ma",
                  "baporna", "dopahar ko"],
    "dinner":    ["dinner", "raatre", "raat ko", "dinner ma", "raat ke khane me",
                  "supper"],
    "snack":     ["snack", "sanje", "shaam ko", "evening", "chai time",
                  "tea time", "bich me"],
}

# canonical food name → (display, qty_unit, typical_qty_list)
# qty_list items are (amount_str, float_for_json)
FOODS_EN: list[tuple[str, str, list[tuple[str, Any]]]] = [
    # Indian breads
    ("roti",            "roti",           [("1 piece",1),("2 pieces",2),("3 pieces",3)]),
    ("chapati",         "chapati",        [("1 piece",1),("2 pieces",2)]),
    ("paratha",         "paratha",        [("1 piece",1),("2 pieces",2)]),
    ("naan",            "naan",           [("1 piece",1),("2 pieces",2)]),
    ("puri",            "puri",           [("2 pieces",2),("4 pieces",4)]),
    ("thepla",          "thepla",         [("1 piece",1),("2 pieces",2),("3 pieces",3)]),
    ("bhakri",          "bhakri",         [("1 piece",1),("2 pieces",2)]),
    ("bhatura",         "bhatura",        [("1 piece",1),("2 pieces",2)]),
    ("idli",            "idli",           [("2 pieces",2),("3 pieces",3),("4 pieces",4)]),
    ("dosa",            "dosa",           [("1 piece",1),("2 pieces",2)]),
    ("uttapam",         "uttapam",        [("1 piece",1),("2 pieces",2)]),
    ("aloo paratha",    "aloo paratha",   [("1 piece",1),("2 pieces",2)]),
    ("methi thepla",    "methi thepla",   [("1 piece",1),("2 pieces",2)]),
    # Rice dishes
    ("rice",            "rice",           [("1 bowl",1),("1 cup",1),("200g","200g")]),
    ("biryani",         "biryani",        [("1 plate",1),("1 bowl",1)]),
    ("pulao",           "pulao",          [("1 plate",1),("1 serving",1)]),
    ("khichdi",         "khichdi",        [("1 bowl",1),("1 katori",1)]),
    ("fried rice",      "fried rice",     [("1 plate",1),("1 bowl",1)]),
    ("dal khichdi",     "dal khichdi",    [("1 bowl",1),("1 serving",1)]),
    # Dal / lentils
    ("dal",             "dal",            [("1 bowl",1),("1 katori",1),("200ml","200ml")]),
    ("dal tadka",       "dal tadka",      [("1 bowl",1),("1 serving",1)]),
    ("sambar",          "sambar",         [("1 bowl",1),("1 cup",1)]),
    ("rajma",           "rajma",          [("1 bowl",1),("1 katori",1)]),
    ("chole",           "chole",          [("1 bowl",1),("1 plate",1)]),
    ("moong dal",       "moong dal",      [("1 bowl",1),("1 katori",1)]),
    # Curries / sabzi
    ("palak paneer",    "palak paneer",   [("1 bowl",1),("1 serving",1)]),
    ("paneer tikka",    "paneer tikka",   [("100g","100g"),("200g","200g"),("1 plate",1)]),
    ("aloo gobi",       "aloo gobi",      [("1 bowl",1),("1 serving",1)]),
    ("matar paneer",    "matar paneer",   [("1 bowl",1),("1 serving",1)]),
    ("bhindi masala",   "bhindi masala",  [("1 bowl",1),("1 serving",1)]),
    ("mixed vegetable", "mixed vegetable",[("1 bowl",1),("1 serving",1)]),
    ("aloo sabzi",      "aloo sabzi",     [("1 bowl",1),("1 serving",1)]),
    # Street food / snacks
    ("samosa",          "samosa",         [("1 piece",1),("2 pieces",2),("3 pieces",3)]),
    ("kachori",         "kachori",        [("1 piece",1),("2 pieces",2)]),
    ("vada pav",        "vada pav",       [("1 piece",1),("2 pieces",2)]),
    ("pav bhaji",       "pav bhaji",      [("1 plate",1),("1 serving",1)]),
    ("dhokla",          "dhokla",         [("2 pieces",2),("4 pieces",4),("1 plate",1)]),
    ("khaman",          "khaman",         [("2 pieces",2),("4 pieces",4)]),
    ("pani puri",       "pani puri",      [("1 plate",1),("6 pieces",6)]),
    ("bhel puri",       "bhel puri",      [("1 plate",1),("1 bowl",1)]),
    ("sev puri",        "sev puri",       [("1 plate",1)]),
    ("chole bhature",   "chole bhature",  [("1 plate",1),("1 serving",1)]),
    ("dabeli",          "dabeli",         [("1 piece",1),("2 pieces",2)]),
    ("fafda",           "fafda",          [("1 serving",1),("50g","50g")]),
    # Sweets / desserts
    ("gulab jamun",     "gulab jamun",    [("1 piece",1),("2 pieces",2)]),
    ("rasgulla",        "rasgulla",       [("1 piece",1),("2 pieces",2)]),
    ("jalebi",          "jalebi",         [("1 piece",1),("2 pieces",2),("100g","100g")]),
    ("kheer",           "kheer",          [("1 bowl",1),("1 katori",1)]),
    ("halwa",           "halwa",          [("1 bowl",1),("1 katori",1)]),
    ("barfi",           "barfi",          [("1 piece",1),("2 pieces",2)]),
    ("kaju katli",      "kaju katli",     [("1 piece",1),("2 pieces",2),("100g","100g")]),
    ("ladoo",           "ladoo",          [("1 piece",1),("2 pieces",2)]),
    ("modak",           "modak",          [("1 piece",1),("2 pieces",2)]),
    # Beverages
    ("chai",            "chai",           [("1 cup",1),("2 cups",2),("150ml","150ml")]),
    ("coffee",          "coffee",         [("1 cup",1),("1 glass",1)]),
    ("green tea",       "green tea",      [("1 cup",1),("2 cups",2)]),
    ("milk",            "milk",           [("1 glass",1),("200ml","200ml"),("250ml","250ml")]),
    ("lassi",           "lassi",          [("1 glass",1),("250ml","250ml")]),
    ("chhas",           "chhas",          [("1 glass",1),("200ml","200ml")]),
    ("coconut water",   "coconut water",  [("1 glass",1),("200ml","200ml")]),
    ("juice",           "juice",          [("1 glass",1),("200ml","200ml")]),
    ("protein shake",   "protein shake",  [("1 serving",1),("250ml","250ml")]),
    # Protein
    ("chicken breast",  "chicken breast", [("100g","100g"),("150g","150g"),("200g","200g")]),
    ("egg",             "egg",            [("1 piece",1),("2 pieces",2),("3 pieces",3)]),
    ("paneer",          "paneer",         [("100g","100g"),("150g","150g"),("1 katori",1)]),
    ("tofu",            "tofu",           [("100g","100g"),("150g","150g")]),
    ("fish",            "fish",           [("100g","100g"),("150g","150g")]),
    ("mutton",          "mutton",         [("100g","100g"),("150g","150g")]),
    # Healthy / other
    ("oats",            "oats",           [("1 bowl",1),("1 cup",1),("50g","50g")]),
    ("poha",            "poha",           [("1 bowl",1),("1 plate",1)]),
    ("upma",            "upma",           [("1 bowl",1),("1 plate",1)]),
    ("dalia",           "dalia",          [("1 bowl",1),("1 cup",1)]),
    ("muesli",          "muesli",         [("1 bowl",1),("50g","50g")]),
    ("cornflakes",      "cornflakes",     [("1 bowl",1),("30g","30g")]),
    ("fruit salad",     "fruit salad",    [("1 bowl",1),("1 plate",1)]),
    ("banana",          "banana",         [("1 piece",1),("2 pieces",2)]),
    ("apple",           "apple",          [("1 piece",1),("2 pieces",2)]),
    ("mango",           "mango",          [("1 piece",1),("1 slice",1)]),
    ("almonds",         "almonds",        [("10 pieces",10),("30g","30g"),("1 handful",1)]),
    ("peanut butter",   "peanut butter",  [("1 tablespoon",1),("2 tablespoons",2)]),
    ("yogurt",          "yogurt",         [("1 cup",1),("100g","100g"),("150g","150g")]),
    ("salad",           "salad",          [("1 bowl",1),("1 plate",1)]),
    ("pizza",           "pizza",          [("1 slice",1),("2 slices",2)]),
    ("burger",          "burger",         [("1 piece",1),("1 serving",1)]),
    ("sandwich",        "sandwich",       [("1 piece",1),("2 pieces",2)]),
    ("pasta",           "pasta",          [("1 bowl",1),("1 plate",1)]),
    ("noodles",         "noodles",        [("1 bowl",1),("1 plate",1)]),
    ("soup",            "soup",           [("1 bowl",1),("1 cup",1)]),
    ("ice cream",       "ice cream",      [("1 scoop",1),("2 scoops",2)]),
    ("chocolate",       "chocolate",      [("2 pieces",2),("30g","30g")]),
    ("biscuit",         "biscuit",        [("2 pieces",2),("4 pieces",4)]),
    ("bread",           "bread",          [("1 slice",1),("2 slices",2)]),
]

EXERCISES: list[tuple[str, str, str]] = [
    # (canonical, display, measurement_unit)
    ("jogging",      "jogging",      "minutes"),
    ("running",      "running",      "km"),
    ("walking",      "walking",      "minutes"),
    ("cycling",      "cycling",      "km"),
    ("swimming",     "swimming",     "minutes"),
    ("yoga",         "yoga",         "minutes"),
    ("push-ups",     "push-ups",     "reps"),
    ("squats",       "squats",       "reps"),
    ("sit-ups",      "sit-ups",      "reps"),
    ("crunches",     "crunches",     "reps"),
    ("lunges",       "lunges",       "reps"),
    ("burpees",      "burpees",      "reps"),
    ("plank",        "plank",        "minutes"),
    ("jumping jacks","jumping jacks","reps"),
    ("HIIT",         "HIIT",         "minutes"),
    ("zumba",        "zumba",        "minutes"),
    ("skipping rope","skipping rope","minutes"),
    ("badminton",    "badminton",    "minutes"),
    ("cricket",      "cricket",      "minutes"),
    ("bench press",  "bench press",  "reps"),
    ("deadlift",     "deadlift",     "reps"),
    ("bicep curls",  "bicep curls",  "reps"),
    ("shoulder press","shoulder press","reps"),
    ("treadmill",    "treadmill",    "minutes"),
    ("surya namaskar","surya namaskar","minutes"),
    ("elliptical",   "elliptical",   "minutes"),
    ("stair climbing","stair climbing","minutes"),
    ("rowing",       "rowing",       "minutes"),
    ("dance",        "dance",        "minutes"),
    ("stretching",   "stretching",   "minutes"),
]

VARIANTS_LIST = ["ghee", "butter", "oil", "fried", "deep fried",
                 "grilled", "boiled", "steamed", "baked", "roasted"]

QTY_EXTRAS = {
    "g":   [(50,"50g"),(75,"75g"),(100,"100g"),(150,"150g"),(200,"200g"),(250,"250g")],
    "ml":  [(100,"100ml"),(150,"150ml"),(200,"200ml"),(250,"250ml")],
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FOOD + QTY  (English)
# ─────────────────────────────────────────────────────────────────────────────
def gen_food_qty_en(cat="food_qty_en") -> list[dict]:
    templates_with_meal = [
        ("I had {qty} {food} for {meal}", True),
        ("ate {qty} {food} for {meal}",   True),
        ("had {qty} {food} at {meal}",    True),
        ("just had {qty} {food} in {meal}",True),
        ("finished {qty} {food} for my {meal}",True),
        ("logged {qty} {food} — {meal}",  True),
        ("{qty} {food} for {meal}",       True),
        ("grabbed {qty} {food} for {meal}",True),
        ("ate {qty} of {food} during {meal}",True),
        ("consumed {qty} {food} at {meal}",True),
    ]
    templates_no_meal = [
        ("had {qty} {food}",              False),
        ("ate {qty} {food}",              False),
        ("I ate {qty} {food}",            False),
        ("just had {qty} {food}",         False),
        ("drank {qty} {food}",            False),  # beverages
        ("{qty} {food} khadhu",           False),
        ("aaj {qty} {food} khadha",       False),
    ]
    out = []
    for food, disp, qty_list in FOODS_EN:
        for qty_str, qty_val in qty_list:
            # with meal
            for tpl, _ in templates_with_meal:
                meal = random.choice(MEALS)
                user = tpl.format(qty=qty_str, food=food, meal=meal)
                out.append(ex(user, cat,
                    intent="log_food", exact_term=food, food_query=food,
                    quantity=qty_str, meal_type=meal,
                    foods=[fi(food, qty_str, None, meal)],
                    missing_detail=None,
                ))
            # without meal
            for tpl, _ in templates_no_meal:
                user = tpl.format(qty=qty_str, food=food)
                out.append(ex(user, cat,
                    intent="log_food", exact_term=food, food_query=food,
                    quantity=qty_str, meal_type=None,
                    foods=[fi(food, qty_str, None, None)],
                    missing_detail="meal_type",
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FOOD, QUANTITY MISSING
# ─────────────────────────────────────────────────────────────────────────────
def gen_food_noqty(cat="food_noqty") -> list[dict]:
    templates = [
        ("ate {food} for {meal}",      True),
        ("had {food} for {meal}",      True),
        ("I had {food} at {meal}",     True),
        ("just ate {food} in {meal}",  True),
        ("{food} khaya {meal} me",     True),
        ("had {food}",                 False),
        ("ate {food}",                 False),
        ("I ate {food}",               False),
        ("drank {food}",               False),
        ("me aaje {food} khadhu",      False),
        ("aaj {food} khaya",           False),
    ]
    out = []
    for food, disp, _ in FOODS_EN:
        for tpl, has_meal in templates:
            meal = random.choice(MEALS) if has_meal else None
            user = tpl.format(food=food, meal=meal or "")
            out.append(ex(user, cat,
                intent="log_food", exact_term=food, food_query=food,
                quantity=None, meal_type=meal,
                foods=[fi(food, None, None, meal)],
                missing_detail="quantity",
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GUJARATI sentences
# ─────────────────────────────────────────────────────────────────────────────
GU_FOODS: list[tuple[str, str, str, str|None, str|None]] = [
    # (user_phrase,  canonical, qty, variant, meal)
    ("aaje me 2 rotli khadhi",                 "roti",         "2 pieces", None,   "dinner"),
    ("savare 3 thepla khadha",                 "thepla",       "3 pieces", None,   "breakfast"),
    ("me aaje lunch ma khichdi khadhi",        "khichdi",      "1 serving",None,   "lunch"),
    ("hu sanje 2 samosa khadho",               "samosa",       "2 pieces", None,   "snack"),
    ("aaje me chaas pidhi",                    "chhas",        "1 glass",  None,   None),
    ("me bapore dal bhaat khadho",             "dal",          "1 serving",None,   "lunch"),
    ("aaje savare poha khadha",                "poha",         "1 bowl",   None,   "breakfast"),
    ("raatre me 1 rotlo khadho",               "bajri rotla",  "1 piece",  None,   "dinner"),
    ("me sanje 2 bhakri ghee sathe khadhi",   "bhakri",       "2 pieces", "ghee", "snack"),
    ("aaje me puri ane dahi khadhi",           "puri",         "2 pieces", None,   "lunch"),
    ("hu breakfast ma upma khadho",            "upma",         "1 bowl",   None,   "breakfast"),
    ("me aaje handvo khadho",                  "handvo",       "1 serving",None,   None),
    ("bapore fafda jalebi khadha",             "fafda",        "1 serving",None,   "snack"),
    ("aaje raatre khichdi ane kadhi",          "khichdi",      "1 serving",None,   "dinner"),
    ("sanje chai ane biskut",                  "chai",         "1 cup",    None,   "snack"),
    ("me aaje nashte ma idli sambar khadhu",   "idli",         "2 pieces", None,   "breakfast"),
    ("hu bapore 1 katori dal khadhi",          "dal",          "1 katori", None,   "lunch"),
    ("aaje me thepla ane chai khadha",         "thepla",       "2 pieces", None,   "breakfast"),
    ("raatre me dal roti khadha",              "roti",         "2 pieces", None,   "dinner"),
    ("me sanje bhel puri khadhi",              "bhel puri",    "1 plate",  None,   "snack"),
    ("aaj savare poha khayo hato",             "poha",         "1 bowl",   None,   "breakfast"),
    ("hu raatre 1 rotlo ne shaak jamyo",       "bajri rotla",  "1 piece",  None,   "dinner"),
    ("me lunch ma rajma rice khadhu",          "rajma",        "1 bowl",   None,   "lunch"),
    ("aaje sanje vada pav khadho",             "vada pav",     "1 piece",  None,   "snack"),
    ("bapore me chole bhature jamya",          "chole bhature","1 plate",  None,   "lunch"),
    ("aaje me khakhra khadha",                 "khakhra",      "4 pieces", None,   "snack"),
    ("me breakfast ma cornflakes milk sathe", "cornflakes",   "1 bowl",   None,   "breakfast"),
    ("hu sanje 1 glass lassi pidhi",           "lassi",        "1 glass",  None,   "snack"),
    ("aaje raatre soup piya",                  "soup",         "1 bowl",   None,   "dinner"),
    ("me aaje dudh piya",                      "milk",         "1 glass",  None,   None),
    ("savare mara nashta ma egg khadho",       "egg",          "2 pieces", None,   "breakfast"),
    ("aaje me paneer khadhu",                  "paneer",       "100g",     None,   None),
    ("me bapore biryani khadhi",               "biryani",      "1 plate",  None,   "lunch"),
    ("hu aaje samosa khadho",                  "samosa",       "2 pieces", None,   None),
    ("aaje me dahi vada khadha",               "dahi vada",    "2 pieces", None,   None),
    ("raatre paneer butter masala khadhu",     "paneer butter masala","1 serving",None,"dinner"),
    ("me breakfast ma toast ane egg khadha",   "toast",        "2 slices", None,   "breakfast"),
    ("aaje bapore dal chawal jamya",           "dal",          "1 bowl",   None,   "lunch"),
    ("hu sanje green tea pidhi",               "green tea",    "1 cup",    None,   "snack"),
    ("me aaje 1 katori kheer khadhi",          "kheer",        "1 katori", None,   None),
    # typos / informal Gujarati
    ("aaje me thaepla khadha",                 "thepla",       "1 piece",  None,   None),
    ("hu bapore dal chaval khadho",            "dal",          "1 serving",None,   "lunch"),
    ("me sanje 2 samosha khadha",              "samosa",       "2 pieces", None,   "snack"),
    ("aaje me dhokhla khadho",                 "dhokla",       "2 pieces", None,   None),
    ("savare mane rotlo khadhto hato",         "bajri rotla",  "1 piece",  None,   "breakfast"),
]

GU_TEMPLATES_NOQTY = [
    ("aaje me {food} khadhu",    None),
    ("me bapore {food} khadho",  "lunch"),
    ("hu sanje {food} khadho",   "snack"),
    ("raatre me {food} khadha",  "dinner"),
    ("savare {food} khadhto",    "breakfast"),
    ("aaje {food} jamyu",        None),
    ("me aaje {food} pidhu",     None),
    ("bapore {food} khadha",     "lunch"),
    ("aaj me {food} khadhu",     None),
    ("me raat ma {food} khadhu", "dinner"),
]

GU_FOODS_EXTRA: list[str] = [
    "thepla","bhakri","handvo","dhokla","fafda","khakhra","undhiyu","khichdi",
    "dal","poha","upma","samosa","chai","lassi","chhas","biryani","rajma",
    "idli","dosa","chole","puri","kheer","halwa","jalebi","gulab jamun",
]

def gen_gujarati(cat="food_gujarati") -> list[dict]:
    out = []
    # explicit curated examples
    for user, food, qty, var, meal in GU_FOODS:
        out.append(ex(user, cat,
            intent="log_food", exact_term=food, food_query=food,
            quantity=qty, variant=var, meal_type=meal,
            foods=[fi(food, qty, var, meal)],
            missing_detail=None if (qty and meal) else (None if not qty and not meal else ("quantity" if not qty else None)),
        ))
    # template-generated no-qty
    for tpl, meal in GU_TEMPLATES_NOQTY:
        for food in GU_FOODS_EXTRA:
            user = tpl.format(food=food)
            out.append(ex(user, cat,
                intent="log_food", exact_term=food, food_query=food,
                quantity=None, meal_type=meal,
                foods=[fi(food, None, None, meal)],
                missing_detail="quantity",
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HINDI / HINGLISH
# ─────────────────────────────────────────────────────────────────────────────
HI_TEMPLATES_WITH_MEAL: list[tuple[str, str]] = [
    ("maine {meal} me {qty} {food} khayi",    "food"),
    ("aaj {meal} me {food} khaya",            "food"),
    ("subah {meal} me {qty} {food} khayi",    "food"),
    ("{meal} ko {qty} {food} khayi",          "food"),
    ("raat ko {meal} pe {qty} {food}",        "food"),
    ("maine {meal} me {food} khaya",          "food"),
    ("aaj maine {meal} ke liye {qty} {food}", "food"),
    ("{qty} {food} {meal} me khayi",          "food"),
    ("dopahar ko {qty} {food}",               "food"),
    ("shaam ko {qty} {food} snack me",        "food"),
]

HI_FOODS: list[tuple[str, str, list[tuple[str,Any]]]] = [
    ("dal chawal",    "dal",       [("1 bowl",1),("1 serving",1)]),
    ("paratha",       "paratha",   [("2 pieces",2),("1 piece",1)]),
    ("roti",          "roti",      [("2 pieces",2),("3 pieces",3)]),
    ("poha",          "poha",      [("1 bowl",1),("1 plate",1)]),
    ("idli",          "idli",      [("3 pieces",3),("2 pieces",2)]),
    ("biryani",       "biryani",   [("1 plate",1),("1 bowl",1)]),
    ("samosa",        "samosa",    [("1 piece",1),("2 pieces",2)]),
    ("rajma chawal",  "rajma",     [("1 bowl",1),("1 plate",1)]),
    ("chole",         "chole",     [("1 bowl",1),("1 plate",1)]),
    ("aloo sabzi",    "aloo sabzi",[("1 katori",1),("1 bowl",1)]),
    ("dahi",          "curd",      [("1 katori",1),("100g","100g")]),
    ("chai",          "chai",      [("1 cup",1),("2 cups",2)]),
    ("doodh",         "milk",      [("1 glass",1),("200ml","200ml")]),
    ("anda",          "egg",       [("2 pieces",2),("1 piece",1)]),
    ("oats",          "oats",      [("1 bowl",1),("50g","50g")]),
    ("khichdi",       "khichdi",   [("1 bowl",1),("1 katori",1)]),
    ("upma",          "upma",      [("1 plate",1),("1 bowl",1)]),
    ("halwa",         "halwa",     [("1 katori",1),("100g","100g")]),
    ("paneer",        "paneer",    [("100g","100g"),("150g","150g")]),
    ("matar paneer",  "matar paneer",[("1 bowl",1),("1 serving",1)]),
    ("lassi",         "lassi",     [("1 glass",1),("250ml","250ml")]),
    ("sabzi",         "mixed vegetable",[("1 katori",1),("1 bowl",1)]),
    ("puri",          "puri",      [("2 pieces",2),("4 pieces",4)]),
    ("pakoda",        "pakora",    [("2 pieces",2),("4 pieces",4)]),
    ("gulab jamun",   "gulab jamun",[("1 piece",1),("2 pieces",2)]),
    ("jalebi",        "jalebi",    [("1 piece",1),("100g","100g")]),
    ("fruit",         "fruit",     [("1 piece",1),("1 bowl",1)]),
    ("aloo paratha",  "aloo paratha",[("1 piece",1),("2 pieces",2)]),
    ("chicken",       "chicken",   [("100g","100g"),("150g","150g")]),
    ("soup",          "soup",      [("1 bowl",1),("1 cup",1)]),
]

HI_MEALS_MAP = {
    "breakfast": ["subah","nashte me","subah ke nashte me","morning me"],
    "lunch":     ["dopahar","dopahar me","lunch me","bich mein"],
    "dinner":    ["raat ko","raat ke khane me","dinner pe","raat"],
    "snack":     ["shaam ko","shaam ke snack me","chai time pe","evening me"],
}

def gen_hindi(cat="food_hindi") -> list[dict]:
    out = []
    for food_raw, food_can, qty_list in HI_FOODS:
        for qty_str, qty_val in qty_list:
            for meal in MEALS:
                meal_words = HI_MEALS_MAP[meal]
                for meal_w in meal_words:
                    templates = [
                        f"maine {meal_w} {qty_str} {food_raw} khayi",
                        f"aaj {meal_w} {food_raw} khaya",
                        f"maine {food_raw} {meal_w} {qty_str} khaya",
                        f"{qty_str} {food_raw} {meal_w} khayi",
                        f"aaj {meal_w} mujhe {qty_str} {food_raw} mila",
                    ]
                    user = random.choice(templates)
                    out.append(ex(user, cat,
                        intent="log_food", exact_term=food_can, food_query=food_can,
                        quantity=qty_str, meal_type=meal,
                        foods=[fi(food_can, qty_str, None, meal)],
                        missing_detail=None,
                    ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TYPOS / PHONETIC
# ─────────────────────────────────────────────────────────────────────────────
TYPO_MAP: list[tuple[str, str]] = [
    # (typo_form, canonical)
    ("rotii",          "roti"),
    ("rooti",          "roti"),
    ("rotti",          "roti"),
    ("chaptti",        "chapati"),
    ("chapathy",       "chapati"),
    ("paratha",        "paratha"),
    ("parattha",       "paratha"),
    ("thaepla",        "thepla"),
    ("theepla",        "thepla"),
    ("theplaa",        "thepla"),
    ("methi thaepla",  "methi thepla"),
    ("daal",           "dal"),
    ("daal tadkaa",    "dal tadka"),
    ("biryaani",       "biryani"),
    ("briyaani",       "biryani"),
    ("briyani",        "biryani"),
    ("samosha",        "samosa"),
    ("samoosa",        "samosa"),
    ("samosaa",        "samosa"),
    ("kachoori",       "kachori"),
    ("paneeer",        "paneer"),
    ("panier",         "paneer"),
    ("panneer",        "paneer"),
    ("paneer tikaa",   "paneer tikka"),
    ("palek paneer",   "palak paneer"),
    ("palack paneer",  "palak paneer"),
    ("mater panner",   "matar paneer"),
    ("rajmah",         "rajma"),
    ("rajmaah",        "rajma"),
    ("chhole",         "chole"),
    ("chhole bhature", "chole bhature"),
    ("chhole bhaturre","chole bhature"),
    ("idlee",          "idli"),
    ("iddli",          "idli"),
    ("sambhar",        "sambar"),
    ("sambhaar",       "sambar"),
    ("dosha",          "dosa"),
    ("dhosa",          "dosa"),
    ("upma",           "upma"),
    ("gulab jamon",    "gulab jamun"),
    ("gudlab jamun",   "gulab jamun"),
    ("jalebi",         "jalebi"),
    ("kaju katlie",    "kaju katli"),
    ("kaju kattli",    "kaju katli"),
    ("lasee",          "lassi"),
    ("lasssi",         "lassi"),
    ("chaas",          "chhas"),
    ("chhaas",         "chhas"),
    ("buttermilk",     "chhas"),
    ("khichdee",       "khichdi"),
    ("khichhdi",       "khichdi"),
    ("poha",           "poha"),
    ("pauha",          "poha"),
    ("halwaa",         "halwa"),
    ("upama",          "upma"),
    ("dhokla",         "dhokla"),
    ("dhokhla",        "dhokla"),
    ("fafadaa",        "fafda"),
    ("bhakri",         "bhakri"),
    ("bhakhri",        "bhakri"),
    ("khakhraa",       "khakhra"),
    ("vada pav",       "vada pav"),
    ("wada pav",       "vada pav"),
    ("wada paav",      "vada pav"),
    ("pav bhajii",     "pav bhaji"),
    ("pani poori",     "pani puri"),
    ("bhelpuri",       "bhel puri"),
    ("chole bhature",  "chole bhature"),
    ("masala dossa",   "masala dosa"),
    ("uttapum",        "uttapam"),
    ("chicken tikaa",  "chicken tikka"),
    ("chiken",         "chicken"),
    ("chickeen",       "chicken"),
    ("fish fry",       "fish fry"),
    ("muttan",         "mutton"),
    ("eeg",            "egg"),
    ("eggg",           "egg"),
    ("milck",          "milk"),
    ("doodh",          "milk"),
    ("chaha",          "chai"),
    ("masla chai",     "masala chai"),
    ("grean tea",      "green tea"),
    ("coffe",          "coffee"),
    ("cofee",          "coffee"),
    ("protien shake",  "protein shake"),
    ("wheey protein",  "whey protein"),
    ("oatts",          "oats"),
    ("cornflaakes",    "cornflakes"),
    ("muessli",        "muesli"),
]

TYPO_TEMPLATES: list[str] = [
    "ate {food}",
    "had {food}",
    "I ate {food}",
    "I had {food}",
    "just had {food} for {meal}",
    "{food} khadhu",
    "aaj {food} khadha",
    "aaj {food} khaya",
    "me {food} khadho",
    "{food} piya",
    "drank {food}",
    "{food} khayi for {meal}",
    "had some {food}",
    "eating {food} for {meal}",
    "{food} for my {meal}",
]

def gen_typos(cat="food_typo") -> list[dict]:
    out = []
    for typo, canon in TYPO_MAP:
        for tpl in TYPO_TEMPLATES:
            meal = random.choice(MEALS)
            user = tpl.format(food=typo, meal=meal)
            out.append(ex(user, cat,
                intent="log_food", exact_term=canon, food_query=canon,
                quantity=None, meal_type=meal if "{meal}" in tpl else None,
                foods=[fi(canon, None, None, meal if "{meal}" in tpl else None)],
                missing_detail="quantity",
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MULTI-FOOD
# ─────────────────────────────────────────────────────────────────────────────
MULTI_COMBINATIONS: list[tuple[str, list, str|None]] = [
    # (user_text, foods_list, meal)
    ("I had 2 roti, 1 bowl dal and salad for lunch",
     [fi("roti","2 pieces",None,"lunch"),fi("dal","1 bowl",None,"lunch"),fi("salad","1 serving",None,"lunch")],"lunch"),
    ("Breakfast was idli, sambar and coconut chutney",
     [fi("idli","2 pieces",None,"breakfast"),fi("sambar","1 bowl",None,"breakfast"),fi("coconut chutney","1 serving",None,"breakfast")],"breakfast"),
    ("ate paneer tikka and butter naan for dinner",
     [fi("paneer tikka","1 serving",None,"dinner"),fi("butter naan","1 piece","butter","dinner")],"dinner"),
    ("aaje breakfast ma thepla, chundo ane chai lidha",
     [fi("thepla","2 pieces",None,"breakfast"),fi("chundo","1 serving",None,"breakfast"),fi("chai","1 cup",None,"breakfast")],"breakfast"),
    ("lunch had rajma, chawal and raita",
     [fi("rajma","1 bowl",None,"lunch"),fi("rice","1 bowl",None,"lunch"),fi("raita","1 katori",None,"lunch")],"lunch"),
    ("aaj maine doodh, anda aur toast khayi for breakfast",
     [fi("milk","1 glass",None,"breakfast"),fi("egg","2 pieces","boiled","breakfast"),fi("toast","2 slices",None,"breakfast")],"breakfast"),
    ("dinner was grilled chicken, rice and steamed vegetables",
     [fi("chicken","150g","grilled","dinner"),fi("rice","1 bowl",None,"dinner"),fi("steamed vegetables","1 serving","steamed","dinner")],"dinner"),
    ("had banana, milk and protein shake before gym",
     [fi("banana","1 piece",None,"snack"),fi("milk","1 glass",None,"snack"),fi("protein shake","1 serving",None,"snack")],"snack"),
    ("me sanje bhel ane pani puri khadha",
     [fi("bhel puri","1 plate",None,"snack"),fi("pani puri","1 plate",None,"snack")],"snack"),
    ("oats with milk and boiled egg for breakfast",
     [fi("oats","1 bowl",None,"breakfast"),fi("milk","1 glass",None,"breakfast"),fi("boiled egg","1 piece","boiled","breakfast")],"breakfast"),
    ("raat ko khichdi, kadhi aur papad khayi",
     [fi("khichdi","1 bowl",None,"dinner"),fi("kadhi","1 bowl",None,"dinner"),fi("papad","2 pieces",None,"dinner")],"dinner"),
    ("morning muesli, yogurt and almonds",
     [fi("muesli","1 bowl",None,"breakfast"),fi("yogurt","1 cup",None,"breakfast"),fi("almonds","30g",None,"breakfast")],"breakfast"),
    ("samosa, kachori and jalebi at the fair",
     [fi("samosa","1 piece",None,"snack"),fi("kachori","1 piece",None,"snack"),fi("jalebi","2 pieces",None,"snack")],"snack"),
    ("chole, puri and lassi for lunch",
     [fi("chole","1 bowl",None,"lunch"),fi("puri","2 pieces",None,"lunch"),fi("lassi","1 glass",None,"lunch")],"lunch"),
    ("snacks me chivda, chakli aur chai piya",
     [fi("chivda","1 bowl",None,"snack"),fi("chakli","3 pieces",None,"snack"),fi("chai","1 cup",None,"snack")],"snack"),
    ("dal chawal aur salad for lunch",
     [fi("dal","1 bowl",None,"lunch"),fi("rice","1 bowl",None,"lunch"),fi("salad","1 bowl",None,"lunch")],"lunch"),
    ("sabah oats, ek glass doodh aur banana khayi",
     [fi("oats","1 bowl",None,"breakfast"),fi("milk","1 glass",None,"breakfast"),fi("banana","1 piece",None,"breakfast")],"breakfast"),
    ("dinner me soup, salad aur grilled chicken",
     [fi("soup","1 bowl",None,"dinner"),fi("salad","1 bowl",None,"dinner"),fi("chicken","150g","grilled","dinner")],"dinner"),
    ("had vada pav and masala chai at snack time",
     [fi("vada pav","1 piece",None,"snack"),fi("masala chai","1 cup",None,"snack")],"snack"),
    ("aaj raat me 2 roti, dal aur sabzi khayi",
     [fi("roti","2 pieces",None,"dinner"),fi("dal","1 katori",None,"dinner"),fi("mixed vegetable","1 katori",None,"dinner")],"dinner"),
    ("subah poha aur ek cup chai piya",
     [fi("poha","1 bowl",None,"breakfast"),fi("chai","1 cup",None,"breakfast")],"breakfast"),
    ("me bapore biryani ane raita khadha",
     [fi("biryani","1 plate",None,"lunch"),fi("raita","1 katori",None,"lunch")],"lunch"),
    ("ate 2 idli, 1 dosa and coffee for breakfast",
     [fi("idli","2 pieces",None,"breakfast"),fi("dosa","1 piece",None,"breakfast"),fi("coffee","1 cup",None,"breakfast")],"breakfast"),
    ("sanje fafda, jalebi ane chai khadha",
     [fi("fafda","1 serving",None,"snack"),fi("jalebi","2 pieces",None,"snack"),fi("chai","1 cup",None,"snack")],"snack"),
    ("had curd rice, papad and mango pickle for dinner",
     [fi("curd rice","1 bowl",None,"dinner"),fi("papad","2 pieces",None,"dinner"),fi("mango pickle","1 serving",None,"dinner")],"dinner"),
    ("aaj brunch me 2 egg, toast and orange juice",
     [fi("egg","2 pieces","boiled","breakfast"),fi("toast","2 slices",None,"breakfast"),fi("orange juice","1 glass",None,"breakfast")],"breakfast"),
    ("chocolate cake aur ice cream for dessert",
     [fi("chocolate cake","1 slice",None,"snack"),fi("ice cream","1 scoop",None,"snack")],"snack"),
    ("tiffin me dhokla, chutney aur chai",
     [fi("dhokla","4 pieces",None,"snack"),fi("chutney","1 serving",None,"snack"),fi("chai","1 cup",None,"snack")],"snack"),
    ("aaj lunch me rajma, chawal, raita aur papad",
     [fi("rajma","1 bowl",None,"lunch"),fi("rice","1 bowl",None,"lunch"),fi("raita","1 katori",None,"lunch"),fi("papad","2 pieces",None,"lunch")],"lunch"),
    ("savare paratha, dahi aur lassi",
     [fi("paratha","1 piece",None,"breakfast"),fi("curd","1 katori",None,"breakfast"),fi("lassi","1 glass",None,"breakfast")],"breakfast"),
    ("aaj raat me palak paneer, 2 roti aur raita",
     [fi("palak paneer","1 bowl",None,"dinner"),fi("roti","2 pieces",None,"dinner"),fi("raita","1 katori",None,"dinner")],"dinner"),
    ("chole bhature aur lassi for lunch",
     [fi("chole bhature","1 plate",None,"lunch"),fi("lassi","1 glass",None,"lunch")],"lunch"),
    ("me breakfast ma thepla, pickle ane chai khadha",
     [fi("thepla","2 pieces",None,"breakfast"),fi("pickle","1 serving",None,"breakfast"),fi("chai","1 cup",None,"breakfast")],"breakfast"),
    ("snack me peanut butter toast aur coffee",
     [fi("peanut butter","1 tablespoon",None,"snack"),fi("toast","2 slices",None,"snack"),fi("coffee","1 cup",None,"snack")],"snack"),
    ("had pasta, garlic bread and salad for dinner",
     [fi("pasta","1 bowl",None,"dinner"),fi("garlic bread","2 slices",None,"dinner"),fi("salad","1 bowl",None,"dinner")],"dinner"),
]

MULTI_REPHRASE_PREFIXES = [
    "", "today ", "aaj ", "just now ", "aaje ",
    "FYI ", "logging — ", "Had: ", "Ate: ",
]

def gen_multi_food(cat="multi_food") -> list[dict]:
    out = []
    for user, foods_list, meal in MULTI_COMBINATIONS:
        # original
        main_fq = foods_list[0]["food_query"] if foods_list else None
        main_qty = foods_list[0].get("quantity") if foods_list else None
        out.append(ex(user, cat,
            intent="log_food", exact_term=main_fq, food_query=main_fq,
            quantity=main_qty, meal_type=meal,
            foods=foods_list, missing_detail=None,
        ))
        # 4 rephrased variants
        for pfx in random.sample(MULTI_REPHRASE_PREFIXES[1:], min(4, len(MULTI_REPHRASE_PREFIXES)-1)):
            variant_user = pfx + user[0].lower() + user[1:]
            out.append(ex(variant_user, cat,
                intent="log_food", exact_term=main_fq, food_query=main_fq,
                quantity=main_qty, meal_type=meal,
                foods=foods_list, missing_detail=None,
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EXERCISE (duration)
# ─────────────────────────────────────────────────────────────────────────────
EX_DUR_TEMPLATES: list[str] = [
    "did {dur} minutes of {ex}",
    "I did {dur} min {ex}",
    "completed {dur} minutes {ex}",
    "went for {dur} minute {ex}",
    "aaj {dur} minute {ex} kiya",
    "aaj maine {dur} min {ex} ki",
    "me aaje {dur} minute {ex} karyu",
    "I worked out — {dur} min {ex}",
    "{ex} for {dur} minutes today",
    "did a {dur}-minute {ex} session",
    "had a {dur} min {ex} workout",
    "aaj {dur} min ki {ex}",
    "{dur} minute {ex} kiya aaj",
    "spent {dur} minutes doing {ex}",
    "I just finished {dur} min of {ex}",
]

DURATION_EXERCISES = [e for e in EXERCISES if e[2] == "minutes"]
DURATIONS = [15, 20, 25, 30, 35, 40, 45, 60, 90]

def gen_exercise_duration(cat="exercise_dur") -> list[dict]:
    out = []
    for ex_can, ex_disp, _ in DURATION_EXERCISES:
        for dur in DURATIONS:
            for tpl in EX_DUR_TEMPLATES:
                user = tpl.format(ex=ex_can, dur=dur)
                out.append(ex(user, cat,
                    intent="log_exercise", exact_term=ex_can,
                    exercise_query=ex_can,
                    exercise_input=f"{ex_can} {dur} minutes",
                    duration_min=float(dur),
                    missing_detail=None,
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EXERCISE (reps / sets)
# ─────────────────────────────────────────────────────────────────────────────
EX_REPS_TEMPLATES: list[str] = [
    "did {reps} {ex}",
    "completed {reps} {ex}",
    "I did {reps} {ex} today",
    "aaj {reps} {ex} kiye",
    "me aaje {reps} {ex} kara",
    "{reps} {ex} kiya",
    "aaj maine {reps} {ex} kiye",
    "knocked out {reps} {ex}",
    "finished {reps} {ex}",
    "{reps} reps of {ex}",
]
EX_SETS_TEMPLATES: list[str] = [
    "did {sets} sets of {reps} {ex}",
    "completed {sets} sets of {reps} {ex} each",
    "aaj {sets} sets {reps} reps {ex} kiye",
    "{sets}x{reps} {ex}",
    "I did {sets} sets of {reps} {ex} at the gym",
    "aaj mene {sets} sets of {reps} {ex} kiya",
    "{ex} — {sets} sets {reps} reps",
    "{sets} sets {ex} {reps} reps each",
]

REPS_EXERCISES = [e for e in EXERCISES if e[2] == "reps"]
REPS_VALUES = [10, 12, 15, 20, 25, 30, 50]
SETS_VALUES = [(2,10),(2,12),(3,10),(3,12),(3,15),(4,10),(4,12),(5,20)]

def gen_exercise_reps(cat="exercise_reps") -> list[dict]:
    out = []
    for ex_can, ex_disp, _ in REPS_EXERCISES:
        for reps in REPS_VALUES:
            for tpl in EX_REPS_TEMPLATES:
                user = tpl.format(ex=ex_can, reps=reps)
                out.append(ex(user, cat,
                    intent="log_exercise", exact_term=ex_can,
                    exercise_query=ex_can,
                    exercise_input=f"{reps} {ex_can}",
                    reps=float(reps),
                    missing_detail=None,
                ))
        for sets, reps in SETS_VALUES:
            for tpl in EX_SETS_TEMPLATES:
                user = tpl.format(ex=ex_can, sets=sets, reps=reps)
                out.append(ex(user, cat,
                    intent="log_exercise", exact_term=ex_can,
                    exercise_query=ex_can,
                    exercise_input=f"{sets} sets of {reps} {ex_can}",
                    sets=float(sets), reps=float(reps),
                    missing_detail=None,
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — EXERCISE (distance)
# ─────────────────────────────────────────────────────────────────────────────
EX_DIST_TEMPLATES: list[str] = [
    "{ex} {dist} {unit}",
    "I did {dist} {unit} {ex}",
    "ran {dist} {unit} today",
    "aaj {dist} {unit} {ex} kiya",
    "covered {dist} {unit} {ex}",
    "{dist} {unit} ki {ex} kari",
    "completed {dist} {unit} of {ex}",
    "aaj maine {dist} {unit} {ex} ki",
    "me aaje {dist} {unit} {ex} karyu",
    "{ex} — {dist} {unit}",
]

DIST_EXERCISES = [e for e in EXERCISES if e[2] == "km"]
DIST_VALUES_KM = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]

def gen_exercise_dist(cat="exercise_dist") -> list[dict]:
    out = []
    for ex_can, ex_disp, _ in DIST_EXERCISES:
        for dist in DIST_VALUES_KM:
            for tpl in EX_DIST_TEMPLATES:
                user = tpl.format(ex=ex_can, dist=dist, unit="km")
                out.append(ex(user, cat,
                    intent="log_exercise", exact_term=ex_can,
                    exercise_query=ex_can,
                    exercise_input=f"{ex_can} {dist} km",
                    distance=dist, distance_unit="km",
                    missing_detail=None,
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — EXERCISE, AMOUNT MISSING
# ─────────────────────────────────────────────────────────────────────────────
EX_NOAMT_TEMPLATES: list[str] = [
    "did {ex} today",
    "I did some {ex}",
    "went {ex}",
    "aaj {ex} kiya",
    "me aaje {ex} karyu",
    "just finished {ex}",
    "{ex} kiya aaj",
    "I worked out with {ex}",
    "did my {ex}",
    "aaj maine {ex} ki",
    "{ex} karna tha so kiya",
    "gym me {ex} kiya",
    "{ex} karyu today",
    "aaj {ex} session tha",
    "I went to do {ex}",
]

def gen_exercise_noamt(cat="exercise_noamt") -> list[dict]:
    out = []
    for ex_can, ex_disp, unit in EXERCISES:
        for tpl in EX_NOAMT_TEMPLATES:
            user = tpl.format(ex=ex_can)
            missing = "amount"
            out.append(ex(user, cat,
                intent="log_exercise", exact_term=ex_can,
                exercise_query=ex_can,
                exercise_input=ex_can,
                missing_detail=missing,
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — NUTRITION QUERIES
# ─────────────────────────────────────────────────────────────────────────────
NUT_TEMPLATES: list[tuple[str, str]] = [
    # (template, default_qty)
    ("how many calories in {qty} {food}",         "{qty}"),
    ("calories in {qty} {food}",                  "{qty}"),
    ("how much protein in {qty} {food}",          "{qty}"),
    ("how much fat in {qty} {food}",              "{qty}"),
    ("how many carbs in {qty} {food}",            "{qty}"),
    ("how many calories does {food} have",        "1 serving"),
    ("{food} me kitni calories hoti hai",         "1 serving"),
    ("ek {food} me kitni calories",               "1 piece"),
    ("100g {food} me kitni calories",             "100g"),
    ("{food} ki calorie count",                   "1 serving"),
    ("{qty} {food} mein kitna protein hai",       "{qty}"),
    ("{food} me kitna fat hai",                   "1 serving"),
    ("how much fiber in {food}",                  "1 serving"),
    ("what are the macros of {food}",             "1 serving"),
    ("{food} ki nutrition information",           "1 serving"),
    ("1 {food} mein kitni calories hoti hain",   "1 piece"),
    ("{food} mein kitne carbs hain",              "1 serving"),
    ("{food} mein sugar kitni hai",               "1 serving"),
    ("protein content of {food}",                "100g"),
    ("{qty} {food} calories",                    "{qty}"),
]

def gen_nutrition(cat="nutrition") -> list[dict]:
    out = []
    for food, disp, qty_list in FOODS_EN:
        for tpl, default_qty in NUT_TEMPLATES:
            qty = qty_list[0][0] if qty_list else "1 serving"
            resolved_qty = tpl.replace("{qty}", qty) if "{qty}" in tpl else default_qty
            # extract just the qty part
            if "{qty}" in tpl:
                use_qty = qty
            elif default_qty == "{qty}":
                use_qty = qty
            else:
                use_qty = default_qty
            user = tpl.format(food=food, qty=qty)
            out.append(ex(user, cat,
                intent="get_calories", exact_term=food,
                food_query=food, quantity=use_qty,
            ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — DAILY LOG / SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY_TEMPLATES: list[tuple[str, str, str|None]] = [
    # (user_text, intent, meal_type)
    ("show me today's summary",               "get_summary", None),
    ("aaj ka summary dikhao",                 "get_summary", None),
    ("today's calorie report",                "get_summary", None),
    ("how many calories did I eat today",     "get_summary", None),
    ("total calories today",                  "get_summary", None),
    ("aaje nu total batavo",                  "get_summary", None),
    ("today mera progress dikhao",            "get_summary", None),
    ("mujhe aaj ka log dikhao",               "get_summary", None),
    ("show my daily summary",                 "get_summary", None),
    ("what have I eaten today",               "query_meal",  None),
    ("aaj kya kya khaya",                     "query_meal",  None),
    ("what's my total for today",             "get_summary", None),
    ("daily log dikhao",                      "get_summary", None),
    ("aaj ka full khana log",                 "query_meal",  None),
    ("show me everything I ate today",        "query_meal",  None),
    ("what did I eat for breakfast today",    "query_meal",  "breakfast"),
    ("aaje breakfast ma su khadhu",           "query_meal",  "breakfast"),
    ("what did I have for lunch",             "query_meal",  "lunch"),
    ("aaj lunch ma su khadhu",                "query_meal",  "lunch"),
    ("dinner items today dikhao",             "query_meal",  "dinner"),
    ("aaje dinner ma su khadhu",              "query_meal",  "dinner"),
    ("what snacks did I have today",          "query_meal",  "snack"),
    ("aaj snack ma su khadhu",                "query_meal",  "snack"),
    ("breakfast log show karo",               "query_meal",  "breakfast"),
    ("mera aaj ka khana batao",               "query_meal",  None),
    ("remaining calories for today",          "get_summary", None),
    ("net calories today",                    "get_summary", None),
    ("aaj kitni calories bach rahi hai",      "get_summary", None),
    ("how many calories left today",          "get_summary", None),
    ("calories burned today",                 "get_summary", None),
]

def gen_daily_log(cat="daily_log") -> list[dict]:
    out = []
    for user, intent, meal in SUMMARY_TEMPLATES:
        kw: dict[str, Any] = dict(intent=intent, date="today")
        if meal:
            kw["meal_type"] = meal
        out.append(ex(user, cat, **kw))
        # yesterday variants
        yes_map = {
            "today": "yesterday",
            "aaj":   "gatkale",
            "aaje":  "gatkale",
        }
        yes_user = user
        for tok, rep in yes_map.items():
            if tok in yes_user.lower():
                yes_user = yes_user.lower().replace(tok, rep)
                break
        if yes_user != user:
            kw2 = dict(kw)
            kw2["date"] = "yesterday"
            out.append(ex(yes_user, cat, **kw2))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — YESTERDAY DATE VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
YESTERDAY_TEMPLATES: list[str] = [
    "yesterday I had {qty} {food} for {meal}",
    "gatkale me {food} khadhi",
    "kal {meal} me {food} khaya",
    "kal raat ko {qty} {food} khayi",
    "I ate {food} yesterday for {meal}",
    "gatkal {food} jamyo hato {meal} ma",
    "kal subah {qty} {food} khayi",
    "gatkale me {qty} {food} {meal} ma khadhu",
    "kal dopahar ko {food} khayi",
    "had {food} yesterday evening",
]

def gen_yesterday(cat="yesterday") -> list[dict]:
    out = []
    for food, disp, qty_list in random.sample(FOODS_EN, 30):
        for meal in MEALS:
            qty_str = qty_list[0][0]
            for tpl in YESTERDAY_TEMPLATES:
                user = tpl.format(food=food, qty=qty_str, meal=meal)
                out.append(ex(user, cat,
                    intent="log_food", exact_term=food, food_query=food,
                    quantity=qty_str, meal_type=meal,
                    foods=[fi(food, qty_str, None, meal)],
                    date="yesterday", missing_detail=None,
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — VARIANTS (cooking methods)
# ─────────────────────────────────────────────────────────────────────────────
VARIANT_FOODS: list[tuple[str, list[str]]] = [
    # (food, applicable_variants)
    ("roti",         ["ghee", "butter", "oil"]),
    ("paratha",      ["ghee", "butter", "oil"]),
    ("thepla",       ["ghee", "oil"]),
    ("egg",          ["boiled", "fried", "scrambled"]),
    ("chicken",      ["grilled", "fried", "boiled", "roasted"]),
    ("fish",         ["grilled", "fried", "steamed", "baked"]),
    ("paneer tikka", ["grilled", "fried"]),
    ("aloo",         ["fried", "baked", "boiled"]),
    ("samosa",       ["fried", "baked"]),
    ("idli",         ["steamed"]),
    ("broccoli",     ["steamed", "grilled", "boiled"]),
    ("vegetables",   ["steamed", "boiled", "grilled", "fried"]),
    ("naan",         ["butter", "ghee", "oil"]),
    ("dal",          ["oil", "ghee"]),
    ("salmon",       ["grilled", "baked", "steamed"]),
    ("potato",       ["boiled", "baked", "fried", "roasted"]),
]

VARIANT_TEMPLATES: list[str] = [
    "had {variant} {food}",
    "ate {variant} {food}",
    "I had {food} with {variant}",
    "aaje me {variant} {food} khadhu",
    "aaj {variant} {food} khaya",
    "{food} {variant} khadha",
    "had {qty} {variant} {food} for {meal}",
    "ate {qty} {food} {variant} style for {meal}",
    "{food} {variant} for {meal}",
]

def gen_variants(cat="variants") -> list[dict]:
    out = []
    for food, variants in VARIANT_FOODS:
        # find a matching qty from FOODS_EN if possible
        food_match = next((f for f in FOODS_EN if f[0] == food), None)
        qty_str = food_match[2][0][0] if food_match else "1 serving"
        for var in variants:
            for tpl in VARIANT_TEMPLATES:
                meal = random.choice(MEALS)
                user = tpl.format(food=food, variant=var, qty=qty_str, meal=meal)
                has_meal = "{meal}" in tpl
                has_qty  = "{qty}" in tpl
                out.append(ex(user, cat,
                    intent="log_food", exact_term=food, food_query=food,
                    quantity=qty_str if has_qty else None,
                    variant=var,
                    meal_type=meal if has_meal else None,
                    foods=[fi(food, qty_str if has_qty else None, var, meal if has_meal else None)],
                    missing_detail=None if (has_qty and has_meal) else (
                        "quantity" if not has_qty and has_meal else (
                        "meal_type" if has_qty and not has_meal else None)),
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — PROFILE / BMR / TDEE
# ─────────────────────────────────────────────────────────────────────────────
PROFILE_TEMPLATES: list[str] = [
    "show my profile",
    "mera profile dikhao",
    "what is my BMR",
    "show my BMR and TDEE",
    "what is my daily calorie target",
    "how many calories should I eat",
    "mara profile ni details batavo",
    "my fitness goals",
    "what is my TDEE",
    "how many calories to lose weight",
    "how many calories to gain muscle",
    "what is my activity level",
    "mera calorie target kya hai",
    "show my target macros",
    "update my profile",
    "what is my body weight in the app",
    "show account details",
    "profile page",
    "aaj ka calorie target kya hai",
    "aaje nu calorie target",
    "mera ideal weight kya hai",
    "BMR kitna hai mera",
    "how active am I set to",
    "my goal weight",
    "maintenance calories kya hai",
    "calorie deficit kya rakhu",
    "how many calories to bulk",
    "mera protein target kya hai",
    "aaj kitni protein leni chahiye",
    "target carbs for today",
]

def gen_profile(cat="profile") -> list[dict]:
    out = []
    for user in PROFILE_TEMPLATES:
        out.append(ex(user, cat, intent="get_profile"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — GREETINGS
# ─────────────────────────────────────────────────────────────────────────────
GREETING_LIST: list[str] = [
    "hi", "hello", "hey", "hey there", "namaste", "kem cho",
    "namaskar", "good morning", "good evening", "good night",
    "hello bot", "hi there", "hii", "helo", "heyy",
    "sup", "yo", "howdy", "what's up", "kem cho bot",
    "namaste bot", "kaise ho", "how are you",
    "hi! how are you", "hello, good morning", "hey, kya hal hai",
    "good afternoon", "hola", "bonjour", "greetings",
]

def gen_greetings(cat="greeting") -> list[dict]:
    out = []
    for user in GREETING_LIST:
        out.append(ex(user, cat, intent="greeting"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — JUNK / OFF-TOPIC (must → clarification_needed)
# ─────────────────────────────────────────────────────────────────────────────
JUNK_LIST: list[str] = [
    "abcdef", "qwerty", "xyz123", "asdfgh", "zxcvbn",
    "kjhgfd", "poiuyt", "mnbvcx", "lkjhgf", "opopipiopiop",
    "qazwsxedc", "pqowieuryt", "mnbvcxzlkjh",
    "what is the weather today", "tell me a joke", "book me a cab",
    "play some music", "who is the president", "what's the time",
    "how tall is mount everest", "translate this to french",
    "send an email to my boss", "set an alarm for 6am",
    "what's the latest news", "remind me to call mom",
    "open calculator", "what movies are playing",
    "is it going to rain", "sports score kya hai",
    "tell me a story", "what is 2+2",
    "kjhgfdsa", "weretuiop", "aaabbbccc",
    "random text here", "!!!###", "1234567890",
    "hello how are you doing today and what can you tell me",
    "what do you think about AI",
    "can you write a poem",
    "give me a recipe for pasta",
]

CLARIFICATION_Q = "I couldn't identify a food or exercise. What would you like to log?"

def gen_junk(cat="junk") -> list[dict]:
    out = []
    for user in JUNK_LIST:
        out.append(ex(user, cat,
            intent="clarification_needed",
            clarification_question=CLARIFICATION_Q,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — AMBIGUOUS / GENERIC (vague food → clarification)
# ─────────────────────────────────────────────────────────────────────────────
AMBIG_LIST: list[tuple[str, str]] = [
    ("I ate something",                  "What food did you eat? Please specify the name."),
    ("had a bit of that thing",          "Which food did you have? Please specify."),
    ("kuch kha liya",                    "Kya khaya? Food ka naam batayein."),
    ("ate the usual",                    "Which food is your usual? Please name it."),
    ("had food",                         "Which food? Please specify."),
    ("main ne kuch piya",                "Aapne kya piya? Drink ka naam batayein."),
    ("me kaik khadhu",                   "Su khadhu? Food nu naam batavo."),
    ("had some stuff",                   "What did you eat? Please specify."),
    ("ate whatever was there",           "Which food? Please name it."),
    ("had my regular meal",              "What was your regular meal? Please specify the food."),
    ("kuch khaya tha",                   "Kya khaya? Kripya food ka naam batayein."),
    ("thoda sa khaya",                   "Kya khaya? Please food ka naam batayein."),
    ("had a little snack",               "Which snack? Please specify."),
    ("ate that thing I usually have",    "Which food is that? Please name it."),
    ("had my typical breakfast",         "What was in your breakfast? Please list the foods."),
    ("kuch pi liya",                     "Kya piya? Drink ka naam batao."),
    ("aaje kaik khadhu",                 "Su khadhu? Food nu naam batavo please."),
    ("had some homemade food",           "Which homemade food? Please specify."),
    ("ate the same as yesterday",        "What was that? Please name the food."),
    ("had some drinks",                  "Which drinks? Please specify."),
]

def gen_ambiguous(cat="ambiguous") -> list[dict]:
    out = []
    for user, clar_q in AMBIG_LIST:
        out.append(ex(user, cat,
            intent="clarification_needed",
            clarification_question=clar_q,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 — SKIP MEAL
# ─────────────────────────────────────────────────────────────────────────────
SKIP_TEMPLATES: list[tuple[str, str]] = [
    ("I skipped breakfast today",       "breakfast"),
    ("aaj lunch nahi khaya",            "lunch"),
    ("dinner skip kar liya",            "dinner"),
    ("me aaje nashtu nahi khadho",      "breakfast"),
    ("aaj snack skip kiya",             "snack"),
    ("raat ko kuch nahi khaya",         "dinner"),
    ("breakfast nahi tha aaj",          "breakfast"),
    ("aaj lunch nahi kiya",             "lunch"),
    ("I didn't eat dinner",             "dinner"),
    ("skipped my morning meal",         "breakfast"),
    ("no breakfast today",              "breakfast"),
    ("no lunch today",                  "lunch"),
    ("aaj dinner nahi kiya",            "dinner"),
    ("I fasted today",                  "breakfast"),
    ("me aaje lunch nahi jamyo",        "lunch"),
    ("dinner skip karna pada",          "dinner"),
    ("had nothing for breakfast",       "breakfast"),
    ("aaj subah kuch nahi khaya",       "breakfast"),
    ("raat ko bhukha raha",             "dinner"),
    ("aaj dopahar nahi khaya",          "lunch"),
]

def gen_skip_meal(cat="skip_meal") -> list[dict]:
    out = []
    for user, meal in SKIP_TEMPLATES:
        out.append(ex(user, cat,
            intent="skip_meal", meal_type=meal, date="today",
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 — EXERCISE READ (query_exercise)
# ─────────────────────────────────────────────────────────────────────────────
EX_READ_TEMPLATES: list[tuple[str, str]] = [
    ("what exercise did I do today",          "today"),
    ("aaj mene kya exercise ki",              "today"),
    ("show my workout for today",             "today"),
    ("how many calories did I burn today",    "today"),
    ("aaje me su exercise kari",              "today"),
    ("today's exercise log",                  "today"),
    ("yesterday's workout summary",           "yesterday"),
    ("gatkale meri exercise kya thi",         "yesterday"),
    ("exercise log dikhao",                   "today"),
    ("show my training today",                "today"),
    ("what workouts did I do today",          "today"),
    ("aaj ke exercise batao",                 "today"),
    ("how much did I burn today",             "today"),
    ("gym log today",                         "today"),
    ("aaje nu exercise log",                  "today"),
    ("what did I train today",                "today"),
    ("gatkale workout kya tha",               "yesterday"),
    ("yesterday exercise",                    "yesterday"),
    ("calories burned today",                 "today"),
    ("today workout recap",                   "today"),
]

def gen_exercise_read(cat="exercise_read") -> list[dict]:
    out = []
    for user, date in EX_READ_TEMPLATES:
        out.append(ex(user, cat,
            intent="query_exercise", date=date,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 21 — FOLLOW-UP / CONTEXTUAL
# ─────────────────────────────────────────────────────────────────────────────
FOLLOWUP: list[tuple[str, dict]] = [
    ("2 pieces",
     dict(intent="log_food", exact_term="roti", food_query="roti",
          quantity="2 pieces", meal_type="lunch",
          foods=[fi("roti","2 pieces",None,"lunch")])),
    ("1 bowl",
     dict(intent="log_food", exact_term="dal", food_query="dal",
          quantity="1 bowl", meal_type="dinner",
          foods=[fi("dal","1 bowl",None,"dinner")])),
    ("ghee",
     dict(intent="log_food", exact_term="paratha", food_query="paratha",
          quantity="1 piece", variant="ghee", meal_type="breakfast",
          foods=[fi("paratha","1 piece","ghee","breakfast")])),
    ("breakfast",
     dict(intent="log_food", exact_term="idli", food_query="idli",
          quantity="2 pieces", meal_type="breakfast",
          foods=[fi("idli","2 pieces",None,"breakfast")])),
    ("lunch",
     dict(intent="log_food", exact_term="biryani", food_query="biryani",
          quantity="1 plate", meal_type="lunch",
          foods=[fi("biryani","1 plate",None,"lunch")])),
    ("100g",
     dict(intent="log_food", exact_term="chicken", food_query="chicken",
          quantity="100g", meal_type="dinner",
          foods=[fi("chicken","100g",None,"dinner")])),
    ("30 minutes",
     dict(intent="log_exercise", exact_term="jogging",
          exercise_query="jogging", exercise_input="jogging 30 minutes",
          duration_min=30.0)),
    ("3 sets of 15",
     dict(intent="log_exercise", exact_term="squats",
          exercise_query="squats", exercise_input="3 sets of 15 squats",
          sets=3.0, reps=15.0)),
]

def gen_followup(cat="followup") -> list[dict]:
    out = []
    for user, kw in FOLLOWUP:
        out.append(ex(user, cat, **kw))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ALL  +  DEDUP  +  SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def build_all() -> list[dict]:
    print("Generating sections...")
    sections = {
        "food_qty_en":    gen_food_qty_en,
        "food_noqty":     gen_food_noqty,
        "food_gujarati":  gen_gujarati,
        "food_hindi":     gen_hindi,
        "food_typo":      gen_typos,
        "multi_food":     gen_multi_food,
        "exercise_dur":   gen_exercise_duration,
        "exercise_reps":  gen_exercise_reps,
        "exercise_dist":  gen_exercise_dist,
        "exercise_noamt": gen_exercise_noamt,
        "nutrition":      gen_nutrition,
        "daily_log":      gen_daily_log,
        "yesterday":      gen_yesterday,
        "variants":       gen_variants,
        "profile":        gen_profile,
        "greeting":       gen_greetings,
        "junk":           gen_junk,
        "ambiguous":      gen_ambiguous,
        "skip_meal":      gen_skip_meal,
        "exercise_read":  gen_exercise_read,
        "followup":       gen_followup,
    }
    all_examples: list[dict] = []
    counts: dict[str, int] = {}
    for name, fn in sections.items():
        items = fn()
        counts[name] = len(items)
        all_examples.extend(items)
        print(f"  {name:<20} {len(items):>6}")

    print(f"\nRaw total: {len(all_examples)}")
    return all_examples, counts


def dedup(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove exact-duplicate user utterances (same text → same training signal)."""
    seen: set[str] = set()
    unique: list[dict] = []
    dups = 0
    for e in examples:
        user_text = e["messages"][1]["content"]
        fp = _fp(user_text)
        if fp in seen:
            dups += 1
        else:
            seen.add(fp)
            unique.append(e)
    return unique, dups


def split_train_eval(examples: list[dict], eval_ratio: float = 0.15,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Stratified split: ensure eval set contains examples from every category.
    The eval fingerprint is checked against train to guarantee zero leakage.
    """
    rng = random.Random(seed)
    # Group by category
    by_cat: dict[str, list[dict]] = {}
    for e in examples:
        cat = e.get("category", "unknown")
        by_cat.setdefault(cat, []).append(e)

    train, eval_ = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * eval_ratio))
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])

    # Verify no user-text leakage
    train_fps = {_fp(e["messages"][1]["content"]) for e in train}
    leaked = [e for e in eval_ if _fp(e["messages"][1]["content"]) in train_fps]
    if leaked:
        # Move leaked items from eval → train (keep eval clean)
        eval_ = [e for e in eval_ if _fp(e["messages"][1]["content"]) not in train_fps]
        train.extend(leaked)

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def validate(examples: list[dict]) -> list[str]:
    """Return list of error strings (empty = all valid)."""
    errors = []
    required_keys = {"intent", "date"}
    for i, e in enumerate(examples):
        msgs = e.get("messages", [])
        if len(msgs) != 3:
            errors.append(f"#{i}: expected 3 messages, got {len(msgs)}")
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"#{i}: wrong roles {roles}")
        try:
            j = json.loads(msgs[2]["content"])
        except json.JSONDecodeError as err:
            errors.append(f"#{i}: invalid JSON in assistant: {err}")
            continue
        for k in required_keys:
            if k not in j:
                errors.append(f"#{i}: missing key '{k}' in {j.get('intent')}")
    return errors


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    all_examples, section_counts = build_all()

    # Dedup
    unique, n_dups = dedup(all_examples)
    print(f"Duplicates removed: {n_dups}")
    print(f"Unique examples   : {len(unique)}")

    # Validate
    errs = validate(unique)
    if errs:
        print(f"\nValidation errors ({len(errs)}):")
        for e in errs[:20]:
            print(f"  {e}")
    else:
        print("Validation: all examples valid")

    # Split
    train, eval_ = split_train_eval(unique)
    print(f"\nTrain : {len(train)}")
    print(f"Eval  : {len(eval_)}")

    # Check no leakage
    train_fps = {_fp(e["messages"][1]["content"]) for e in train}
    eval_fps  = {_fp(e["messages"][1]["content"]) for e in eval_}
    overlap   = train_fps & eval_fps
    print(f"Train/eval overlap: {len(overlap)}  (must be 0)")

    # Write JSONL (strip category key — not part of the training format)
    def strip_cat(e: dict) -> dict:
        return {k: v for k, v in e.items() if k != "category"}

    train_path = os.path.join(out_dir, "dataset_train.jsonl")
    eval_path  = os.path.join(out_dir, "dataset_eval.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for e in train:
            f.write(json.dumps(strip_cat(e), ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for e in eval_:
            f.write(json.dumps(strip_cat(e), ensure_ascii=False) + "\n")

    # Stats file
    stats_path = os.path.join(out_dir, "dataset_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("FITNESS CHATBOT — DATASET STATS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Raw generated   : {len(all_examples)}\n")
        f.write(f"After dedup     : {len(unique)}\n")
        f.write(f"Train           : {len(train)}\n")
        f.write(f"Eval            : {len(eval_)}\n")
        f.write(f"Train/eval leak : {len(overlap)}\n")
        f.write(f"Validation errs : {len(errs)}\n\n")
        f.write("Per-section counts (before dedup)\n")
        f.write("-" * 40 + "\n")
        total_raw = 0
        for k, v in sorted(section_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {k:<22} {v:>7}\n")
            total_raw += v
        f.write(f"  {'TOTAL':<22} {total_raw:>7}\n")

    print(f"\nFiles written:")
    print(f"  {train_path}")
    print(f"  {eval_path}")
    print(f"  {stats_path}")
    print(f"\nSection counts:")
    for k, v in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<22} {v:>7}")


if __name__ == "__main__":
    main()
