"""
food_emoji.py - Deterministic rule-based emoji mapping for food, meal, and exercise items.
Assigns granular, relevant emojis based on keyword matching.
Zero LLM calls.
"""
import re

# ============================================================================
# GRANULAR FOOD EMOJI RULES (Evaluated in priority order)
# ============================================================================
_FOOD_RULES = [
    # ── Fast Food, Sandwiches & Prepared Dishes (Specific composite foods) ────
    ("🥪", ["sandwich", "sandwitch", "toast sandwich"]),
    ("🍜", ["noodles", "noodle", "noodels", "nudles", "maggi", "pasta", "spaghetti", "macaroni", "chowmein", "ramen"]),
    ("🌯", ["roll", "frankie", "wrap", "burrito", "taco", "shawarma"]),
    ("🍕", ["pizza", "pizzza", "calzone"]),
    ("🍔", ["burger", "burgur", "hamburger", "cheeseburger"]),
    ("🍟", ["fries", "french fries", "frnch fries"]),

    # ── Eggs, Meat & Seafood (specific protein dishes) ──────────────────────
    ("🥚", ["egg", "eggs", "omelette", "omlet", "boiled egg", "scrambled egg", "egg bhurji", "anda", "egg white", "egg curry", "anda curry"]),
    ("🍗", ["chicken", "turkey", "poultry", "tandoori chicken", "chicken biryani", "chicken tikka"]),
    ("🥩", ["mutton", "lamb", "beef", "pork", "steak", "keema", "kebab", "meat", "bacon", "sausage", "ham"]),
    ("🐟", ["fish", "salmon", "tuna", "pomfret", "surmai", "rohu", "katla", "hilsa", "seafood"]),
    ("🍤", ["prawn", "prawns", "shrimp", "crab", "lobster"]),

    # ── Flatbreads & Breads ────────────────────────────────────────────────
    ("🫓", [
        "roti", "chapati", "chapatti", "paratha", "parotta", "naan", "thepla",
        "thepala", "phulka", "bhakri", "bhakhari", "poori", "puri", "bhatura",
        "kulcha", "flatbread", "khakhra", "rotla", "rotlo"
    ]),
    ("🍞", ["bread", "toast", "pav", "paav", "bun", "bagel", "croissant", "tortilla", "pita"]),

    # ── Beverages & Drinks ─────────────────────────────────────────────────
    ("☕", ["chai", "chaye", "tea", "green tea", "black tea", "coffee", "coffe", "espresso", "latte", "cappuccino", "filter coffee", "frappe"]),
    ("🥛", ["milk", "lassi", "lasssi", "chaas", "chhas", "chhaas", "buttermilk", "smoothie", "milkshake", "shake", "almond milk", "soy milk"]),
    ("🧃", ["juice", "lemonade", "nimbu pani", "coconut water", "cold drink", "soda", "coke", "pepsi", "fanta", "sprite"]),
    ("🍺", ["beer", "wine", "cider", "alcohol", "cocktail"]),
    ("💧", ["water", "mineral water"]),

    # ── Rice & Grains ──────────────────────────────────────────────────────
    ("🍚", ["rice", "biryani", "pulao", "pulav", "khichdi", "khichri", "keechdi", "fried rice", "jeera rice", "curd rice", "lemon rice", "poha"]),
    ("🥣", ["upma", "upmaa", "uppma", "poha", "pohaa", "pohe", "powha", "pauva", "pauwa", "oats", "oatmeal", "daliya", "daliyaa", "porridge", "soup", "stew", "broth", "shorba", "cornflakes", "muesli"]),

    # ── Dal, Lentils, Curries ──────────────────────────────────────────────
    ("🍲", ["dal", "daal", "sambar", "saambar", "sambhar", "rasam", "rajma", "raajma", "chhole", "chole", "chana", "kadhi", "curry", "gravy"]),

    # ── Ice Cream & Frozen Desserts ────────────────────────────────────────
    ("🍦", ["ice cream", "icecream", "kulfi", "gelato", "sundae"]),

    # ── Dairy & Fats ───────────────────────────────────────────────────────
    ("🧀", ["paneer", "cheese", "curd", "dahi", "yogurt", "yoghurt", "tofu"]),
    ("🧈", ["butter", "ghee", "makkhan", "makhan", "malai", "cream", "mayo", "mayonnaise"]),

    # ── Fruits ─────────────────────────────────────────────────────────────
    ("🍎", ["apple", "seb"]),
    ("🍌", ["banana", "kela"]),
    ("🥭", ["mango", "aam"]),
    ("🍊", ["orange", "santra", "mosambi", "citrus", "tangerine"]),
    ("🍓", ["strawberry", "berry", "berries", "blueberry", "blackberry"]),
    ("🍉", ["watermelon", "tarbooz", "melon", "kharbooza", "cantaloupe"]),
    ("🍇", ["grape", "grapes", "angoor"]),
    ("🍍", ["pineapple", "ananas"]),
    ("🥑", ["avocado"]),
    ("🥥", ["coconut", "nariyal"]),
    ("🍋", ["lemon", "lime", "nimbu"]),
    ("🍐", ["pear", "nashpati", "peach", "plum", "cherry"]),
    ("🍈", ["papaya", "guava", "amrood", "pomegranate", "anar", "kiwi", "fig", "dates", "khajoor", "chikoo", "chiku", "custard apple", "sitaphal"]),

    # ── Vegetables & Greens ────────────────────────────────────────────────
    ("🥗", ["salad", "green salad", "kachumber"]),
    ("🥦", ["broccoli", "cauliflower", "gobi", "cabbage", "patta gobi"]),
    ("🥬", ["palak", "spinach", "methi", "lettuce", "greens", "saag"]),
    ("🥕", ["carrot", "gajar", "beetroot", "chukandar"]),
    ("🌽", ["corn", "sweet corn", "makai", "bhutta"]),
    ("🥔", ["potato", "aloo", "alu"]),
    ("🍅", ["tomato", "tamatar"]),
    ("🥒", ["cucumber", "kheera", "kakdi"]),
    ("🍆", ["eggplant", "brinjal", "baingan"]),
    ("🫑", ["capsicum", "bell pepper", "shimla mirch"]),
    ("🍄", ["mushroom", "mushrooms"]),
    ("🧅", ["onion", "pyaz", "garlic", "lehsun", "ginger", "adrak"]),
    ("🌿", ["bhindi", "okra", "ladyfinger", "peas", "matar", "lauki", "doodhi", "bottle gourd", "karela", "bitter gourd", "tinda", "turai", "vegetable", "vegetables", "sabzi", "subzi", "sabji"]),

    # ── Asian, Street Food & Savory Snacks ─────────────────────────────────
    ("🥟", ["momo", "momos", "dumpling", "dumplings", "samosa", "kachori", "pakora", "pakoda", "bhajia", "spring roll"]),
    ("🥞", ["dosa", "uttapam", "pancake", "pancakes", "waffle", "waffles"]),
    ("🥠", ["idli", "vada", "medu vada", "dhokla", "khaman", "gathiya", "fafda", "sev", "bhujia", "chaat", "pani puri", "sev puri", "bhel", "bhelpuri"]),

    # ── Sweets & Desserts ──────────────────────────────────────────────────
    ("🍫", ["chocolate", "brownie"]),
    ("🍰", ["cake", "pastry", "cupcake", "cheesecake"]),
    ("🍩", ["donut", "doughnut"]),
    ("🍪", ["cookie", "cookies", "biscuit", "biscuits"]),
    ("🍬", [
        "gulab jamun", "jalebi", "barfi", "burfi", "halwa", "laddu", "ladoo",
        "kheer", "rasgulla", "rasmalai", "sandesh", "peda", "mysore pak",
        "dessert", "mithai", "pudding", "sweet", "sweets", "sheera", "shrikhand"
    ]),

    # ── Nuts & Seeds ───────────────────────────────────────────────────────
    ("🥜", [
        "peanut", "peanuts", "mungfali", "almond", "almonds", "badam",
        "walnut", "walnuts", "akhrot", "cashew", "cashews", "kaju",
        "pistachio", "pista", "chia", "flax", "pumpkin seed", "sunflower seed",
        "seed", "seeds", "dry fruit", "dry fruits", "raisin", "raisins", "kishmish"
    ]),
]

# Compile food regex rules with word boundary matching
_COMPILED_FOOD_RULES = [
    (emoji, [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords])
    for emoji, keywords in _FOOD_RULES
]

DEFAULT_FOOD_EMOJI = "🍽️"


def get_food_emoji(food_name: str = "", category: str = "") -> str:
    """Return the most relevant emoji for a given food item or category."""
    text = f"{category} {food_name}".strip()
    if not text:
        return DEFAULT_FOOD_EMOJI

    for emoji, patterns in _COMPILED_FOOD_RULES:
        for pat in patterns:
            if pat.search(text):
                return emoji

    return DEFAULT_FOOD_EMOJI


# ============================================================================
# MEAL TYPE EMOJIS
# ============================================================================
_MEAL_EMOJIS = {
    "breakfast": "🍳",
    "nashta": "🍳",
    "nasta": "🍳",
    "nashto": "🍳",
    "નાસ્તો": "🍳",
    "नाश्ता": "🍳",
    "morning": "🌞",
    "savar": "🌞",
    "savare": "🌞",
    "savaar": "🌞",
    "subah": "🌞",
    "subhe": "🌞",
    "સવાર": "🌞",
    "સવારે": "🌞",
    "सुबह": "🌞",
    "lunch": "🍱",
    "bapor": "🍱",
    "bapore": "🍱",
    "dopahar": "🍱",
    "dopaher": "🍱",
    "afternoon": "🍱",
    "બપોર": "🍱",
    "બપોરે": "🍱",
    "दोपहर": "🍱",
    "snack": "☕",
    "snacks": "☕",
    "tea time": "☕",
    "teatime": "☕",
    "chai time": "☕",
    "ચા": "☕",
    "चाय": "☕",
    "evening": "🌇",
    "sanje": "🌇",
    "sanj": "🌇",
    "saanj": "🌇",
    "shaam": "🌇",
    "sham": "🌇",
    "સાંજ": "🌇",
    "સાંજે": "🌇",
    "शाम": "🌇",
    "dinner": "🍽️",
    "raat": "🍽️",
    "raatre": "🍽️",
    "ratre": "🍽️",
    "rate": "🍽️",
    "supper": "🍽️",
    "રાત": "🍽️",
    "રાત્રે": "🍽️",
    "रात": "🍽️",
    "रात्रि": "🍽️",
}

DEFAULT_MEAL_EMOJI = "🍽️"


def get_meal_emoji(meal_type: str = "") -> str:
    """Return relevant emoji for meal types, matching canonical names or substrings."""
    key = (meal_type or "").strip().lower().replace("-", "_")
    if key in _MEAL_EMOJIS:
        return _MEAL_EMOJIS[key]
    for mk, emoji in _MEAL_EMOJIS.items():
        if mk in key:
            return emoji
    return DEFAULT_MEAL_EMOJI


# ============================================================================
# EXERCISE & FITNESS EMOJIS
# ============================================================================
_EXERCISE_RULES = [
    ("🏋️", [
        "strength", "weight", "weights", "weight lifting", "weightlifting", "dumbbell",
        "barbell", "gym", "deadlift", "bench press", "squat", "squats", "overhead press",
        "shoulder press", "bicep curl", "curl", "curls", "lat pulldown", "row", "rows",
        "leg press", "lunge", "lunges", "resistance", "hypertrophy", "bodybuilding",
        "workout", "kasrat", "વર્કઆઉટ", "જિમ", "जिम", "व्यायाम"
    ]),
    ("💪", [
        "push up", "push-up", "pushups", "push-ups", "push ups", "pushup",
        "pull up", "pull-up", "pullups", "pull-ups", "pull ups", "pullup",
        "chin up", "chin-up", "chinups", "chin-ups", "chin ups", "chinup",
        "dips", "dip", "plank", "planks", "crunch", "crunches",
        "sit up", "situps", "sit-up", "sit-ups", "sit ups", "calisthenics", "burpee", "burpees",
        "mountain climber", "jumping jack", "jumping jacks"
    ]),
    ("🏃", [
        "run", "running", "jog", "jogging", "sprint", "sprinting", "treadmill",
        "marathon", "5k", "10k", "cardio", "hiit",
        "dodhvu", "dodvu", "dodyo", "daud", "daudna", "dauda",
        "દોડવું", "દોડ્યો", "दौड़ना"
    ]),
    ("🚴", [
        "cycle", "cycling", "bicycle", "biking", "bike", "spin", "spinning",
        "stationary bike", "indoor cycling",
        "સાયકલ", "સાયકલિંગ", "साइकिल", "साइकिलिंग"
    ]),
    ("🧘", [
        "yoga", "surya namaskar", "pranayama", "asana", "stretching", "stretch",
        "pilates", "meditation", "flexibility",
        "યોગ", "કસરત", "યોગાસન", "योग", "प्राणायाम", "सूर्य नमस्कार"
    ]),
    ("🏊", [
        "swim", "swimming", "freestyle", "breaststroke", "butterfly", "laps", "pool",
        "tarna", "taryu", "taryo", "tarvu", "તરવું", "તર્યો", "તૈરના", "तैरना"
    ]),
    ("⚽", [
        "sport", "sports", "football", "soccer", "cricket", "badminton", "tennis",
        "basketball", "volleyball", "table tennis", "squash", "hockey", "baseball",
        "boxing", "kickboxing", "mma", "martial arts", "golf"
    ]),
    ("🚶", [
        "walk", "walking", "brisk walk", "brisk walking", "stroll", "steps",
        "chalvu", "chalyo", "chalya", "chalna", "tehelna", "sair",
        "ચાલવું", "ચાલ્યો", "ટેહલના", "ઘૂમના", "घूमना", "सैर"
    ]),
]

_COMPILED_EXERCISE_RULES = [
    (emoji, [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords])
    for emoji, keywords in _EXERCISE_RULES
]

DEFAULT_EXERCISE_EMOJI = "🏃"


def get_exercise_emoji(exercise_name: str = "", category: str = "") -> str:
    """Return relevant emoji for workout/fitness/exercise activity."""
    text = f"{category} {exercise_name}".strip()
    if not text:
        return DEFAULT_EXERCISE_EMOJI

    for emoji, patterns in _COMPILED_EXERCISE_RULES:
        for pat in patterns:
            if pat.search(text):
                return emoji

    return DEFAULT_EXERCISE_EMOJI
