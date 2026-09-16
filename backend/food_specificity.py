"""
Food Specificity & Generic Term Clarification Module.

Identifies generic umbrella food terms (e.g. "curry", "sabji", "shak", "fruit",
"juice", "sweet", "soup", "snack", "vegetable") that must NOT be silently
resolved to arbitrary concrete or branded guesses (like "Beef Curry", French cereal,
or "Sabzi Polo"). Instead, these trigger targeted clarification asking the user
for the specific dish.

Distinguishes standalone generic phrases ("I ate some curry", "me shak khadhu",
"had 2 bowls of curry") from specific compound preparations ("bhindi nu shaak",
"egg curry", "apple juice", "tomato soup", "dal makhani").
"""
from __future__ import annotations

import re
from typing import Optional


GENERIC_FOOD_CATEGORIES: dict[str, dict] = {
    "sabji": {
        "terms": {"sabji", "sabzi", "sabjiyan", "sabziyan", "shak", "shaak", "shakbaji", "tarkari"},
        "label": "sabzi / shaak",
        "prompt_en": "Which sabzi or shaak did you have? (e.g., bhindi, paneer, aloo gobi, sev tameta)",
        "prompt_hi": "Aapne kaunsi sabzi khayi? (jaise bhindi, paneer, aloo gobhi, mix veg)",
        "prompt_gu": "Tame kayu shaak jamya? (jem ke bhindi, sev tameta, paneer, aloo gobi)",
    },
    "curry": {
        "terms": {"curry", "curries", "gravy", "gravies", "tari", "salan"},
        "label": "curry",
        "prompt_en": "Which curry did you have? (e.g., egg curry, chicken curry, paneer butter masala, chana masala)",
        "prompt_hi": "Aapne kaunsi curry khayi? (jaise egg curry, chicken curry, paneer masala, chana masala)",
        "prompt_gu": "Tame kayi curry jamya? (jem ke egg curry, chicken curry, paneer curry, chana masala)",
    },
    "fruit": {
        "terms": {"fruit", "fruits", "fal", "phall", "fada"},
        "label": "fruit",
        "prompt_en": "Which fruit did you have? (e.g., apple, banana, orange, papaya)",
        "prompt_hi": "Aapne kaunsa fruit khaya? (jaise kela, seb, santra, papita)",
        "prompt_gu": "Tame kayu fal khadhu? (jem ke kedu, safarjan, santru, papaiyu)",
    },
    "juice": {
        "terms": {"juice", "juices", "ras", "sherbet", "sharbat"},
        "label": "juice",
        "prompt_en": "Which juice did you drink? (e.g., orange juice, mosambi, apple juice, sugarcane)",
        "prompt_hi": "Aapne kaunsa juice piya? (jaise orange, mosambi, apple, ganne ka juice)",
        "prompt_gu": "Tame kayu juice pidhu? (jem ke santra nu juice, mosambi, safarjan)",
    },
    "sweet": {
        "terms": {"sweet", "sweets", "mithai", "mithiyan", "dessert", "desserts"},
        "label": "sweet / dessert",
        "prompt_en": "Which sweet or dessert did you have? (e.g., gulab jamun, kaju katli, rasgulla, ice cream)",
        "prompt_hi": "Aapne kaunsi mithai / dessert khayi? (jaise gulab jamun, kaju katli, rasgulla)",
        "prompt_gu": "Tame kayi mithai khadhi? (jem ke gulab jamun, kaju katli, jalebi)",
    },
    "snack": {
        "terms": {"snack", "snacks", "nasto", "nashta"},
        "label": "snack",
        "prompt_en": "Which snack did you have? (e.g., samosa, dhokla, kachori, roasted chana)",
        "prompt_hi": "Aapne kaunsa snack khaya? (jaise samosa, dhokla, kachori)",
        "prompt_gu": "Tame kayu nasto karyo? (jem ke samosa, dhokla, khakhra, kachori)",
    },
    "vegetable": {
        "terms": {"vegetable", "vegetables", "veggie", "veggies"},
        "label": "vegetable",
        "prompt_en": "Which vegetable did you have? (e.g., boiled veggies, sauteed broccoli, spinach)",
        "prompt_hi": "Aapne kaunsi vegetable khayi? (jaise boiled veggies, broccoli, palak)",
        "prompt_gu": "Tame kayi vegetable khadhi? (jem ke boiled veggies, broccoli, palak)",
    },
    "soup": {
        "terms": {"soup", "soups"},
        "label": "soup",
        "prompt_en": "Which soup did you have? (e.g., tomato soup, sweet corn soup, vegetable clear soup)",
        "prompt_hi": "Aapne kaunsa soup piya? (jaise tomato soup, sweet corn, veg clear soup)",
        "prompt_gu": "Tame kayu soup pidhu? (jem ke tomato soup, sweet corn soup)",
    },
    "drink": {
        "terms": {"drink", "drinks", "beverage", "beverages"},
        "label": "drink",
        "prompt_en": "Which drink did you have? (e.g., lemonade, coconut water, iced tea, soda)",
        "prompt_hi": "Aapne kya drink liya? (jaise nimbu pani, nariyal pani, iced tea)",
        "prompt_gu": "Tame shu drink lidhu? (jem ke nimbu pani, nariyal pani, iced tea)",
    },
    "dal": {
        "terms": {"dal", "daal"},
        "label": "dal",
        "prompt_en": "Which dal did you have? (e.g., moong dal, toor dal, dal tadka, dal makhani)",
        "prompt_hi": "Aapne kaunsi dal khayi? (jaise moong dal, toor dal, dal tadka, dal makhani)",
        "prompt_gu": "Tame kayi dal lidhi? (jem ke toor dal, moong dal, gujarati dal)",
    },
    "chutney": {
        "terms": {"chutney", "chatni"},
        "label": "chutney",
        "prompt_en": "Which chutney did you have? (e.g., green mint chutney, coconut chutney, garlic chutney)",
        "prompt_hi": "Kaunsi chutney thi? (jaise green chutney, coconut chutney, lehsun chutney)",
        "prompt_gu": "Kayi chutney lidhi? (jem ke green chutney, nariyal chutney, lasan ni chutney)",
    },
    "kadhi": {
        "terms": {"kadhi"},
        "label": "kadhi",
        "prompt_en": "Which kadhi did you have? (e.g., gujarati kadhi, punjabi pakora kadhi)",
        "prompt_hi": "Kaunsi kadhi khayi? (jaise gujarati kadhi, pakora kadhi)",
        "prompt_gu": "Kayi kadhi lidhi? (jem ke gujarati kadhi, sindhi kadhi)",
    },
    "sambhar": {
        "terms": {"sambhar", "sambar"},
        "label": "sambar",
        "prompt_en": "Which sambar did you have? (e.g., drumstick sambar, mixed veg sambar)",
        "prompt_hi": "Kaunsa sambar tha? (jaise drumstick sambar, veg sambar)",
        "prompt_gu": "Kayu sambar lidhu? (jem ke veg sambar, drumstick sambar)",
    },
    "rice": {
        "terms": {"rice", "chawal", "bhat", "bhaat"},
        "label": "rice",
        "prompt_en": "Which rice dish did you have? (e.g., plain white rice, brown rice, jeera rice, pulao)",
        "prompt_hi": "Kaunsa rice khaya? (jaise white rice, brown rice, jeera rice, pulao)",
        "prompt_gu": "Kaya rice lidha? (jem ke plain white rice, brown rice, jeera rice, pulao)",
    },
    "rotli": {
        "terms": {"rotli"},
        "label": "rotli",
        "prompt_en": "How many rotli / phulka did you have?",
        "prompt_hi": "Kitni rotli khayi?",
        "prompt_gu": "Ketli rotli jamya?",
    },
}

ALL_GENERIC_TERMS: set[str] = set().union(*(cat["terms"] for cat in GENERIC_FOOD_CATEGORIES.values()))

# Words that do NOT qualify a food into a specific dish.
# If after stripping these, only generic umbrella terms remain, the phrase is GENERIC.
NON_QUALIFIERS: set[str] = {
    "i", "we", "you", "a", "an", "the", "some", "any", "that", "this", "my", "your", "our",
    "had", "have", "has", "having", "ate", "eaten", "eating", "eat",
    "drank", "drunk", "drinking", "took", "take", "taking",
    "just", "already", "finally", "ended", "up", "tried", "treated",
    "thing", "dish", "item", "something", "stuff", "type", "types",
    "random", "usual", "regular", "normal", "simple", "light", "quick",
    "plain", "homemade", "fresh", "hot", "cold",
    # Hindi / Hinglish / Gujarati fillers & verbs
    "me", "maine", "mene", "mein", "main", "hu", "hoon", "yaar",
    "aaj", "aaje", "aj", "aje", "to", "bhi", "pan", "pn",
    "koi", "kuch", "kai", "kaik", "thodu", "thoda", "thodi", "si", "sa",
    "ek", "do", "tin", "be", "tran", "one", "two", "three", "1", "2", "3", "4", "5",
    "jevu", "jeva", "jevi", "hattu", "hatu", "hata", "hati", "tha", "thi", "the",
    "wo", "te", "lidhu", "lidhi", "lidha", "lidho", "khadhu", "khadhi", "khadha",
    "khadho", "khaya", "khayi", "khaye", "khai", "jamya", "jamyu", "jamva",
    "piya", "piyi", "piye", "pi", "li", "liya", "liye", "liyu", "banavu", "chhu",
    "bas", "hi", "sirf", "matra", "only", "bit", "lot", "plate", "plates",
    "bowl", "bowls", "katori", "katoris", "vatki", "vatkis", "glass", "glasses",
    "cup", "cups", "serving", "servings", "piece", "pieces",
    "for", "at", "in", "after", "before", "with", "and", "or", "of",
    "breakfast", "lunch", "dinner", "morning", "evening", "night",
    "savare", "bapore", "sanje", "raatre", "dopahar", "subah", "shaam", "raat",
    "sathe", "saath", "ane", "aur", "ne", "ke", "ka", "ki", "nu", "ni", "na",
}


def is_generic_food_query(text: str) -> tuple[bool, Optional[str]]:
    """
    Check if `text` is a standalone generic umbrella food query that requires
    clarification rather than an arbitrary database guess.

    Returns:
        (True, category_key) if standalone generic.
        (False, None) if specific or non-generic.
    """
    if not text:
        return False, None

    # Tokenize words
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return False, None

    content = [t for t in tokens if t not in NON_QUALIFIERS]
    if not content:
        # Check if any original token was a generic term (e.g. "had a drink")
        generic_in_tokens = [t for t in tokens if t in ALL_GENERIC_TERMS]
        if generic_in_tokens:
            for cat, data in GENERIC_FOOD_CATEGORIES.items():
                if any(t in data["terms"] for t in generic_in_tokens):
                    return True, cat
            return True, "generic"
        return False, None

    # If all remaining content tokens belong to our generic categories, it's generic!
    if all(t in ALL_GENERIC_TERMS for t in content):
        for cat, data in GENERIC_FOOD_CATEGORIES.items():
            if any(t in data["terms"] for t in content):
                return True, cat
        return True, "generic"

    return False, None


def get_generic_clarification_message(cat: Optional[str], lang: str = "en") -> str:
    """Return a natural multilingual clarification question asking for the specific dish."""
    lang = (lang or "en").lower()
    if cat and cat in GENERIC_FOOD_CATEGORIES:
        data = GENERIC_FOOD_CATEGORIES[cat]
        if lang.startswith("gu"):
            return data["prompt_gu"]
        elif lang.startswith("hi"):
            return data["prompt_hi"]
        return data["prompt_en"]

    if lang.startswith("gu"):
        return "Tame kayu specific food / shaak jamya te janavo? (jem ke bhindi, paneer, dal)"
    elif lang.startswith("hi"):
        return "Aapne kaunsi specific dish khayi? (jaise bhindi, paneer, dal)"
    return "Could you specify which dish you had? (e.g., bhindi, paneer, chicken curry, dal)"
