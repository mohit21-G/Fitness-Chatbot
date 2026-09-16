"""
Import food, exercise, and fitness knowledge datasets into MongoDB.
Populates dynamic data-driven schema fields for every food:
- quantity_units
- quantity_options
- preparation_variants
- variant_nutrition

Datasets imported:
1. Clean dataset/master_food_database.csv
2. Clean dataset/food_aliases.csv
3. Clean dataset/master_exercise_database.csv
4. Raw dataset/Indian_Food_Nutrition_Processed.csv
5. Raw dataset/daily_food_nutrition_dataset.csv
6. Raw dataset/india_statewise_food_master.csv
7. Raw dataset/wikipedia_indian_dishes_fitness_master.csv
8. Raw dataset/gym_yoga_exercises_v2.csv

Run: python import_to_mongodb.py
"""
import os
import sys
import csv
import re
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import sync_db, sync_client
from rag_engine import AdvancedRAGEngine

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(PARENT_DIR, "Clean dataset")
RAW_DIR = os.path.join(PARENT_DIR, "Raw dataset")


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        m = re.search(r"(\d+(?:\.\d+)?)", str(val))
        return float(m.group(1)) if m else default
    except (ValueError, TypeError):
        return default


def generate_food_schema_fields(food_name: str, category: str = "", subcategory: str = "", serving_unit: str = "") -> dict:
    """
    Generate data-driven, food-specific quantity and variant schema fields for a MongoDB food document.
    Ensures no irrelevant variants (e.g. no milk in juice, no ghee in fruit, no pieces in dal).
    """
    name = food_name.lower().strip()
    cat = (category or "").lower().strip()
    subcat = (subcategory or "").lower().strip()

    # Helper matchers
    def is_match(terms):
        return any(t in name or t in cat or t in subcat for t in terms)

    # 1. Flatbreads & Breads - Rotlo / Rotla (Millet flatbreads)
    if is_match({"rotlo", "rotla"}):
        return {
            "quantity_units": ["rotlo", "rotla", "piece"],
            "quantity_options": ["1 rotlo", "2 rotla", "3 rotla"],
            "preparation_variants": ["Normal"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
            }
        }

    # 1b. Flatbreads & Breads (Roti, Chapati, Paratha, etc.)
    if is_match({"bhakri", "roti", "chapati", "chapatti", "paratha", "parantha", "naan", "kulcha", "thepla", "khakhra", "puri", "poori", "dalpuri", "dal poori", "bhatura", "flatbread", "tortilla", "pita"}):
        return {
            "quantity_units": ["piece", "serving"],
            "quantity_options": ["1 piece", "1.5 pieces", "2 pieces", "3 pieces", "4 pieces"],
            "preparation_variants": ["Normal", "Ghee", "Oil", "Butter"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                "ghee": {"calorie_multiplier": 1.25, "fat_multiplier": 1.50, "is_estimated": True},
                "oil": {"calorie_multiplier": 1.20, "fat_multiplier": 1.45, "is_estimated": True},
                "butter": {"calorie_multiplier": 1.20, "fat_multiplier": 1.40, "is_estimated": True},
            }
        }

    # 2. Rice, Dal, Grains, Khichdi, Biryani, Pulao, Curries, Sabzi, Sambhar, Rasam, Porridge, Oats, Curd, Raita, Pasta, Noodles
    if is_match({"rice", "dal", "daal", "khichdi", "biryani", "pulao", "chawal", "curry", "gravy", "sambar", "rasam", "lentil", "rajma", "chole", "sabzi", "bhaji", "korma", "kadhi", "upma", "poha", "sheera", "kheer", "halwa", "porridge", "oats", "curd", "dahi", "raita", "yogurt", "sprouts", "muesli", "cereal", "salad", "pasta", "noodle", "chowmein", "macaroni", "spaghetti"}):
        return {
            "quantity_units": ["bowl", "katori", "plate", "g"],
            "quantity_options": ["0.5 bowl", "1 bowl", "1.5 bowls", "2 bowls", "1 plate"],
            "preparation_variants": ["Normal", "Ghee Tadka", "Oil Preparation"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                "ghee tadka": {"calorie_multiplier": 1.20, "fat_multiplier": 1.35, "is_estimated": True},
                "oil preparation": {"calorie_multiplier": 1.15, "fat_multiplier": 1.30, "is_estimated": True},
            }
        }

    # 3. Milk & Dairy Milk Drinks
    if is_match({"milk", "doodh", "dudh", "badam milk", "almond milk", "lassi", "smoothie", "shake"}) and not is_match({"tea", "chai", "coffee"}):
        return {
            "quantity_units": ["glass", "cup", "ml"],
            "quantity_options": ["100ml", "200ml", "250ml", "1 glass"],
            "preparation_variants": ["Whole Milk", "Skimmed Milk", "Low Fat Milk"],
            "variant_nutrition": {
                "whole milk": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                "skimmed milk": {"calorie_multiplier": 0.60, "fat_multiplier": 0.15, "is_estimated": True},
                "low fat milk": {"calorie_multiplier": 0.75, "fat_multiplier": 0.40, "is_estimated": True},
            }
        }

    # 4. Tea & Coffee
    if is_match({"chai", "tea", "green tea", "black tea", "coffee", "latte", "cappuccino", "espresso"}):
        return {
            "quantity_units": ["cup", "glass", "ml"],
            "quantity_options": ["1 cup", "150ml", "200ml", "2 cups"],
            "preparation_variants": ["With Sugar & Milk", "Without Sugar (Black)", "Low Sugar"],
            "variant_nutrition": {
                "with sugar & milk": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                "without sugar (black)": {"calorie_multiplier": 0.10, "fat_multiplier": 0.05, "is_estimated": True},
                "low sugar": {"calorie_multiplier": 0.65, "fat_multiplier": 0.70, "is_estimated": True},
            }
        }

    # 5. Juices, Water, Chaas, Soda & Soups
    if is_match({"juice", "lemonade", "aam panna", "chaas", "buttermilk", "water", "soda", "coke", "pepsi", "sharbat", "drink", "beverage", "soup", "shorba"}):
        return {
            "quantity_units": ["bowl", "cup", "ml"] if "soup" in name or "shorba" in name else ["glass", "ml"],
            "quantity_options": ["1 bowl", "1 cup", "250ml", "300ml"] if "soup" in name or "shorba" in name else ["1 glass", "250ml", "500ml", "1 bottle"],
            "preparation_variants": ["Normal"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False}
            }
        }

    # 6. Dumplings, Finger Snacks, South Indian Tiffins & Sweets Pieces
    if is_match({"momos", "dim sum", "samosa", "kachori", "spring roll", "cutlet", "tikki", "nuggets", "vada", "khaman", "idli", "dosa", "uttapam", "appe", "pav", "bun", "muffin", "cookie", "biscuit", "pastry", "cake", "brownie", "tart", "donut", "doughnut", "gulab jamun", "rasgulla", "jalebi", "peda", "barfi", "laddoo", "laddu", "modak", "chikki", "waffle", "pancake"}):
        return {
            "quantity_units": ["piece", "plate"],
            "quantity_options": ["1 piece", "2 pieces", "3 pieces", "4 pieces"] if is_match({"dosa", "uttapam", "waffle", "pancake", "bun", "muffin", "cake", "brownie"}) else ["2 pieces", "4 pieces", "6 pieces", "1 plate"],
            "preparation_variants": ["Steamed", "Fried", "Tandoori"] if is_match({"momos", "samosa", "kachori", "vada", "khaman", "tikki", "nuggets"}) else ["Normal"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                "steamed": {"calorie_multiplier": 0.85, "fat_multiplier": 0.60, "is_estimated": True},
                "fried": {"calorie_multiplier": 1.35, "fat_multiplier": 1.70, "is_estimated": True},
                "tandoori": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
            }
        }

    # 7. Poultry, Meat, Fish & Eggs
    if is_match({"chicken", "mutton", "lamb", "fish", "egg", "anda", "prawns", "seafood", "omelette", "kebab", "tikka", "meatball", "kofta"}):
        is_piece_item = is_match({"leg", "wings", "tikka", "egg", "anda", "lolipop", "lollipop", "nuggets", "kebab", "meatball", "kofta", "fry", "roast", "piece"})
        return {
            "quantity_units": ["piece", "serving"] if is_piece_item else ["serving", "plate", "g"],
            "quantity_options": ["1 piece", "2 pieces", "3 pieces", "1 serving"] if is_piece_item else ["1 serving", "1 plate", "200g", "300g"],
            "preparation_variants": ["Grilled/Roasted", "Boiled/Steamed", "Curry/Gravy", "Deep Fried"],
            "variant_nutrition": {
                "grilled/roasted": {"calorie_multiplier": 0.90, "fat_multiplier": 0.75, "is_estimated": True},
                "boiled/steamed": {"calorie_multiplier": 0.85, "fat_multiplier": 0.60, "is_estimated": True},
                "curry/gravy": {"calorie_multiplier": 1.15, "fat_multiplier": 1.35, "is_estimated": True},
                "deep fried": {"calorie_multiplier": 1.45, "fat_multiplier": 1.90, "is_estimated": True},
            }
        }

    # 8. Pizza, Toast, Sandwiches, Burgers, Rolls, Wraps
    if is_match({"pizza", "toast", "sandwich", "burger", "roll", "frankie", "wrap", "taco", "burrito"}):
        if "pizza" in name or "pizza" in cat:
            return {
                "quantity_units": ["slice", "personal", "medium"],
                "quantity_options": ["1 slice", "2 slices", "3 slices", "1 personal pizza"],
                "preparation_variants": ["Normal", "Extra Cheese"],
                "variant_nutrition": {
                    "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                    "extra cheese": {"calorie_multiplier": 1.25, "fat_multiplier": 1.40, "is_estimated": True},
                }
            }
        return {
            "quantity_units": ["slice", "piece"],
            "quantity_options": ["1 piece", "2 pieces", "3 pieces", "4 pieces"] if is_match({"roll", "frankie", "wrap", "taco", "burrito", "burger"}) else ["1 slice", "2 slices", "3 slices", "4 slices"],
            "preparation_variants": ["Normal", "Butter Toast", "Grilled"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                "butter toast": {"calorie_multiplier": 1.20, "fat_multiplier": 1.35, "is_estimated": True},
                "grilled": {"calorie_multiplier": 1.05, "fat_multiplier": 1.10, "is_estimated": True},
            }
        }

    # 9. Gujarati Snacks & Farsan
    if is_match({"handvo", "dhokla", "gathiya", "fafda", "patra", "muthiya", "khandvi", "sev", "chivda", "bhujia", "namkeen", "farsan"}):
        return {
            "quantity_units": ["piece", "plate", "g"],
            "quantity_options": ["2 pieces", "4 pieces", "1 plate", "100g"],
            "preparation_variants": ["Normal", "Steamed", "Tadka/Fried"] if is_match({"handvo", "dhokla", "patra", "muthiya", "khandvi"}) else ["Normal"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False},
                "steamed": {"calorie_multiplier": 0.90, "fat_multiplier": 0.70, "is_estimated": True},
                "tadka/fried": {"calorie_multiplier": 1.30, "fat_multiplier": 1.60, "is_estimated": True},
            }
        }

    # 10. Whole Fruits, Nuts & Dry Fruits
    if is_match({"apple", "banana", "orange", "mango", "chikoo", "guava", "pear", "peach", "plum", "kiwi", "papaya", "watermelon", "almond", "cashew", "walnut", "peanut", "raisin", "date", "fig", "pistachio"}):
        return {
            "quantity_units": ["piece", "g", "serving"],
            "quantity_options": ["1 piece", "2 pieces", "3 pieces", "100g"],
            "preparation_variants": ["Normal"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False}
            }
        }

    # 11. Paneer, Tofu, Cheese, Butter & Condiments
    if is_match({"paneer", "tofu", "cheese", "butter", "ghee", "jam", "honey", "chutney", "dip", "sauce", "mayo"}):
        return {
            "quantity_units": ["g", "piece", "tbsp", "serving"],
            "quantity_options": ["50g", "100g", "150g", "200g"],
            "preparation_variants": ["Normal"],
            "variant_nutrition": {
                "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False}
            }
        }

    # 12. Smart Fallback based on unit / category / name
    u_lower = (serving_unit or "").lower()
    if "piece" in u_lower or "slice" in u_lower or "unit" in u_lower or "portion" in u_lower:
        q_u = ["piece", "serving"]
        q_o = ["1 piece", "1.5 pieces", "2 pieces", "3 pieces", "4 pieces"]
    elif "glass" in u_lower or "ml" in u_lower or "cup" in u_lower or "bottle" in u_lower or "can" in u_lower:
        q_u = ["glass", "cup", "ml"]
        q_o = ["1 glass", "250ml", "500ml", "1 bottle"]
    elif "bowl" in u_lower or "katori" in u_lower or "plate" in u_lower:
        q_u = ["bowl", "plate", "g"]
        q_o = ["0.5 bowl", "1 bowl", "1.5 bowls", "2 bowls", "1 plate"]
    else:
        q_u = ["serving", "100g", "plate"]
        q_o = ["1 serving", "1 plate", "100g", "200g"]

    return {
        "quantity_units": q_u,
        "quantity_options": q_o,
        "preparation_variants": ["Normal"],
        "variant_nutrition": {
            "normal": {"calorie_multiplier": 1.0, "fat_multiplier": 1.0, "is_estimated": False}
        }
    }


def import_master_foods(collection) -> int:
    """Import master_food_database.csv into MongoDB 'foods' collection with dynamic fields."""
    filepath = os.path.join(CLEAN_DIR, "master_food_database.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return 0

    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_name = row["food_name"].strip().lower()
            cat = row.get("category", "")
            subcat = row.get("subcategory", "")
            unit = row.get("serving_unit", "serving")

            schema_fields = generate_food_schema_fields(food_name, cat, subcat, unit)

            doc = {
                "food_id": row["food_id"],
                "food_name": food_name,
                "food_name_display": row["food_name_display"],
                "aliases": [a.strip() for a in row.get("aliases", "").split(",") if a.strip()],
                "category": cat,
                "subcategory": subcat,
                "region": row.get("region", ""),
                "state": row.get("state", ""),
                "vegetarian_status": row.get("vegetarian_status", "Veg"),
                "serving_size_g": _safe_float(row.get("serving_size_g"), 100.0),
                "serving_unit": unit,
                "calories_per_100g": _safe_float(row.get("calories_per_100g"), 0.0),
                "calories_kcal": _safe_float(row.get("calories_kcal"), 0.0),
                "protein_g": _safe_float(row.get("protein_g"), 0.0),
                "carbs_g": _safe_float(row.get("carbs_g"), 0.0),
                "fat_g": _safe_float(row.get("fat_g"), 0.0),
                "fiber_g": _safe_float(row.get("fiber_g"), 0.0),
                "sugar_g": _safe_float(row.get("sugar_g"), 0.0),
                "sodium_mg": _safe_float(row.get("sodium_mg"), 0.0),
                "preparation_variant": row.get("preparation_variant", "normal"),
                "quantity_units": schema_fields["quantity_units"],
                "quantity_options": schema_fields["quantity_options"],
                "preparation_variants": schema_fields["preparation_variants"],
                "variant_nutrition": schema_fields["variant_nutrition"],
                "is_verified": True,
                "data_source": "Master Food Database",
            }
            docs.append(doc)

    for doc in docs:
        collection.update_one({"food_id": doc["food_id"]}, {"$set": doc}, upsert=True)

    print(f"  [DONE] Master Foods: {len(docs)} documents imported/updated")
    return len(docs)


def import_raw_indian_foods(collection) -> int:
    """Import Raw dataset/Indian_Food_Nutrition_Processed.csv into 'foods' collection."""
    filepath = os.path.join(RAW_DIR, "Indian_Food_Nutrition_Processed.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return 0

    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_dish = row.get("Dish Name", "").strip()
            if not raw_dish:
                continue

            m = re.match(r"^([^(]+)(?:\(([^)]+)\))?", raw_dish)
            clean_name = m.group(1).strip() if m else raw_dish
            paren_alias = m.group(2).strip() if (m and m.group(2)) else ""

            norm_name = clean_name.lower()
            existing = collection.find_one({"food_name": norm_name})
            schema_fields = generate_food_schema_fields(norm_name, "Indian Dish", "", "100g")

            if existing and existing.get("is_verified"):
                # Enrich verified doc with schema fields if missing
                collection.update_one(
                    {"food_name": norm_name},
                    {"$set": {
                        "quantity_units": schema_fields["quantity_units"],
                        "quantity_options": schema_fields["quantity_options"],
                        "preparation_variants": schema_fields["preparation_variants"],
                        "variant_nutrition": schema_fields["variant_nutrition"],
                    }}
                )
                continue

            cals = _safe_float(row.get("Calories (kcal)"))
            carbs = _safe_float(row.get("Carbohydrates (g)"))
            protein = _safe_float(row.get("Protein (g)"))
            fat = _safe_float(row.get("Fats (g)"))
            fiber = _safe_float(row.get("Fibre (g)"))
            sugar = _safe_float(row.get("Free Sugar (g)"))
            sodium = _safe_float(row.get("Sodium (mg)"))

            slug = re.sub(r"[^\w]", "_", norm_name).strip("_")
            food_id = f"raw_ifn_{slug}"

            aliases = [clean_name.lower()]
            if paren_alias:
                aliases.append(paren_alias.lower())

            doc = {
                "food_id": food_id,
                "food_name": norm_name,
                "food_name_display": clean_name.title(),
                "aliases": aliases,
                "category": "Indian Dish",
                "vegetarian_status": "Veg",
                "serving_size_g": 100.0,
                "serving_unit": "100g",
                "calories_per_100g": cals,
                "calories_kcal": cals,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
                "fiber_g": fiber,
                "sugar_g": sugar,
                "sodium_mg": sodium,
                "preparation_variant": "normal",
                "quantity_units": schema_fields["quantity_units"],
                "quantity_options": schema_fields["quantity_options"],
                "preparation_variants": schema_fields["preparation_variants"],
                "variant_nutrition": schema_fields["variant_nutrition"],
                "is_verified": False,
                "data_source": "Indian Food Nutrition Processed",
            }

            collection.update_one({"food_name": norm_name}, {"$set": doc}, upsert=True)
            count += 1

    print(f"  [DONE] Indian Food Nutrition Processed: {count} documents ingested")
    return count


def import_raw_daily_foods(collection) -> int:
    """Import Raw dataset/daily_food_nutrition_dataset.csv into 'foods' collection."""
    filepath = os.path.join(RAW_DIR, "daily_food_nutrition_dataset.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return 0

    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_item = row.get("Food_Item", "").strip()
            if not raw_item:
                continue

            m = re.match(r"^([^(]+)(?:\(([^)]+)\))?", raw_item)
            clean_name = m.group(1).strip() if m else raw_item
            norm_name = clean_name.lower()
            cat = row.get("Category", "General")

            schema_fields = generate_food_schema_fields(norm_name, cat, "", "serving")
            existing = collection.find_one({"food_name": norm_name})

            if existing and existing.get("is_verified"):
                collection.update_one(
                    {"food_name": norm_name},
                    {"$set": {
                        "quantity_units": schema_fields["quantity_units"],
                        "quantity_options": schema_fields["quantity_options"],
                        "preparation_variants": schema_fields["preparation_variants"],
                        "variant_nutrition": schema_fields["variant_nutrition"],
                    }}
                )
                continue

            cals = _safe_float(row.get("Calories (kcal)"))
            protein = _safe_float(row.get("Protein (g)"))
            carbs = _safe_float(row.get("Carbohydrates (g)"))
            fat = _safe_float(row.get("Fat (g)"))
            fiber = _safe_float(row.get("Fiber (g)"))
            sugar = _safe_float(row.get("Sugars (g)"))
            sodium = _safe_float(row.get("Sodium (mg)"))

            slug = re.sub(r"[^\w]", "_", norm_name).strip("_")
            food_id = f"raw_daily_{slug}"

            doc = {
                "food_id": food_id,
                "food_name": norm_name,
                "food_name_display": clean_name.title(),
                "aliases": [raw_item.lower(), clean_name.lower()],
                "category": cat,
                "vegetarian_status": "Veg",
                "serving_size_g": 100.0,
                "serving_unit": "serving",
                "calories_per_100g": cals,
                "calories_kcal": cals,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
                "fiber_g": fiber,
                "sugar_g": sugar,
                "sodium_mg": sodium,
                "preparation_variant": "normal",
                "quantity_units": schema_fields["quantity_units"],
                "quantity_options": schema_fields["quantity_options"],
                "preparation_variants": schema_fields["preparation_variants"],
                "variant_nutrition": schema_fields["variant_nutrition"],
                "is_verified": False,
                "data_source": "Daily Food Nutrition Dataset",
            }

            collection.update_one({"food_name": norm_name}, {"$set": doc}, upsert=True)
            count += 1

    print(f"  [DONE] Daily Food Nutrition Dataset: {count} documents ingested")
    return count


def import_raw_statewise_foods(collection) -> int:
    """Import Raw dataset/india_statewise_food_master.csv into 'foods' collection."""
    filepath = os.path.join(RAW_DIR, "india_statewise_food_master.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return 0

    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_name = row.get("food_name", "").strip()
            if not food_name:
                continue

            norm_name = food_name.lower()
            subcat = row.get("subcategory", "Indian Regional")
            schema_fields = generate_food_schema_fields(norm_name, subcat, "", "serving")

            existing = collection.find_one({"food_name": norm_name})
            if existing and existing.get("is_verified"):
                collection.update_one(
                    {"food_name": norm_name},
                    {"$set": {
                        "quantity_units": schema_fields["quantity_units"],
                        "quantity_options": schema_fields["quantity_options"],
                        "preparation_variants": schema_fields["preparation_variants"],
                        "variant_nutrition": schema_fields["variant_nutrition"],
                    }}
                )
                continue

            serving_g = _safe_float(row.get("serving_size_g"), 100.0)
            cals_100g = _safe_float(row.get("calories_per_100g"), 0.0)
            cals_kcal = _safe_float(row.get("calories_kcal"), 0.0)
            protein = _safe_float(row.get("protein_g"), 0.0)
            carbs = _safe_float(row.get("carbs_g"), 0.0)
            fat = _safe_float(row.get("fat_g"), 0.0)
            fiber = _safe_float(row.get("fiber_g"), 0.0)

            slug = re.sub(r"[^\w]", "_", norm_name).strip("_")
            food_id = f"raw_statewise_{slug}"

            doc = {
                "food_id": food_id,
                "food_name": norm_name,
                "food_name_display": food_name.title(),
                "aliases": [norm_name],
                "category": subcat,
                "state": row.get("state", ""),
                "vegetarian_status": row.get("food_type", "Veg"),
                "serving_size_g": serving_g,
                "serving_unit": "serving",
                "calories_per_100g": cals_100g,
                "calories_kcal": cals_kcal,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
                "fiber_g": fiber,
                "preparation_variant": "normal",
                "quantity_units": schema_fields["quantity_units"],
                "quantity_options": schema_fields["quantity_options"],
                "preparation_variants": schema_fields["preparation_variants"],
                "variant_nutrition": schema_fields["variant_nutrition"],
                "is_verified": False,
                "data_source": "Statewise Food Master",
            }

            collection.update_one({"food_name": norm_name}, {"$set": doc}, upsert=True)
            count += 1

    print(f"  [DONE] Statewise Food Master: {count} documents ingested")
    return count


def import_raw_wikipedia_foods(collection) -> int:
    """Import Raw dataset/wikipedia_indian_dishes_fitness_master.csv into 'foods' collection."""
    filepath = os.path.join(RAW_DIR, "wikipedia_indian_dishes_fitness_master.csv")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return 0

    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_name = row.get("food_name", "").strip()
            if not food_name:
                continue

            norm_name = food_name.lower()
            cat = row.get("category", "Indian Fitness Food")
            schema_fields = generate_food_schema_fields(norm_name, cat, "", "serving")

            existing = collection.find_one({"food_name": norm_name})
            if existing and existing.get("is_verified"):
                collection.update_one(
                    {"food_name": norm_name},
                    {"$set": {
                        "quantity_units": schema_fields["quantity_units"],
                        "quantity_options": schema_fields["quantity_options"],
                        "preparation_variants": schema_fields["preparation_variants"],
                        "variant_nutrition": schema_fields["variant_nutrition"],
                    }}
                )
                continue

            serving_g = _safe_float(row.get("serving_size_g"), 100.0)
            cals_kcal = _safe_float(row.get("estimated_calories_kcal"), 0.0)
            cals_100g = _safe_float(row.get("estimated_calories_per_100g"), 0.0)
            protein = _safe_float(row.get("estimated_protein_g"), 0.0)
            carbs = _safe_float(row.get("estimated_carbs_g"), 0.0)
            fat = _safe_float(row.get("estimated_fat_g"), 0.0)
            fiber = _safe_float(row.get("estimated_fiber_g"), 0.0)

            slug = re.sub(r"[^\w]", "_", norm_name).strip("_")
            food_id = f"raw_wiki_{slug}"

            doc = {
                "food_id": food_id,
                "food_name": norm_name,
                "food_name_display": food_name.title(),
                "aliases": [norm_name],
                "category": cat,
                "region": row.get("region", ""),
                "vegetarian_status": row.get("vegetarian_status", "Veg"),
                "serving_size_g": serving_g,
                "serving_unit": "serving",
                "calories_per_100g": cals_100g,
                "calories_kcal": cals_kcal,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
                "fiber_g": fiber,
                "preparation_variant": "normal",
                "quantity_units": schema_fields["quantity_units"],
                "quantity_options": schema_fields["quantity_options"],
                "preparation_variants": schema_fields["preparation_variants"],
                "variant_nutrition": schema_fields["variant_nutrition"],
                "is_verified": False,
                "data_source": "Wikipedia Indian Dishes Fitness Master",
            }

            collection.update_one({"food_name": norm_name}, {"$set": doc}, upsert=True)
            count += 1

    print(f"  [DONE] Wikipedia Indian Dishes Fitness Master: {count} documents ingested")
    return count


def import_exercises(collection) -> int:
    """Import master_exercise_database.csv and gym_yoga_exercises_v2.csv into 'exercises' collection."""
    master_path = os.path.join(CLEAN_DIR, "master_exercise_database.csv")
    gym_path = os.path.join(RAW_DIR, "gym_yoga_exercises_v2.csv")

    docs = []
    if os.path.exists(master_path):
        with open(master_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc = {
                    "exercise_id": row["exercise_id"],
                    "exercise_name": row["exercise_name"].strip().lower(),
                    "exercise_name_display": row["exercise_name_display"],
                    "category": row.get("category", "General"),
                    "measurement_unit": row.get("measurement_unit", "minutes"),
                    "calories_per_unit_min": _safe_float(row.get("calories_per_unit_min")),
                    "calories_per_unit_max": _safe_float(row.get("calories_per_unit_max")),
                    "calories_per_unit_avg": _safe_float(row.get("calories_per_unit_avg")),
                    "typical_duration_min": _safe_float(row.get("typical_duration_min"), 30.0),
                    "is_verified": True,
                }
                docs.append(doc)

    if os.path.exists(gym_path):
        with open(gym_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ex_name = row.get("Exercise", "").strip()
                if not ex_name:
                    continue
                norm_name = ex_name.lower()
                cals_unit = _safe_float(row.get("Calories burned per 1 time"))
                raw_unit = row.get("Unit (1 time)", "1 min").lower()
                # Sanitize the malformed per-unit string ("1 rep" → "reps",
                # "1 min hold" → "minutes") so the stored unit is clean.
                from exercise_calculator import normalize_exercise_unit
                unit_str = normalize_exercise_unit(raw_unit)

                slug = re.sub(r"[^\w]", "_", norm_name).strip("_")
                exercise_id = f"ex_gym_{slug}"

                doc = {
                    "exercise_id": exercise_id,
                    "exercise_name": norm_name,
                    "exercise_name_display": ex_name,
                    "category": row.get("Category", "Fitness"),
                    "measurement_unit": unit_str,
                    "calories_per_unit_min": cals_unit * 0.8,
                    "calories_per_unit_max": cals_unit * 1.2,
                    "calories_per_unit_avg": cals_unit,
                    "typical_duration_min": 30.0,
                    "is_verified": False,
                }
                docs.append(doc)

    for doc in docs:
        collection.update_one({"exercise_id": doc["exercise_id"]}, {"$set": doc}, upsert=True)

    print(f"  [DONE] Exercises: {len(docs)} documents imported/updated")
    return len(docs)


def import_aliases(collection) -> int:
    """Import food_aliases.csv and extra aliases into MongoDB 'food_aliases' collection."""
    filepath = os.path.join(CLEAN_DIR, "food_aliases.csv")
    docs = []

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docs.append({
                    "food_id": row["food_id"],
                    "alias": row["alias"].strip().lower(),
                    "alias_type": row.get("alias_type", "primary"),
                    "language": row.get("language", "en"),
                })

    for doc in docs:
        collection.update_one({"alias": doc["alias"], "food_id": doc["food_id"]}, {"$set": doc}, upsert=True)

    print(f"  [DONE] Aliases: {len(docs)} documents imported/updated")
    return len(docs)


def build_knowledge_base(sync_db_client) -> int:
    """Build Advanced RAG knowledge chunks from fitness knowledge datasets."""
    rag_engine = AdvancedRAGEngine()
    chunks = []

    filepath = os.path.join(RAW_DIR, "wikipedia_indian_dishes_fitness_master.csv")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dish = row.get("food_name", "").strip()
                region = row.get("region", "India")
                category = row.get("category", "Dish")
                cals = row.get("estimated_calories_kcal", "")
                prot = row.get("estimated_protein_g", "")

                text = (
                    f"Dish Name: {dish}. Region: {region}. Category: {category}.\n"
                    f"Fitness Context: {dish} is a traditional {region} preparation. "
                    f"It provides a typical estimated profile of ~{cals} kcal and {prot}g protein per serving. "
                    f"Suitable for balanced dietary choices when cooked with minimal oils."
                )

                chunk_list = rag_engine.semantic_chunk_text(
                    text=text,
                    topic=f"Indian Dish Guide: {dish}",
                    category=category,
                    source="Wikipedia Indian Dishes Fitness Master"
                )
                chunks.extend(chunk_list)

    general_knowledge = [
        ("Weight Loss Guidelines", "Nutrition & Fitness",
         "For effective weight loss, maintain a calorie deficit of 300 to 500 kcal below your TDEE. "
         "Prioritize high-protein foods (1.6g to 2.2g per kg bodyweight) to preserve lean muscle mass. "
         "Incorporate regular resistance training 3-4 times per week alongside cardiovascular exercise. "
         "Ensure 7-8 hours of sleep and adequate hydration (3-4 liters of water daily)."),
        ("Muscle Building Protocol", "Strength Training",
         "To build muscle hypertrophy, consume a slight calorie surplus of 250-400 kcal per day. "
         "Target progressive overload in compound resistance exercises like squats, deadlifts, bench press, and rows. "
         "Distribute protein intake evenly across 4-5 meals daily."),
        ("Hydration & Exercise", "Recovery",
         "Proper hydration is crucial for athletic performance and recovery. Drink at least 500ml water 2 hours before workouts "
         "and replenish electrolytes during intense sessions lasting over 60 minutes."),
    ]

    for topic, cat, body in general_knowledge:
        chunk_list = rag_engine.semantic_chunk_text(
            text=body,
            topic=topic,
            category=cat,
            source="Fitness Knowledge Base"
        )
        chunks.extend(chunk_list)

    count = rag_engine.sync_ingest_knowledge_base(chunks, sync_db_client)
    print(f"  [DONE] RAG Knowledge Base: {count} chunks indexed into 'knowledge_chunks'")
    return count


def main():
    print("=" * 65)
    print("  PRODUCTION DATA ARCHITECTURE & DYNAMIC SCHEMA ENRICHMENT")
    print(f"  Target Database: {sync_client.address}")
    print("=" * 65)

    foods_col = sync_db["foods"]
    exercises_col = sync_db["exercises"]
    aliases_col = sync_db["food_aliases"]

    print("\n1. Ingesting & Enriching Master Food Database...")
    f1 = import_master_foods(foods_col)

    print("\n2. Ingesting & Enriching Indian Food Nutrition Processed...")
    f2 = import_raw_indian_foods(foods_col)

    print("\n3. Ingesting & Enriching Daily Food Nutrition Dataset...")
    f3 = import_raw_daily_foods(foods_col)

    print("\n4. Ingesting & Enriching Statewise Food Master...")
    f4 = import_raw_statewise_foods(foods_col)

    print("\n5. Ingesting & Enriching Wikipedia Indian Dishes Fitness Master...")
    f5 = import_raw_wikipedia_foods(foods_col)

    print("\n6. Ingesting Exercises (Master + Gym & Yoga v2)...")
    ex_count = import_exercises(exercises_col)

    print("\n7. Ingesting Food Aliases...")
    alias_count = import_aliases(aliases_col)

    print("\n8. Building Advanced RAG Knowledge Vectors...")
    rag_count = build_knowledge_base(sync_db)

    # Ensure all documents have dynamic schema fields populated
    print("\n9. Verifying Dynamic Schema Enrichment Across ALL Foods...")
    all_foods = list(foods_col.find({}))
    enriched_count = 0
    for doc in all_foods:
        if "quantity_options" not in doc or "preparation_variants" not in doc:
            schema_fields = generate_food_schema_fields(
                doc.get("food_name", ""),
                doc.get("category", ""),
                doc.get("subcategory", ""),
                doc.get("serving_unit", "")
            )
            foods_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "quantity_units": schema_fields["quantity_units"],
                    "quantity_options": schema_fields["quantity_options"],
                    "preparation_variants": schema_fields["preparation_variants"],
                    "variant_nutrition": schema_fields["variant_nutrition"],
                }}
            )
            enriched_count += 1
    print(f"  [DONE] Verified {len(all_foods)} foods! (Enriched {enriched_count} missing documents)")

    # Indexes Creation
    print("\n10. Building Production MongoDB Indexes...")
    foods_col.create_index("food_id", unique=True)
    foods_col.create_index("food_name")
    foods_col.create_index("aliases")
    foods_col.create_index("category")
    foods_col.create_index("data_source")

    exercises_col.create_index("exercise_id", unique=True)
    exercises_col.create_index("exercise_name")
    exercises_col.create_index("category")

    aliases_col.create_index("alias")
    aliases_col.create_index("food_id")

    sync_db["knowledge_chunks"].create_index("chunk_id", unique=True)
    sync_db["knowledge_chunks"].create_index("topic")

    print("  [DONE] Production MongoDB Indexes Successfully Created!")

    print("\n" + "=" * 65)
    print("  PRODUCTION DATA INGESTION COMPLETE")
    print(f"  Total Foods in MongoDB:     {foods_col.count_documents({})}")
    print(f"  Total Exercises in MongoDB: {exercises_col.count_documents({})}")
    print(f"  Total Aliases in MongoDB:   {aliases_col.count_documents({})}")
    print(f"  Total RAG Knowledge Chunks: {sync_db['knowledge_chunks'].count_documents({})}")
    print("=" * 65)


if __name__ == "__main__":
    main()
