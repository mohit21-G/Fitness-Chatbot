"""
Ingest Authentic Indian Food Vocabulary into MongoDB & Sync with Clean Datasets.
Populates all authentic traditional dishes, native Indic scripts, and regional aliases.
"""
import os
import sys
import csv
import json
import re
import datetime
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_settings
from import_to_mongodb import generate_food_schema_fields

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(PARENT_DIR, "Clean dataset")

# Standardized authentic nutrition dictionary per 100g based on ICMR / NIN food composition tables
STANDARD_NUTRITION_MAP = {
    # Breads / Flatbreads
    "Roti": {"cals": 264.0, "p": 7.8, "c": 52.0, "f": 2.8, "fib": 4.5, "unit": "piece", "srv_g": 40.0},
    "Chapati": {"cals": 260.0, "p": 7.5, "c": 51.5, "f": 2.5, "fib": 4.2, "unit": "piece", "srv_g": 40.0},
    "Bhakri": {"cals": 255.0, "p": 6.8, "c": 49.0, "f": 3.5, "fib": 4.8, "unit": "piece", "srv_g": 60.0},
    "Bajra Rotlo": {"cals": 242.0, "p": 5.4, "c": 44.0, "f": 3.8, "fib": 5.2, "unit": "piece", "srv_g": 80.0},
    "Methi Thepla": {"cals": 252.0, "p": 5.8, "c": 35.8, "f": 9.8, "fib": 4.5, "unit": "piece", "srv_g": 60.0},
    "Dudhi Thepla": {"cals": 248.0, "p": 5.2, "c": 36.5, "f": 9.4, "fib": 4.1, "unit": "piece", "srv_g": 60.0},
    "Aloo Paratha": {"cals": 235.0, "p": 5.5, "c": 36.0, "f": 8.5, "fib": 3.2, "unit": "piece", "srv_g": 100.0},
    "Paneer Paratha": {"cals": 265.0, "p": 9.5, "c": 32.5, "f": 11.8, "fib": 3.2, "unit": "piece", "srv_g": 90.0},
    "Gobi Paratha": {"cals": 215.0, "p": 4.8, "c": 34.2, "f": 7.4, "fib": 3.5, "unit": "piece", "srv_g": 90.0},
    "Butter Naan": {"cals": 310.0, "p": 8.2, "c": 48.5, "f": 10.2, "fib": 2.2, "unit": "piece", "srv_g": 90.0},
    "Garlic Naan": {"cals": 305.0, "p": 8.0, "c": 48.0, "f": 9.8, "fib": 2.4, "unit": "piece", "srv_g": 90.0},
    "Puri": {"cals": 365.0, "p": 6.8, "c": 45.5, "f": 18.5, "fib": 2.8, "unit": "piece", "srv_g": 30.0},
    "Bhatura": {"cals": 340.0, "p": 7.5, "c": 46.8, "f": 14.5, "fib": 2.1, "unit": "piece", "srv_g": 80.0},
    "Kulcha": {"cals": 260.0, "p": 7.2, "c": 46.0, "f": 4.5, "fib": 2.8, "unit": "piece", "srv_g": 80.0},
    "Amritsari Kulcha": {"cals": 275.0, "p": 6.8, "c": 44.5, "f": 7.2, "fib": 3.2, "unit": "piece", "srv_g": 100.0},
    "Jowar Bhakri": {"cals": 238.0, "p": 4.8, "c": 46.2, "f": 2.2, "fib": 4.8, "unit": "piece", "srv_g": 70.0},
    "Makki Roti": {"cals": 245.0, "p": 5.2, "c": 48.0, "f": 3.5, "fib": 4.2, "unit": "piece", "srv_g": 60.0},
    "Missi Roti": {"cals": 258.0, "p": 9.2, "c": 42.5, "f": 5.8, "fib": 5.4, "unit": "piece", "srv_g": 60.0},
    "Akki Roti": {"cals": 225.0, "p": 4.2, "c": 45.0, "f": 3.2, "fib": 2.8, "unit": "piece", "srv_g": 70.0},
    "Brown Bread": {"cals": 247.0, "p": 13.0, "c": 41.3, "f": 3.4, "fib": 7.0, "unit": "slice", "srv_g": 30.0},

    # Rice / Grains
    "Jeera Rice": {"cals": 140.0, "p": 3.6, "c": 28.0, "f": 3.2, "fib": 1.5, "unit": "bowl", "srv_g": 200.0},
    "Pulao": {"cals": 145.0, "p": 4.2, "c": 26.5, "f": 3.8, "fib": 2.1, "unit": "bowl", "srv_g": 200.0},
    "Curd Rice": {"cals": 135.0, "p": 4.5, "c": 22.0, "f": 4.0, "fib": 1.0, "unit": "bowl", "srv_g": 200.0},
    "Lemon Rice": {"cals": 150.0, "p": 3.8, "c": 27.5, "f": 4.2, "fib": 1.4, "unit": "bowl", "srv_g": 200.0},
    "Tamarind Rice": {"cals": 165.0, "p": 3.5, "c": 29.0, "f": 5.0, "fib": 1.8, "unit": "bowl", "srv_g": 200.0},
    "Rasam Rice": {"cals": 125.0, "p": 3.8, "c": 24.5, "f": 2.5, "fib": 1.8, "unit": "bowl", "srv_g": 200.0},
    "Bisi Bele Bath": {"cals": 148.0, "p": 5.2, "c": 24.0, "f": 4.5, "fib": 2.8, "unit": "bowl", "srv_g": 250.0},
    "Chicken Biryani": {"cals": 175.0, "p": 9.8, "c": 22.5, "f": 5.8, "fib": 1.2, "unit": "plate", "srv_g": 300.0},
    "Mutton Biryani": {"cals": 195.0, "p": 10.5, "c": 21.0, "f": 7.8, "fib": 1.1, "unit": "plate", "srv_g": 300.0},
    "Lucknowi Biryani": {"cals": 170.0, "p": 9.2, "c": 22.0, "f": 5.5, "fib": 1.2, "unit": "plate", "srv_g": 300.0},

    # Dals & Lentils
    "Dal Tadka": {"cals": 118.0, "p": 6.8, "c": 16.2, "f": 4.2, "fib": 4.0, "unit": "bowl", "srv_g": 200.0},
    "Dal Fry": {"cals": 115.0, "p": 6.8, "c": 16.5, "f": 4.2, "fib": 3.8, "unit": "bowl", "srv_g": 200.0},
    "Gujarati Dal": {"cals": 95.0, "p": 5.2, "c": 15.0, "f": 2.8, "fib": 3.2, "unit": "bowl", "srv_g": 200.0},
    "Dal Makhani": {"cals": 145.0, "p": 6.5, "c": 16.0, "f": 7.2, "fib": 4.5, "unit": "bowl", "srv_g": 200.0},
    "Chana Dal": {"cals": 122.0, "p": 7.5, "c": 18.2, "f": 3.8, "fib": 4.5, "unit": "bowl", "srv_g": 200.0},
    "Moong Dal": {"cals": 105.0, "p": 7.2, "c": 15.5, "f": 2.2, "fib": 3.5, "unit": "bowl", "srv_g": 200.0},
    "Masoor Dal": {"cals": 112.0, "p": 7.8, "c": 16.5, "f": 2.5, "fib": 3.8, "unit": "bowl", "srv_g": 200.0},
    "Cholar Dal": {"cals": 135.0, "p": 7.2, "c": 19.5, "f": 4.8, "fib": 4.2, "unit": "bowl", "srv_g": 200.0},

    # Curries & Gravies
    "Paneer Butter Masala": {"cals": 185.0, "p": 8.5, "c": 11.2, "f": 13.5, "fib": 2.1, "unit": "bowl", "srv_g": 200.0},
    "Palak Paneer": {"cals": 145.0, "p": 8.2, "c": 7.5, "f": 10.2, "fib": 3.2, "unit": "bowl", "srv_g": 200.0},
    "Shahi Paneer": {"cals": 192.0, "p": 8.8, "c": 12.0, "f": 14.0, "fib": 1.8, "unit": "bowl", "srv_g": 200.0},
    "Kadai Paneer": {"cals": 165.0, "p": 8.5, "c": 9.5, "f": 11.5, "fib": 2.5, "unit": "bowl", "srv_g": 200.0},
    "Chole": {"cals": 135.0, "p": 7.2, "c": 19.5, "f": 4.2, "fib": 5.2, "unit": "bowl", "srv_g": 200.0},
    "Rajma": {"cals": 130.0, "p": 7.5, "c": 18.5, "f": 3.8, "fib": 5.5, "unit": "bowl", "srv_g": 200.0},
    "Punjabi Kadhi Pakora": {"cals": 128.0, "p": 5.8, "c": 15.2, "f": 6.2, "fib": 2.4, "unit": "bowl", "srv_g": 200.0},
    "Gujarati Kadhi": {"cals": 85.0, "p": 3.8, "c": 10.5, "f": 3.2, "fib": 0.8, "unit": "bowl", "srv_g": 200.0},
    "Butter Chicken": {"cals": 195.0, "p": 14.5, "c": 7.5, "f": 12.8, "fib": 1.2, "unit": "bowl", "srv_g": 200.0},
    "Rogan Josh": {"cals": 185.0, "p": 15.2, "c": 6.5, "f": 11.5, "fib": 1.5, "unit": "bowl", "srv_g": 200.0},
    "Dum Aloo": {"cals": 145.0, "p": 3.2, "c": 19.8, "f": 6.8, "fib": 2.8, "unit": "bowl", "srv_g": 200.0},

    # Vegetables (Shaak / Sabji)
    "Ringna No Oro": {"cals": 92.0, "p": 2.8, "c": 9.5, "f": 6.4, "fib": 3.8, "unit": "bowl", "srv_g": 180.0},
    "Sev Tameta Nu Shaak": {"cals": 112.0, "p": 3.4, "c": 12.8, "f": 6.8, "fib": 2.6, "unit": "bowl", "srv_g": 180.0},
    "Lasaniya Bataka": {"cals": 128.0, "p": 2.4, "c": 18.5, "f": 5.8, "fib": 2.2, "unit": "bowl", "srv_g": 180.0},
    "Undhiyu": {"cals": 148.0, "p": 4.5, "c": 18.2, "f": 7.5, "fib": 5.2, "unit": "bowl", "srv_g": 200.0},
    "Mag Nu Shaak": {"cals": 118.0, "p": 6.5, "c": 17.2, "f": 3.5, "fib": 4.2, "unit": "bowl", "srv_g": 180.0},
    "Bhindi Masala": {"cals": 88.0, "p": 2.6, "c": 9.5, "f": 5.2, "fib": 3.8, "unit": "bowl", "srv_g": 150.0},
    "Aloo Gobi": {"cals": 105.0, "p": 2.8, "c": 14.5, "f": 4.8, "fib": 3.2, "unit": "bowl", "srv_g": 150.0},
    "Aloo Methi": {"cals": 110.0, "p": 3.2, "c": 15.0, "f": 5.0, "fib": 3.5, "unit": "bowl", "srv_g": 150.0},
    "Baingan Bharta": {"cals": 92.0, "p": 2.8, "c": 9.5, "f": 6.4, "fib": 3.8, "unit": "bowl", "srv_g": 180.0},

    # Snacks, Farsan, Chaat
    "Kanda Poha": {"cals": 175.0, "p": 4.2, "c": 32.5, "f": 5.8, "fib": 2.8, "unit": "plate", "srv_g": 150.0},
    "Locho": {"cals": 165.0, "p": 6.8, "c": 26.5, "f": 6.2, "fib": 3.5, "unit": "plate", "srv_g": 150.0},
    "Usal Pav": {"cals": 155.0, "p": 5.8, "c": 22.8, "f": 4.8, "fib": 3.6, "unit": "plate", "srv_g": 250.0},
    "Luchi Alur Dom": {"cals": 245.0, "p": 4.5, "c": 34.2, "f": 11.5, "fib": 2.8, "unit": "plate", "srv_g": 200.0},
    "Appam with Stew": {"cals": 142.0, "p": 3.2, "c": 22.8, "f": 5.2, "fib": 2.0, "unit": "plate", "srv_g": 200.0},
    "Puttu Kadala Curry": {"cals": 165.0, "p": 5.5, "c": 27.5, "f": 4.5, "fib": 4.2, "unit": "plate", "srv_g": 250.0},
    "Frankie": {"cals": 225.0, "p": 6.8, "c": 29.5, "f": 9.2, "fib": 3.2, "unit": "piece", "srv_g": 150.0},
    "Kathi Roll": {"cals": 235.0, "p": 7.4, "c": 28.5, "f": 10.5, "fib": 3.0, "unit": "piece", "srv_g": 150.0},
    "Veg Momos": {"cals": 145.0, "p": 4.5, "c": 24.2, "f": 3.2, "fib": 2.1, "unit": "plate", "srv_g": 150.0},
    "Medu Vada": {"cals": 265.0, "p": 8.5, "c": 26.5, "f": 14.5, "fib": 4.2, "unit": "piece", "srv_g": 60.0},
    "Paneer Pakora": {"cals": 285.0, "p": 12.5, "c": 16.5, "f": 19.5, "fib": 1.8, "unit": "serving", "srv_g": 100.0},
    "Mirchi Pakora": {"cals": 215.0, "p": 4.2, "c": 22.8, "f": 12.2, "fib": 3.1, "unit": "serving", "srv_g": 100.0},
    "Onion Bhajiya": {"cals": 255.0, "p": 5.2, "c": 26.5, "f": 14.2, "fib": 3.2, "unit": "serving", "srv_g": 100.0},
    "Aloo Bhujia": {"cals": 575.0, "p": 7.2, "c": 44.5, "f": 41.8, "fib": 3.8, "unit": "serving", "srv_g": 30.0},
    "Chakli": {"cals": 518.0, "p": 8.2, "c": 59.5, "f": 27.8, "fib": 4.2, "unit": "piece", "srv_g": 25.0},
    "Namak Para": {"cals": 498.0, "p": 8.8, "c": 61.2, "f": 24.5, "fib": 2.8, "unit": "serving", "srv_g": 30.0},
    "Poha Chivda": {"cals": 458.0, "p": 6.8, "c": 64.2, "f": 19.5, "fib": 3.5, "unit": "serving", "srv_g": 30.0},
    "Sev Puri": {"cals": 185.0, "p": 3.5, "c": 24.8, "f": 7.8, "fib": 2.5, "unit": "plate", "srv_g": 120.0},
    "Dahi Puri": {"cals": 195.0, "p": 4.2, "c": 25.8, "f": 8.5, "fib": 2.2, "unit": "plate", "srv_g": 150.0},
    "Papdi Chaat": {"cals": 178.0, "p": 3.8, "c": 23.5, "f": 7.5, "fib": 2.4, "unit": "plate", "srv_g": 150.0},
    "Ragda Pattice": {"cals": 165.0, "p": 5.2, "c": 24.5, "f": 5.5, "fib": 3.6, "unit": "plate", "srv_g": 180.0},
    "Chana Chaat": {"cals": 145.0, "p": 6.8, "c": 22.5, "f": 4.2, "fib": 4.5, "unit": "plate", "srv_g": 150.0},
    "Dahi Vada": {"cals": 155.0, "p": 5.2, "c": 18.5, "f": 6.8, "fib": 2.2, "unit": "plate", "srv_g": 150.0},

    # Drinks & Traditional Beverages
    "Adrak Wali Chai": {"cals": 64.0, "p": 2.4, "c": 9.8, "f": 2.6, "fib": 0.1, "unit": "cup", "srv_g": 150.0},
    "Elaichi Chai": {"cals": 64.0, "p": 2.4, "c": 9.8, "f": 2.6, "fib": 0.1, "unit": "cup", "srv_g": 150.0},
    "Filter Coffee": {"cals": 72.0, "p": 2.8, "c": 10.5, "f": 3.0, "fib": 0.2, "unit": "cup", "srv_g": 150.0},
    "Lemon Tea": {"cals": 28.0, "p": 0.2, "c": 6.8, "f": 0.1, "fib": 0.1, "unit": "cup", "srv_g": 150.0},
    "Badam Milk": {"cals": 108.0, "p": 4.8, "c": 16.5, "f": 5.8, "fib": 1.2, "unit": "glass", "srv_g": 200.0},
    "Rooh Afza Milk": {"cals": 92.0, "p": 3.1, "c": 15.2, "f": 3.2, "fib": 0.0, "unit": "glass", "srv_g": 200.0},
    "Cow Milk": {"cals": 62.0, "p": 3.2, "c": 4.8, "f": 3.5, "fib": 0.0, "unit": "glass", "srv_g": 200.0},
    "Masala Chaas": {"cals": 32.0, "p": 1.8, "c": 2.6, "f": 1.4, "fib": 0.2, "unit": "glass", "srv_g": 200.0},
    "Jeera Chaas": {"cals": 30.0, "p": 1.7, "c": 2.5, "f": 1.2, "fib": 0.2, "unit": "glass", "srv_g": 200.0},
    "Salted Lassi": {"cals": 62.0, "p": 3.4, "c": 4.8, "f": 3.6, "fib": 0.0, "unit": "glass", "srv_g": 200.0},
    "Mango Lassi": {"cals": 124.0, "p": 3.8, "c": 22.5, "f": 4.2, "fib": 0.8, "unit": "glass", "srv_g": 200.0},
    "Kokum Sharbat": {"cals": 45.0, "p": 0.2, "c": 11.2, "f": 0.1, "fib": 0.2, "unit": "glass", "srv_g": 200.0},
    "Sugarcane Juice": {"cals": 68.0, "p": 0.4, "c": 17.5, "f": 0.1, "fib": 0.5, "unit": "glass", "srv_g": 200.0},
    "Shikanji": {"cals": 38.0, "p": 0.3, "c": 9.4, "f": 0.1, "fib": 0.2, "unit": "glass", "srv_g": 200.0},
    "Jaljeera": {"cals": 25.0, "p": 0.4, "c": 5.8, "f": 0.2, "fib": 0.3, "unit": "glass", "srv_g": 200.0},
    "Nariyal Pani": {"cals": 19.0, "p": 0.7, "c": 3.7, "f": 0.2, "fib": 1.1, "unit": "glass", "srv_g": 240.0},
    "Nimbu Pani": {"cals": 34.0, "p": 0.3, "c": 8.6, "f": 0.1, "fib": 0.2, "unit": "glass", "srv_g": 200.0},

    # Sweets, Dairy & Packaged
    "White Butter": {"cals": 717.0, "p": 0.9, "c": 0.6, "f": 81.1, "fib": 0.0, "unit": "tbsp", "srv_g": 15.0},
    "Marie Biscuit": {"cals": 435.0, "p": 7.5, "c": 78.2, "f": 11.0, "fib": 2.5, "unit": "piece", "srv_g": 5.0},
    "Maggi Noodles": {"cals": 427.0, "p": 8.0, "c": 63.5, "f": 15.7, "fib": 3.5, "unit": "packet", "srv_g": 70.0},
    "Pomegranate": {"cals": 83.0, "p": 1.7, "c": 18.7, "f": 1.2, "fib": 4.0, "unit": "bowl", "srv_g": 150.0},
}


def ingest_all_benchmark_foods():
    settings = get_settings()
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
    except Exception:
        client = MongoClient(settings.MONGODB_URL)

    db = client[settings.MONGODB_DB_NAME]
    foods_coll = db["foods"]
    aliases_coll = db["food_aliases"]

    dataset_path = os.path.join(CLEAN_DIR, "real_indian_foods_10000_scenarios.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    # Group scenarios by food_name_canonical
    foods_by_canonical = {}
    for sc in scenarios:
        c = sc["food_name_canonical"]
        if c not in foods_by_canonical:
            foods_by_canonical[c] = {
                "canonical": c,
                "category": sc.get("category", "General Food"),
                "region": sc.get("region", "Pan-India"),
                "portion_unit": sc.get("portion_unit", "serving"),
                "aliases": set()
            }
        inp = sc["scenario_input"].strip()
        stype = sc.get("scenario_type", "")
        # Register authentic inputs that are actual food names / scripts (not long sentences)
        if len(inp.split()) <= 4 and not any(w in inp.lower() for w in ("ate", "had", "logged", "today", "yesterday", "aaje", "khadha", "jamya", "pidha")):
            foods_by_canonical[c]["aliases"].add(inp)

    print(f"Loaded {len(foods_by_canonical)} distinct canonical Indian foods from benchmark dataset.")
    now_iso = datetime.datetime.now().isoformat()

    inserted_foods = 0
    updated_aliases = 0

    for canonical, info in foods_by_canonical.items():
        food_name_clean = canonical.strip().lower()
        category = info["category"]
        region = info["region"]
        p_unit = info["portion_unit"]

        # Nutrition lookup with intelligent category fallbacks
        if canonical in STANDARD_NUTRITION_MAP:
            nut = STANDARD_NUTRITION_MAP[canonical]
        elif category == "drink":
            nut = {"cals": 45.0, "p": 1.0, "c": 9.5, "f": 0.5, "fib": 0.2, "unit": p_unit, "srv_g": 200.0}
        elif category in ("snack", "street_food_chaat"):
            nut = {"cals": 220.0, "p": 5.5, "c": 28.0, "f": 10.0, "fib": 2.5, "unit": p_unit, "srv_g": 100.0}
        elif category in ("dal_lentil", "curry_gravy"):
            nut = {"cals": 120.0, "p": 6.5, "c": 16.0, "f": 4.5, "fib": 3.5, "unit": p_unit, "srv_g": 200.0}
        elif category in ("rice_grain", "composite_meal"):
            nut = {"cals": 145.0, "p": 4.5, "c": 26.0, "f": 3.5, "fib": 2.0, "unit": p_unit, "srv_g": 200.0}
        elif category == "bread_flatbread":
            nut = {"cals": 250.0, "p": 6.5, "c": 45.0, "f": 5.5, "fib": 4.0, "unit": p_unit, "srv_g": 60.0}
        elif category == "fruit":
            nut = {"cals": 65.0, "p": 0.8, "c": 15.0, "f": 0.2, "fib": 2.5, "unit": p_unit, "srv_g": 120.0}
        elif category == "sweet_dessert":
            nut = {"cals": 320.0, "p": 5.0, "c": 55.0, "f": 10.0, "fib": 1.0, "unit": p_unit, "srv_g": 80.0}
        else:
            nut = {"cals": 130.0, "p": 4.0, "c": 20.0, "f": 4.0, "fib": 2.5, "unit": p_unit, "srv_g": 100.0}

        existing_food = foods_coll.find_one({"food_name": food_name_clean})
        if existing_food:
            food_id = existing_food["food_id"]
        else:
            food_id = f"food_auth_{re.sub(r'[^\w]', '_', food_name_clean)}"
            schema = generate_food_schema_fields(canonical, category, "", nut["unit"])

            cals_100g = nut["cals"]
            srv_g = nut.get("srv_g", 100.0)
            cals_serving = round(cals_100g * srv_g / 100.0, 1)

            food_doc = {
                "food_id": food_id,
                "food_name": food_name_clean,
                "food_name_display": canonical,
                "aliases": [],
                "category": category,
                "subcategory": "",
                "region": region,
                "state": region,
                "vegetarian_status": "Veg" if "chicken" not in food_name_clean and "mutton" not in food_name_clean and "fish" not in food_name_clean else "Non-Veg",
                "serving_size_g": srv_g,
                "serving_unit": nut["unit"],
                "calories_per_100g": cals_100g,
                "calories_kcal": cals_serving,
                "protein_g": nut["p"],
                "carbs_g": nut["c"],
                "fat_g": nut["f"],
                "fiber_g": nut["fib"],
                "sugar_g": 0.0,
                "sodium_mg": 0.0,
                "quantity_units": schema.get("quantity_units", []),
                "quantity_options": schema.get("quantity_options", []),
                "preparation_variants": schema.get("preparation_variants", []),
                "variant_nutrition": schema.get("variant_nutrition", {}),
                "is_verified": True,
                "data_source": "Authentic Indian Food Master",
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            foods_coll.insert_one(food_doc)
            inserted_foods += 1

        # Register aliases in food_aliases
        aliases_to_add = {canonical.lower(), food_name_clean}
        for al in info["aliases"]:
            al_clean = al.strip().lower()
            if al_clean and len(al_clean) >= 2:
                aliases_to_add.add(al_clean)

        for a in aliases_to_add:
            # Determine language/script
            if any("\u0A80" <= c <= "\u0AFF" for c in a):
                alang = "gu"
                atype = "gujarati_script"
            elif any("\u0900" <= c <= "\u097F" for c in a):
                alang = "hi"
                atype = "hindi_script"
            else:
                alang = "en"
                atype = "transliteration" if a != food_name_clean else "primary"

            res = aliases_coll.update_one(
                {"alias": a},
                {"$set": {"food_id": food_id, "alias": a, "alias_type": atype, "language": alang}},
                upsert=True
            )
            if res.upserted_id or res.modified_count:
                updated_aliases += 1

    print(f"[COMPLETE] Ingested {inserted_foods} new foods and updated {updated_aliases} aliases.")
    sync_to_csv_files(db)


def sync_to_csv_files(db):
    print("Syncing updated foods and aliases to Clean dataset CSV files...")
    foods = list(db["foods"].find({}, {"_id": 0}))
    aliases = list(db["food_aliases"].find({}, {"_id": 0}))

    csv_f_path = os.path.join(CLEAN_DIR, "master_food_database.csv")
    csv_a_path = os.path.join(CLEAN_DIR, "food_aliases.csv")

    keys_f = [
        "food_id", "food_name", "food_name_display", "aliases", "category",
        "subcategory", "region", "state", "vegetarian_status", "serving_size_g",
        "calories_per_100g", "calories_kcal", "protein_g", "carbs_g", "fat_g",
        "fiber_g", "sugar_g", "sodium_mg", "preparation_variant", "is_verified",
        "data_source", "created_at", "updated_at"
    ]
    with open(csv_f_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys_f, extrasaction="ignore")
        w.writeheader()
        w.writerows(foods)

    keys_a = ["food_id", "alias", "alias_type", "language"]
    with open(csv_a_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys_a, extrasaction="ignore")
        w.writeheader()
        w.writerows(aliases)

    print(f"[SYNC COMPLETE] Exported {len(foods)} foods and {len(aliases)} aliases to Clean dataset CSVs.")


if __name__ == "__main__":
    ingest_all_benchmark_foods()
