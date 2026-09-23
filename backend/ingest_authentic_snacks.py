"""
Ingest Comprehensive Authentic Indian Snack and Farsan Catalog into MongoDB.
Covers Gujarati/Kathiyawadi, North, South, East, West Indian snacks, healthy/roasted,
street food, and packaged namkeens with standard ICMR/NIN nutrition and Indic aliases.
"""
import os
import sys
import csv
import re
import datetime
from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_settings
from import_to_mongodb import generate_food_schema_fields

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(PARENT_DIR, "Clean dataset")

AUTHENTIC_SNACKS_CATALOG = [
    # ── 1. Gujarati / Kathiyawadi Farsan ──────────────────────────────────────
    {
        "name": "Bhavnagari Gathiya", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 100.0,
        "cals": 520.0, "p": 14.0, "c": 42.0, "f": 34.0, "fib": 6.0,
        "gu_script": "ભાવનગરી ગાંઠિયા", "gu_translit": "bhavnagari gathiya",
        "hi_script": "भावनगरी गांठिया", "hi_translit": "bhavnagari gathiya",
        "aliases": ["bhavnagri gathiya", "bhavnagari ganthiya", "gathiya bhavnagari"]
    },
    {
        "name": "Vanela Gathiya", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 100.0,
        "cals": 510.0, "p": 13.0, "c": 44.0, "f": 32.0, "fib": 5.5,
        "gu_script": "વાણેલા ગાંઠિયા", "gu_translit": "vanela gathiya",
        "hi_script": "वनेला गांठिया", "hi_translit": "vanela gathiya",
        "aliases": ["vanela ganthiya", "kathiyawadi vanela gathiya", "hot vanela gathiya"]
    },
    {
        "name": "Tikha Gathiya", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 100.0,
        "cals": 525.0, "p": 14.0, "c": 41.0, "f": 35.0, "fib": 6.0,
        "gu_script": "તીખા ગાંઠિયા", "gu_translit": "tikha gathiya",
        "hi_script": "तीखा गांठिया", "hi_translit": "teekha gathiya",
        "aliases": ["spicy gathiya", "tikha ganthiya", "teekha gathiya"]
    },
    {
        "name": "Papdi Gathiya", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 100.0,
        "cals": 515.0, "p": 13.0, "c": 43.0, "f": 33.0, "fib": 5.8,
        "gu_script": "પાપડી ગાંઠિયા", "gu_translit": "papdi gathiya",
        "hi_script": "पापड़ी गांठिया", "hi_translit": "papdi gathiya",
        "aliases": ["papdi farsan", "besan papdi", "gujarati papdi"]
    },
    {
        "name": "Champakali Gathiya", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 100.0,
        "cals": 530.0, "p": 14.0, "c": 40.0, "f": 36.0, "fib": 5.5,
        "gu_script": "ચંપાકલી ગાંઠિયા", "gu_translit": "champakali gathiya",
        "hi_script": "चंपाकली गांठिया", "hi_translit": "champakali gathiya",
        "aliases": ["champakali ganthiya"]
    },
    {
        "name": "Mari Gathiya", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 100.0,
        "cals": 515.0, "p": 13.5, "c": 42.0, "f": 33.0, "fib": 6.0,
        "gu_script": "મરી ગાંઠિયા", "gu_translit": "mari gathiya",
        "hi_script": "काली मिर्च गांठिया", "hi_translit": "kali mirch gathiya",
        "aliases": ["black pepper gathiya", "mari ganthiya"]
    },
    {
        "name": "Fafda", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 100.0,
        "cals": 485.0, "p": 12.5, "c": 48.0, "f": 28.0, "fib": 5.2,
        "gu_script": "ફાફડા", "gu_translit": "fafda",
        "hi_script": "फाफड़ा", "hi_translit": "fafda",
        "aliases": ["gujarati fafda", "crispy fafda", "fafda jalebi snack"]
    },
    {
        "name": "Nylon Khaman", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 150.0,
        "cals": 160.0, "p": 6.5, "c": 28.0, "f": 2.5, "fib": 3.0,
        "gu_script": "નાયલોન ખમણ", "gu_translit": "nylon khaman",
        "hi_script": "नायलॉन खमन", "hi_translit": "nylon khaman",
        "aliases": ["surti khaman", "spongy khaman", "khaman dhokla", "soft nylon khaman"]
    },
    {
        "name": "Amiri Khaman", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 150.0,
        "cals": 220.0, "p": 7.5, "c": 32.0, "f": 7.5, "fib": 3.5,
        "gu_script": "અમીરી ખમણ", "gu_translit": "amiri khaman",
        "hi_script": "अमीरी खमन", "hi_translit": "amiri khaman",
        "aliases": ["sev khamani", "crushed khaman with sev"]
    },
    {
        "name": "Vati Dal Khaman", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 150.0,
        "cals": 175.0, "p": 8.0, "c": 29.0, "f": 3.0, "fib": 4.2,
        "gu_script": "વાટી દાળના ખમણ", "gu_translit": "vati dal na khaman",
        "hi_script": "वाटी दाल खमन", "hi_translit": "vati dal khaman",
        "aliases": ["vati dal khaman", "chana dal khaman"]
    },
    {
        "name": "Khatta Dhokla", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 150.0,
        "cals": 150.0, "p": 5.5, "c": 27.0, "f": 2.0, "fib": 2.5,
        "gu_script": "ખાટા ઢોકળા", "gu_translit": "khatta dhokla",
        "hi_script": "खट्टा ढोकला", "hi_translit": "khatta dhokla",
        "aliases": ["white dhokla", "idada", "safed dhokla", "fermented dhokla"]
    },
    {
        "name": "Rava Dhokla", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 150.0,
        "cals": 165.0, "p": 4.8, "c": 30.0, "f": 3.0, "fib": 2.0,
        "gu_script": "રવા ઢોકળા", "gu_translit": "rava dhokla",
        "hi_script": "रवा ढोकला", "hi_translit": "suji dhokla",
        "aliases": ["suji dhokla", "semolina dhokla", "instant rava dhokla"]
    },
    {
        "name": "Sandwich Dhokla", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 150.0,
        "cals": 170.0, "p": 5.8, "c": 28.0, "f": 4.0, "fib": 2.5,
        "gu_script": "સેન્ડવિચ ઢોકળા", "gu_translit": "sandwich dhokla",
        "hi_script": "सैंडविच ढोकला", "hi_translit": "sandwich dhokla",
        "aliases": ["layer dhokla", "tri color dhokla"]
    },
    {
        "name": "Khandvi", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 120.0,
        "cals": 140.0, "p": 5.2, "c": 18.0, "f": 5.5, "fib": 2.2,
        "gu_script": "ખાંડવી", "gu_translit": "khandvi",
        "hi_script": "खांडवी", "hi_translit": "khandvi",
        "aliases": ["surti khandvi", "suralichi vadi", "patuli"]
    },
    {
        "name": "Handvo", "category": "snack", "region": "Gujarat", "unit": "piece", "srv_g": 100.0,
        "cals": 185.0, "p": 7.0, "c": 27.0, "f": 5.5, "fib": 4.5,
        "gu_script": "હાંડવો", "gu_translit": "handvo",
        "hi_script": "हांडवो", "hi_translit": "handvo",
        "aliases": ["gujarati handvo", "baked handvo", "vegetable handvo"]
    },
    {
        "name": "Patra", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 120.0,
        "cals": 175.0, "p": 5.5, "c": 26.0, "f": 5.8, "fib": 4.0,
        "gu_script": "પાત્રા", "gu_translit": "patra",
        "hi_script": "पात्रा", "hi_translit": "patra",
        "aliases": ["patode", "alu vadi", "taro leaves roll", "steamed patra"]
    },
    {
        "name": "Surti Locho", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 150.0,
        "cals": 165.0, "p": 6.8, "c": 26.5, "f": 6.2, "fib": 3.5,
        "gu_script": "સુરતી લોચો", "gu_translit": "surti locho",
        "hi_script": "सुरती लोचो", "hi_translit": "surati locho",
        "aliases": ["locho", "butter locho", "oil locho", "cheese locho"]
    },
    {
        "name": "Methi Khakhra", "category": "snack", "region": "Gujarat", "unit": "piece", "srv_g": 30.0,
        "cals": 410.0, "p": 12.0, "c": 68.0, "f": 10.0, "fib": 8.0,
        "gu_script": "મેથી ખાખરા", "gu_translit": "methi khakhra",
        "hi_script": "मेथी खाखरा", "hi_translit": "methi khakhra",
        "aliases": ["fenugreek khakhra", "crispy methi khakhra", "diet methi khakhra"]
    },
    {
        "name": "Jeera Khakhra", "category": "snack", "region": "Gujarat", "unit": "piece", "srv_g": 30.0,
        "cals": 405.0, "p": 11.0, "c": 70.0, "f": 9.5, "fib": 7.5,
        "gu_script": "જીરું ખાખરા", "gu_translit": "jeera khakhra",
        "hi_script": "जीरा खाखरा", "hi_translit": "jeera khakhra",
        "aliases": ["cumin khakhra", "jira khakhra"]
    },
    {
        "name": "Chorafali", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 60.0,
        "cals": 460.0, "p": 16.0, "c": 54.0, "f": 20.0, "fib": 7.0,
        "gu_script": "ચોરાફળી", "gu_translit": "chorafali",
        "hi_script": "चोराफली", "hi_translit": "chorafali",
        "aliases": ["chola fali", "cholafali", "diwali chorafali"]
    },
    {
        "name": "Mathiya", "category": "snack", "region": "Gujarat", "unit": "piece", "srv_g": 25.0,
        "cals": 440.0, "p": 13.0, "c": 58.0, "f": 18.0, "fib": 6.0,
        "gu_script": "મઠીયા", "gu_translit": "mathiya",
        "hi_script": "मठिया", "hi_translit": "mathiya",
        "aliases": ["gujarati mathiya", "diwali mathiya"]
    },
    {
        "name": "Sev Mamra", "category": "snack", "region": "Gujarat", "unit": "bowl", "srv_g": 80.0,
        "cals": 430.0, "p": 8.5, "c": 68.0, "f": 14.0, "fib": 4.5,
        "gu_script": "સેવ મમરા", "gu_translit": "sev mamra",
        "hi_script": "सेव मुरमुरा", "hi_translit": "sev murmura",
        "aliases": ["sev murmura", "lasaniya sev mamra", "garlic sev mamra"]
    },
    {
        "name": "Vagharela Mamra", "category": "snack", "region": "Gujarat", "unit": "bowl", "srv_g": 60.0,
        "cals": 380.0, "p": 6.5, "c": 74.0, "f": 7.0, "fib": 3.5,
        "gu_script": "વઘારેલા મમરા", "gu_translit": "vagharela mamra",
        "hi_script": "तड़का मुरमुरा", "hi_translit": "tadka murmura",
        "aliases": ["roasted murmura", "masala mamra", "puffed rice snack"]
    },
    {
        "name": "Lilva Kachori", "category": "snack", "region": "Gujarat", "unit": "piece", "srv_g": 50.0,
        "cals": 320.0, "p": 8.0, "c": 36.0, "f": 16.0, "fib": 4.5,
        "gu_script": "લીલવા કચોરી", "gu_translit": "lilva kachori",
        "hi_script": "लीलवा कचौरी", "hi_translit": "lilva kachori",
        "aliases": ["fresh tuver kachori", "gujarati kachori"]
    },
    {
        "name": "Methi Na Gota", "category": "snack", "region": "Gujarat", "unit": "plate", "srv_g": 120.0,
        "cals": 245.0, "p": 6.8, "c": 30.0, "f": 11.0, "fib": 4.0,
        "gu_script": "મેથીના ગોટા", "gu_translit": "methi na gota",
        "hi_script": "मेथी के पकोड़े", "hi_translit": "methi ke pakode",
        "aliases": ["dakor na gota", "methi gota", "fenugreek pakora"]
    },

    # ── 2. North Indian Snacks & Namkeens ────────────────────────────────────
    {
        "name": "Samosa", "category": "snack", "region": "North India", "unit": "piece", "srv_g": 80.0,
        "cals": 262.0, "p": 4.5, "c": 32.0, "f": 13.0, "fib": 2.5,
        "gu_script": "સમોસા", "gu_translit": "samosa",
        "hi_script": "समोसा", "hi_translit": "samosa",
        "aliases": ["aloo samosa", "crispy samosa", "punjabi samosa"]
    },
    {
        "name": "Pyaaz Kachori", "category": "snack", "region": "Rajasthan", "unit": "piece", "srv_g": 100.0,
        "cals": 340.0, "p": 6.2, "c": 38.0, "f": 18.0, "fib": 3.5,
        "gu_script": "ડુંગળી કચોરી", "gu_translit": "dungli kachori",
        "hi_script": "प्याज़ कचौरी", "hi_translit": "pyaaz kachori",
        "aliases": ["jodhpur pyaaz kachori", "onion kachori", "rajasthani kachori"]
    },
    {
        "name": "Khasta Kachori", "category": "snack", "region": "North India", "unit": "piece", "srv_g": 80.0,
        "cals": 380.0, "p": 7.5, "c": 42.0, "f": 20.0, "fib": 4.0,
        "gu_script": "ખસ્તા કચોરી", "gu_translit": "khasta kachori",
        "hi_script": "खस्ता कचौड़ी", "hi_translit": "khasta kachori",
        "aliases": ["moong dal kachori", "urad dal kachori", "halwai kachori"]
    },
    {
        "name": "Raj Kachori", "category": "street_food_chaat", "region": "North India", "unit": "plate", "srv_g": 200.0,
        "cals": 230.0, "p": 5.5, "c": 28.0, "f": 11.0, "fib": 3.2,
        "gu_script": "રાજ કચોરી", "gu_translit": "raj kachori",
        "hi_script": "राज कचौरी", "hi_translit": "raj kachori",
        "aliases": ["shahi raj kachori", "royal kachori chaat"]
    },
    {
        "name": "Aloo Tikki", "category": "street_food_chaat", "region": "North India", "unit": "piece", "srv_g": 80.0,
        "cals": 180.0, "p": 3.5, "c": 28.0, "f": 6.5, "fib": 2.8,
        "gu_script": "બટાકા ટીક્કી", "gu_translit": "bataka tikki",
        "hi_script": "आलू टिक्की", "hi_translit": "aloo tikki",
        "aliases": ["crispy aloo tikki", "delhi style aloo tikki", "aloo tikki chaat"]
    },
    {
        "name": "Paneer Tikki", "category": "snack", "region": "North India", "unit": "piece", "srv_g": 80.0,
        "cals": 220.0, "p": 10.5, "c": 18.0, "f": 12.0, "fib": 2.0,
        "gu_script": "પનીર ટીક્કી", "gu_translit": "paneer tikki",
        "hi_script": "पनीर टिक्की", "hi_translit": "paneer tikki",
        "aliases": ["cottage cheese cutlet", "paneer cutlet"]
    },
    {
        "name": "Bread Pakora", "category": "snack", "region": "North India", "unit": "piece", "srv_g": 100.0,
        "cals": 240.0, "p": 5.8, "c": 30.0, "f": 11.0, "fib": 2.5,
        "gu_script": "બ્રેડ પકોડા", "gu_translit": "bread pakora",
        "hi_script": "ब्रेड पकोड़ा", "hi_translit": "bread pakora",
        "aliases": ["aloo bread pakora", "stuffed bread pakora"]
    },
    {
        "name": "Mirchi Vada", "category": "snack", "region": "Rajasthan", "unit": "piece", "srv_g": 100.0,
        "cals": 225.0, "p": 4.8, "c": 26.0, "f": 11.5, "fib": 3.0,
        "gu_script": "મરચાં વડા", "gu_translit": "marcha vada",
        "hi_script": "मिर्ची वड़ा", "hi_translit": "mirchi vada",
        "aliases": ["jodhpuri mirchi bada", "mirchi pakora", "chilli fritter"]
    },
    {
        "name": "Dahi Bhalla", "category": "street_food_chaat", "region": "North India", "unit": "plate", "srv_g": 180.0,
        "cals": 160.0, "p": 5.5, "c": 20.0, "f": 6.5, "fib": 2.2,
        "gu_script": "દહીં ભલ્લા", "gu_translit": "dahi bhalla",
        "hi_script": "दही भल्ला", "hi_translit": "dahi bhalla",
        "aliases": ["dahi pakodi", "delhi dahi bhalla", "dahi vada chaat"]
    },
    {
        "name": "Ram Ladoo", "category": "street_food_chaat", "region": "North India", "unit": "plate", "srv_g": 150.0,
        "cals": 210.0, "p": 8.5, "c": 28.0, "f": 7.5, "fib": 4.5,
        "gu_script": "રામ લાડુ", "gu_translit": "ram ladoo",
        "hi_script": "राम लड्डू", "hi_translit": "ram ladoo",
        "aliases": ["moong dal pakodi chaat", "lajpat nagar ram ladoo"]
    },
    {
        "name": "Bikaneri Bhujia", "category": "snack", "region": "Rajasthan", "unit": "serving", "srv_g": 30.0,
        "cals": 560.0, "p": 13.5, "c": 42.0, "f": 38.0, "fib": 5.5,
        "gu_script": "બિકાનેરી ભુજિયા", "gu_translit": "bikaneri bhujia",
        "hi_script": "बीकानेरी भुजिया", "hi_translit": "bikaneri bhujia",
        "aliases": ["moth dal bhujia", "haldiram bikaneri bhujia"]
    },
    {
        "name": "Ratlami Sev", "category": "snack", "region": "Madhya Pradesh", "unit": "serving", "srv_g": 30.0,
        "cals": 540.0, "p": 13.0, "c": 42.0, "f": 36.0, "fib": 5.0,
        "gu_script": "રતલામી સેવ", "gu_translit": "ratlami sev",
        "hi_script": "रतलामी सेव", "hi_translit": "ratlami sev",
        "aliases": ["clove sev", "laung sev", "malwa ratlami sev"]
    },

    # ── 3. South Indian Snacks ───────────────────────────────────────────────
    {
        "name": "Medu Vada", "category": "snack", "region": "South India", "unit": "piece", "srv_g": 60.0,
        "cals": 265.0, "p": 8.5, "c": 26.5, "f": 14.5, "fib": 4.2,
        "gu_script": "મેદુ વડા", "gu_translit": "medu vada",
        "hi_script": "मेदु वड़ा", "hi_translit": "medu vada",
        "aliases": ["ulundu vadai", "crispy medu vada", "south indian vada"]
    },
    {
        "name": "Masala Vada", "category": "snack", "region": "South India", "unit": "piece", "srv_g": 50.0,
        "cals": 290.0, "p": 10.5, "c": 30.0, "f": 14.0, "fib": 6.5,
        "gu_script": "મસાલા વડા", "gu_translit": "masala vada",
        "hi_script": "मसाला वड़ा", "hi_translit": "masala vada",
        "aliases": ["paruppu vadai", "dal vada south style", "chana dal vadai"]
    },
    {
        "name": "Maddur Vada", "category": "snack", "region": "Karnataka", "unit": "piece", "srv_g": 70.0,
        "cals": 340.0, "p": 7.5, "c": 40.0, "f": 17.0, "fib": 4.0,
        "gu_script": "મદ્દુર વડા", "gu_translit": "maddur vada",
        "hi_script": "मद्दूर वड़ा", "hi_translit": "maddur vada",
        "aliases": ["karnataka maddur vade", "maddur crispy vada"]
    },
    {
        "name": "Mysore Bonda", "category": "snack", "region": "Karnataka", "unit": "piece", "srv_g": 60.0,
        "cals": 280.0, "p": 6.0, "c": 34.0, "f": 13.5, "fib": 2.5,
        "gu_script": "મૈસુર બોંડા", "gu_translit": "mysore bonda",
        "hi_script": "मैसूर बोंडा", "hi_translit": "mysore bonda",
        "aliases": ["mysore bajji", "maida bonda", "urad dal bonda"]
    },
    {
        "name": "Murukku", "category": "snack", "region": "South India", "unit": "piece", "srv_g": 30.0,
        "cals": 490.0, "p": 7.8, "c": 64.0, "f": 22.5, "fib": 3.8,
        "gu_script": "મુરુકુ", "gu_translit": "murukku",
        "hi_script": "मुरुक्कु", "hi_translit": "murukku",
        "aliases": ["chakli south style", "rice flour murukku", "crispy murukku"]
    },
    {
        "name": "Ribbon Pakoda", "category": "snack", "region": "South India", "unit": "serving", "srv_g": 40.0,
        "cals": 520.0, "p": 9.5, "c": 56.0, "f": 29.0, "fib": 4.5,
        "gu_script": "રીબન પકોડા", "gu_translit": "ribbon pakoda",
        "hi_script": "रिबन पकोड़ा", "hi_translit": "ribbon pakoda",
        "aliases": ["nada thenkuzhal", "ribbon murukku", "ola pakoda"]
    },
    {
        "name": "Banana Chips", "category": "snack", "region": "Kerala", "unit": "serving", "srv_g": 40.0,
        "cals": 519.0, "p": 2.3, "c": 58.4, "f": 33.6, "fib": 4.0,
        "gu_script": "કેળાં વેફર્સ", "gu_translit": "kela wafers",
        "hi_script": "केले के चिप्स", "hi_translit": "kele ke chips",
        "aliases": ["kerala banana chips", "nendran banana chips", "yellow banana chips"]
    },
    {
        "name": "Madras Mixture", "category": "snack", "region": "Tamil Nadu", "unit": "serving", "srv_g": 40.0,
        "cals": 535.0, "p": 11.0, "c": 50.0, "f": 32.0, "fib": 5.0,
        "gu_script": "મદ્રાસ મિક્સચર", "gu_translit": "madras mixture",
        "hi_script": "मद्रास मिक्सचर", "hi_translit": "madras mixture",
        "aliases": ["south indian mixture", "spicy mixture namkeen"]
    },

    # ── 4. Maharashtrian / West Indian Snacks ────────────────────────────────
    {
        "name": "Vada Pav", "category": "street_food_chaat", "region": "Maharashtra", "unit": "piece", "srv_g": 120.0,
        "cals": 245.0, "p": 5.5, "c": 36.0, "f": 9.0, "fib": 3.2,
        "gu_script": "વડા પાઉં", "gu_translit": "vada pav",
        "hi_script": "वड़ा पाव", "hi_translit": "vada pav",
        "aliases": ["mumbai vada pav", "batata vada pav", "wada pav"]
    },
    {
        "name": "Misal Pav", "category": "composite_meal", "region": "Maharashtra", "unit": "plate", "srv_g": 250.0,
        "cals": 165.0, "p": 6.8, "c": 22.0, "f": 5.8, "fib": 4.2,
        "gu_script": "મિસળ પાઉં", "gu_translit": "misal pav",
        "hi_script": "मिसल पाव", "hi_translit": "misal pav",
        "aliases": ["puneri misal", "kolhapuri misal", "spicy misal pav"]
    },
    {
        "name": "Thalipeeth", "category": "bread_flatbread", "region": "Maharashtra", "unit": "piece", "srv_g": 100.0,
        "cals": 215.0, "p": 6.5, "c": 32.0, "f": 7.2, "fib": 4.8,
        "gu_script": "થાલીપીઠ", "gu_translit": "thalipeeth",
        "hi_script": "थालीपीठ", "hi_translit": "thalipeeth",
        "aliases": ["bhajani thalipeeth", "multigrain thalipeeth"]
    },
    {
        "name": "Sabudana Vada", "category": "snack", "region": "Maharashtra", "unit": "piece", "srv_g": 60.0,
        "cals": 310.0, "p": 3.5, "c": 44.0, "f": 13.5, "fib": 2.0,
        "gu_script": "સાબુદાણા વડા", "gu_translit": "sabudana vada",
        "hi_script": "साबूदाना वड़ा", "hi_translit": "sabudana vada",
        "aliases": ["farali sabudana vada", "sago patty", "crispy sabudana vada"]
    },
    {
        "name": "Kothimbir Vadi", "category": "snack", "region": "Maharashtra", "unit": "piece", "srv_g": 40.0,
        "cals": 195.0, "p": 7.2, "c": 24.0, "f": 8.0, "fib": 4.0,
        "gu_script": "કોથમીર વડી", "gu_translit": "kothimbir vadi",
        "hi_script": "कोथिंबीर वड़ी", "hi_translit": "kothimbir vadi",
        "aliases": ["coriander fritters", "maharashtrian kothimbir vadi"]
    },
    {
        "name": "Bhakarwadi", "category": "snack", "region": "Maharashtra", "unit": "piece", "srv_g": 25.0,
        "cals": 480.0, "p": 8.5, "c": 54.0, "f": 26.0, "fib": 4.5,
        "gu_script": "ભાખરવડી", "gu_translit": "bhakarwadi",
        "hi_script": "भाकरवड़ी", "hi_translit": "bhakarwadi",
        "aliases": ["chitale bhakarwadi", "pune bhakarwadi", "spicy roll snack"]
    },
    {
        "name": "Poha Chivda", "category": "snack", "region": "Maharashtra", "unit": "serving", "srv_g": 40.0,
        "cals": 458.0, "p": 6.8, "c": 64.2, "f": 19.5, "fib": 3.5,
        "gu_script": "પૌંઆ ચેવડો", "gu_translit": "pauva chevdo",
        "hi_script": "पोहा चिवड़ा", "hi_translit": "poha chivda",
        "aliases": ["roasted poha chivda", "diet poha chivda", "maharashtrian chivda"]
    },

    # ── 5. East Indian Snacks ────────────────────────────────────────────────
    {
        "name": "Singara", "category": "snack", "region": "Bengal", "unit": "piece", "srv_g": 80.0,
        "cals": 255.0, "p": 4.2, "c": 33.0, "f": 12.0, "fib": 2.2,
        "gu_script": "સિંગારા", "gu_translit": "singara",
        "hi_script": "सिंगाड़ा समोसा", "hi_translit": "singara",
        "aliases": ["bengali samosa", "kolkata singara", "phulkopir singara"]
    },
    {
        "name": "Jhalmuri", "category": "street_food_chaat", "region": "Bengal", "unit": "plate", "srv_g": 100.0,
        "cals": 180.0, "p": 4.5, "c": 32.0, "f": 4.2, "fib": 3.0,
        "gu_script": "ઝાલમૂરી", "gu_translit": "jhalmuri",
        "hi_script": "झालमुड़ी", "hi_translit": "jhalmuri",
        "aliases": ["kolkata jhalmuri", "spicy puffed rice", "bengali bhel"]
    },
    {
        "name": "Ghugni Chaat", "category": "street_food_chaat", "region": "Bengal", "unit": "plate", "srv_g": 180.0,
        "cals": 135.0, "p": 6.8, "c": 21.0, "f": 2.8, "fib": 4.8,
        "gu_script": "ઘુગની ચાટ", "gu_translit": "ghugni chaat",
        "hi_script": "घुघनी चाट", "hi_translit": "ghugni chaat",
        "aliases": ["yellow peas chaat", "matar ghugni", "bengali ghugni"]
    },
    {
        "name": "Beguni", "category": "snack", "region": "Bengal", "unit": "piece", "srv_g": 60.0,
        "cals": 220.0, "p": 4.0, "c": 24.0, "f": 12.0, "fib": 3.5,
        "gu_script": "બેગુની", "gu_translit": "beguni",
        "hi_script": "बैगुनी", "hi_translit": "beguni",
        "aliases": ["eggplant fritter", "baingan pakora bengali"]
    },
    {
        "name": "Litti Chokha", "category": "composite_meal", "region": "Bihar", "unit": "plate", "srv_g": 250.0,
        "cals": 210.0, "p": 8.5, "c": 36.0, "f": 5.0, "fib": 5.2,
        "gu_script": "લિટી ચોખા", "gu_translit": "litti chokha",
        "hi_script": "लिट्टी चोखा", "hi_translit": "litti chokha",
        "aliases": ["bihari litti", "sattu litti with baingan chokha"]
    },

    # ── 6. Healthy, Roasted & Diet Snacks ─────────────────────────────────────
    {
        "name": "Roasted Makhana", "category": "snack", "region": "Pan-India", "unit": "bowl", "srv_g": 40.0,
        "cals": 360.0, "p": 9.5, "c": 74.0, "f": 2.5, "fib": 7.5,
        "gu_script": "શેકેલા મખાના", "gu_translit": "shekela makhana",
        "hi_script": "भुना मखाना", "hi_translit": "bhuna makhana",
        "aliases": ["foxnut snack", "phool makhana", "salted makhana", "diet makhana", "peri peri makhana"]
    },
    {
        "name": "Roasted Chana", "category": "snack", "region": "Pan-India", "unit": "bowl", "srv_g": 50.0,
        "cals": 370.0, "p": 21.5, "c": 58.0, "f": 5.5, "fib": 14.5,
        "gu_script": "શેકેલા ચણા", "gu_translit": "shekela chana",
        "hi_script": "भुना चना", "hi_translit": "bhuna chana",
        "aliases": ["bhuna chana", "roasted chickpeas", "diet chana", "haldi chana"]
    },
    {
        "name": "Diet Chivda", "category": "snack", "region": "Pan-India", "unit": "bowl", "srv_g": 40.0,
        "cals": 390.0, "p": 8.5, "c": 72.0, "f": 7.5, "fib": 6.0,
        "gu_script": "ડાયેટ ચેવડો", "gu_translit": "diet chevdo",
        "hi_script": "डाइट चिवड़ा", "hi_translit": "diet chivda",
        "aliases": ["roasted chivda", "low calorie chivda", "diet mixture"]
    },
    {
        "name": "Sprouted Chaat", "category": "snack", "region": "Pan-India", "unit": "bowl", "srv_g": 150.0,
        "cals": 110.0, "p": 7.5, "c": 18.0, "f": 1.2, "fib": 5.5,
        "gu_script": "ફણગાવેલા કઠોળ ચાટ", "gu_translit": "fangavela kathol chaat",
        "hi_script": "अंकुरित चाट", "hi_translit": "ankurit chaat",
        "aliases": ["moong sprout chaat", "sprouts salad", "diet sprout chaat"]
    },
    {
        "name": "Singdana", "category": "snack", "region": "Pan-India", "unit": "serving", "srv_g": 40.0,
        "cals": 570.0, "p": 25.5, "c": 17.5, "f": 46.0, "fib": 8.0,
        "gu_script": "સીંગદાણા", "gu_translit": "singdana",
        "hi_script": "मूंगफली", "hi_translit": "mungfali",
        "aliases": ["roasted peanuts", "salted peanuts", "khari sing", "masala sing"]
    },

    # ── 7. Street Food & Packaged Items ──────────────────────────────────────
    {
        "name": "Pani Puri", "category": "street_food_chaat", "region": "Pan-India", "unit": "plate", "srv_g": 150.0,
        "cals": 155.0, "p": 3.0, "c": 26.0, "f": 4.5, "fib": 2.5,
        "gu_script": "પાણી પૂરી", "gu_translit": "pani puri",
        "hi_script": "पानी पूरी", "hi_translit": "pani puri",
        "aliases": ["golgappe", "golgappa", "puchka", "phuchka", "batasha", "paani poori"]
    },
    {
        "name": "Bhel Puri", "category": "street_food_chaat", "region": "Pan-India", "unit": "plate", "srv_g": 150.0,
        "cals": 175.0, "p": 4.2, "c": 30.0, "f": 4.5, "fib": 3.0,
        "gu_script": "ભેળ પૂરી", "gu_translit": "bhel puri",
        "hi_script": "भेल पूरी", "hi_translit": "bhel puri",
        "aliases": ["mumbai bhel", "sukha bhel", "geela bhel", "spicy bhel"]
    },
    {
        "name": "Dabeli", "category": "street_food_chaat", "region": "Gujarat", "unit": "piece", "srv_g": 120.0,
        "cals": 215.0, "p": 4.8, "c": 34.0, "f": 6.8, "fib": 3.2,
        "gu_script": "દાબેલી", "gu_translit": "dabeli",
        "hi_script": "दाबेली", "hi_translit": "dabeli",
        "aliases": ["kutchi dabeli", "double roti", "spicy dabeli"]
    },
    {
        "name": "Pav Bhaji", "category": "composite_meal", "region": "Maharashtra", "unit": "plate", "srv_g": 250.0,
        "cals": 145.0, "p": 3.8, "c": 21.0, "f": 5.5, "fib": 3.2,
        "gu_script": "પાઉં ભાજી", "gu_translit": "pav bhaji",
        "hi_script": "पाव भाजी", "hi_translit": "pav bhaji",
        "aliases": ["butter pav bhaji", "mumbai pav bhaji", "cheese pav bhaji"]
    },
    {
        "name": "Kurkure", "category": "packaged_food", "region": "Pan-India", "unit": "packet", "srv_g": 70.0,
        "cals": 550.0, "p": 6.0, "c": 56.0, "f": 34.0, "fib": 2.5,
        "gu_script": "કુરકુરે", "gu_translit": "kurkure",
        "hi_script": "कुरकुरे", "hi_translit": "kurkure",
        "aliases": ["kurkure masala munch", "kurkure green chutney", "kurkure chilli chatka"]
    },
    {
        "name": "Lays Magic Masala", "category": "packaged_food", "region": "Pan-India", "unit": "packet", "srv_g": 50.0,
        "cals": 540.0, "p": 7.0, "c": 52.0, "f": 34.0, "fib": 3.5,
        "gu_script": "લેઝ મેજિક મસાલા", "gu_translit": "lays magic masala",
        "hi_script": "लेज़ मैजिक मसाला", "hi_translit": "lays magic masala",
        "aliases": ["lays chips", "potato chips masala", "lays blue packet", "lays wafers"]
    },
    {
        "name": "Parle-G", "category": "packaged_food", "region": "Pan-India", "unit": "piece", "srv_g": 5.0,
        "cals": 450.0, "p": 7.5, "c": 78.0, "f": 12.5, "fib": 2.0,
        "gu_script": "પાર્લે જી", "gu_translit": "parle g",
        "hi_script": "पारले-जी", "hi_translit": "parle g",
        "aliases": ["parle g biscuit", "glucose biscuit", "chai biscuit"]
    },
    {
        "name": "Monaco", "category": "packaged_food", "region": "Pan-India", "unit": "piece", "srv_g": 4.0,
        "cals": 480.0, "p": 8.0, "c": 65.0, "f": 21.0, "fib": 2.5,
        "gu_script": "મોનાકો", "gu_translit": "monaco",
        "hi_script": "मोनाको", "hi_translit": "monaco",
        "aliases": ["monaco biscuit", "salted crackers", "monaco namkeen biscuit"]
    },
    {
        "name": "Khari Biscuit", "category": "snack", "region": "Pan-India", "unit": "piece", "srv_g": 15.0,
        "cals": 520.0, "p": 7.0, "c": 58.0, "f": 29.0, "fib": 2.2,
        "gu_script": "ખારી બિસ્કીટ", "gu_translit": "khari biscuit",
        "hi_script": "खारी बिस्कुट", "hi_translit": "khari biscuit",
        "aliases": ["puff pastry biscuit", "maska khari", "methi khari", "chai khari"]
    },
]


def ingest_authentic_snacks():
    settings = get_settings()
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
    except Exception:
        client = MongoClient(settings.MONGODB_URL)

    db = client[settings.MONGODB_DB_NAME]
    foods_coll = db["foods"]
    aliases_coll = db["food_aliases"]

    print(f"Connecting to MongoDB: {db.name}")
    initial_foods = foods_coll.count_documents({})
    initial_aliases = aliases_coll.count_documents({})
    print(f"Initial: {initial_foods} foods, {initial_aliases} aliases")

    inserted_foods = 0
    updated_aliases = 0
    now_iso = datetime.datetime.now().isoformat()

    for item in AUTHENTIC_SNACKS_CATALOG:
        canonical = item["name"]
        food_name_clean = canonical.strip().lower()
        category = item["category"]
        region = item["region"]
        p_unit = item["unit"]
        srv_g = item["srv_g"]
        cals_100g = item["cals"]
        cals_serving = round(cals_100g * srv_g / 100.0, 1)

        existing_food = foods_coll.find_one({"food_name": food_name_clean})
        if existing_food:
            food_id = existing_food["food_id"]
        else:
            food_id = f"food_snk_{re.sub(r'[^\w]', '_', food_name_clean)}"
            schema = generate_food_schema_fields(canonical, category, "", p_unit)

            food_doc = {
                "food_id": food_id,
                "food_name": food_name_clean,
                "food_name_display": canonical,
                "aliases": [],
                "category": category,
                "subcategory": "authentic_farsan",
                "region": region,
                "state": region,
                "vegetarian_status": "Veg",
                "serving_size_g": srv_g,
                "serving_unit": p_unit,
                "calories_per_100g": cals_100g,
                "calories_kcal": cals_serving,
                "protein_g": item["p"],
                "carbs_g": item["c"],
                "fat_g": item["f"],
                "fiber_g": item["fib"],
                "sugar_g": 0.0,
                "sodium_mg": 0.0,
                "quantity_units": schema.get("quantity_units", []),
                "quantity_options": schema.get("quantity_options", []),
                "preparation_variants": schema.get("preparation_variants", []),
                "variant_nutrition": schema.get("variant_nutrition", {}),
                "is_verified": True,
                "data_source": "Authentic Indian Snack Master",
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            foods_coll.insert_one(food_doc)
            inserted_foods += 1

        # Build alias set
        aliases_to_add = {
            (canonical.lower(), "primary", "en"),
            (food_name_clean, "primary", "en"),
            (item["gu_script"].strip().lower(), "regional_script", "gu"),
            (item["gu_translit"].strip().lower(), "transliteration", "gu"),
            (item["hi_script"].strip().lower(), "regional_script", "hi"),
            (item["hi_translit"].strip().lower(), "transliteration", "hi")
        }
        for al in item.get("aliases", []):
            al_clean = al.strip().lower()
            if al_clean:
                aliases_to_add.add((al_clean, "colloquial_alias", "en"))

        for a_text, a_type, a_lang in aliases_to_add:
            if a_text:
                res = aliases_coll.update_one(
                    {"alias": a_text},
                    {"$set": {"food_id": food_id, "alias": a_text, "alias_type": a_type, "language": a_lang}},
                    upsert=True
                )
                if res.upserted_id or res.modified_count:
                    updated_aliases += 1

    final_foods = foods_coll.count_documents({})
    final_aliases = aliases_coll.count_documents({})
    print(f"[SNACK INGESTION COMPLETE]")
    print(f"  Foods: {initial_foods} -> {final_foods} (+{inserted_foods} new snacks)")
    print(f"  Aliases: {initial_aliases} -> {final_aliases} (+{final_aliases - initial_aliases} new aliases)")

    # Sync to Clean dataset CSVs
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
    ingest_authentic_snacks()
