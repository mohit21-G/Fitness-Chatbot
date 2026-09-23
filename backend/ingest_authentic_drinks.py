"""
Ingest Comprehensive Authentic Indian & Global Drink Catalog into MongoDB.
Covers regular & diet/zero soft drinks, energy drinks, fresh & packaged juices,
Indian traditional drinks (chaas, lassi, nimbu pani, jaljeera, sharbat, coconut water),
teas, coffees, protein & sports drinks, and mocktails/sparkling water.
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

AUTHENTIC_DRINKS_CATALOG = [
    # ── 1. Regular Soft Drinks ────────────────────────────────────────────────
    {
        "name": "Coca Cola", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 42.0, "p": 0.0, "c": 10.6, "f": 0.0, "fib": 0.0,
        "gu_script": "કોકા કોલા", "gu_translit": "coca cola",
        "hi_script": "कोका कोला", "hi_translit": "coca cola",
        "aliases": ["coke", "coca-cola", "cola", "regular coke", "classic coke", "coke can", "coke bottle", "coca cola can", "coca cola bottle"]
    },
    {
        "name": "Pepsi", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 43.0, "p": 0.0, "c": 10.8, "f": 0.0, "fib": 0.0,
        "gu_script": "પેપ્સી", "gu_translit": "pepsi",
        "hi_script": "पेप्सी", "hi_translit": "pepsi",
        "aliases": ["regular pepsi", "pepsi cola", "pepsi can", "pepsi bottle"]
    },
    {
        "name": "Sprite", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 40.0, "p": 0.0, "c": 10.0, "f": 0.0, "fib": 0.0,
        "gu_script": "સ્પ્રાઈટ", "gu_translit": "sprite",
        "hi_script": "स्प्राइट", "hi_translit": "sprite",
        "aliases": ["regular sprite", "sprite cold drink", "sprite can", "sprite bottle"]
    },
    {
        "name": "Thums Up", "category": "drink", "region": "India", "unit": "bottle", "srv_g": 250.0,
        "cals": 40.0, "p": 0.0, "c": 10.0, "f": 0.0, "fib": 0.0,
        "gu_script": "થમ્સ અપ", "gu_translit": "thums up",
        "hi_script": "थम्स अप", "hi_translit": "thums up",
        "aliases": ["thumbs up", "thumps up", "toofan cold drink", "thums up can", "thums up bottle"]
    },
    {
        "name": "Fanta", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 48.0, "p": 0.0, "c": 12.0, "f": 0.0, "fib": 0.0,
        "gu_script": "ફેન્ટા", "gu_translit": "fanta",
        "hi_script": "फैंटा", "hi_translit": "fanta",
        "aliases": ["fanta orange", "fanta can", "fanta bottle", "orange cold drink"]
    },
    {
        "name": "Mirinda", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 52.0, "p": 0.0, "c": 13.0, "f": 0.0, "fib": 0.0,
        "gu_script": "મિરિન્ડા", "gu_translit": "mirinda",
        "hi_script": "मिरिंडा", "hi_translit": "mirinda",
        "aliases": ["mirinda orange", "mirinda can", "mirinda bottle"]
    },
    {
        "name": "Mountain Dew", "category": "drink", "region": "Global", "unit": "bottle", "srv_g": 250.0,
        "cals": 48.0, "p": 0.0, "c": 12.0, "f": 0.0, "fib": 0.0,
        "gu_script": "માઉન્ટેન ડ્યુ", "gu_translit": "mountain dew",
        "hi_script": "माउंटेन ड्यू", "hi_translit": "mountain dew",
        "aliases": ["dew", "mountain dew can", "mountain dew bottle"]
    },
    {
        "name": "7Up", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 41.0, "p": 0.0, "c": 10.2, "f": 0.0, "fib": 0.0,
        "gu_script": "સેવન અપ", "gu_translit": "seven up",
        "hi_script": "सेवन अप", "hi_translit": "seven up",
        "aliases": ["seven up", "7 up", "7up can", "7up bottle"]
    },
    {
        "name": "Limca", "category": "drink", "region": "India", "unit": "bottle", "srv_g": 250.0,
        "cals": 36.0, "p": 0.0, "c": 9.0, "f": 0.0, "fib": 0.0,
        "gu_script": "લીમ્કા", "gu_translit": "limca",
        "hi_script": "लिम्का", "hi_translit": "limca",
        "aliases": ["limca cold drink", "limca bottle", "limca can", "lemon limca"]
    },

    # ── 2. Diet & Zero Soft Drinks (CRITICAL: ZERO / LOW CALORIE) ──────────────
    {
        "name": "Diet Coke", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 0.3, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "ડાયેટ કોક", "gu_translit": "diet coke",
        "hi_script": "डाइट कोक", "hi_translit": "diet coke",
        "aliases": ["diet coca cola", "diet coke can", "coke diet", "diet coca-cola", "diet coke bottle"]
    },
    {
        "name": "Coke Zero", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 0.3, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "કોક ઝીરો", "gu_translit": "coke zero",
        "hi_script": "कोक ज़ीरो", "hi_translit": "coke zero",
        "aliases": ["coca cola zero", "coke zero sugar", "zero coke", "coke zero can", "coca-cola zero sugar", "coke zero bottle"]
    },
    {
        "name": "Pepsi Black", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 0.3, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "પેપ્સી બ્લેક", "gu_translit": "pepsi black",
        "hi_script": "पेप्सी ब्लैक", "hi_translit": "pepsi black",
        "aliases": ["pepsi zero sugar", "diet pepsi", "pepsi black can", "pepsi zero", "pepsi black bottle"]
    },
    {
        "name": "Sprite Zero", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 0.3, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "સ્પ્રાઈટ ઝીરો", "gu_translit": "sprite zero",
        "hi_script": "स्प्राइट ज़ीरो", "hi_translit": "sprite zero",
        "aliases": ["sprite zero sugar", "diet sprite", "sprite zero can", "sprite zero bottle"]
    },
    {
        "name": "Thums Up Zero", "category": "drink", "region": "India", "unit": "can", "srv_g": 300.0,
        "cals": 0.3, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "થમ્સ અપ ઝીરો", "gu_translit": "thums up zero",
        "hi_script": "थम्स अप ज़ीरो", "hi_translit": "thums up zero",
        "aliases": ["thums up charged zero", "thumbs up zero", "diet thums up", "thums up zero sugar"]
    },

    # ── 3. Energy Drinks ──────────────────────────────────────────────────────
    {
        "name": "Red Bull", "category": "drink", "region": "Global", "unit": "can", "srv_g": 250.0,
        "cals": 45.0, "p": 0.0, "c": 11.0, "f": 0.0, "fib": 0.0,
        "gu_script": "રેડ બુલ", "gu_translit": "red bull",
        "hi_script": "रेड बुल", "hi_translit": "red bull",
        "aliases": ["redbull", "regular red bull", "red bull energy drink", "red bull can"]
    },
    {
        "name": "Red Bull Sugarfree", "category": "drink", "region": "Global", "unit": "can", "srv_g": 250.0,
        "cals": 3.0, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "રેડ બુલ સુગરફ્રી", "gu_translit": "red bull sugarfree",
        "hi_script": "रेड बुल शुगरफ्री", "hi_translit": "red bull sugarfree",
        "aliases": ["red bull zero", "sugar free red bull", "red bull sugar free", "diet red bull", "redbull sugarfree", "redbull zero"]
    },
    {
        "name": "Monster Energy", "category": "drink", "region": "Global", "unit": "can", "srv_g": 500.0,
        "cals": 47.0, "p": 0.0, "c": 12.0, "f": 0.0, "fib": 0.0,
        "gu_script": "મોન્સ્ટર એનર્જી", "gu_translit": "monster energy",
        "hi_script": "मॉनस्टर एनर्जी", "hi_translit": "monster energy",
        "aliases": ["monster", "regular monster", "monster energy drink", "monster can", "green monster"]
    },
    {
        "name": "Monster Ultra", "category": "drink", "region": "Global", "unit": "can", "srv_g": 500.0,
        "cals": 2.0, "p": 0.0, "c": 0.5, "f": 0.0, "fib": 0.0,
        "gu_script": "મોન્સ્ટર અલ્ટ્રા", "gu_translit": "monster ultra",
        "hi_script": "मॉनस्टर अल्ट्रा", "hi_translit": "monster ultra",
        "aliases": ["monster zero ultra", "monster ultra white", "monster zero sugar", "monster energy zero", "diet monster", "white monster"]
    },
    {
        "name": "Sting Energy Drink", "category": "drink", "region": "India", "unit": "bottle", "srv_g": 250.0,
        "cals": 28.0, "p": 0.0, "c": 7.0, "f": 0.0, "fib": 0.0,
        "gu_script": "સ્ટિંગ", "gu_translit": "sting",
        "hi_script": "स्टिंग एनर्जी", "hi_translit": "sting energy",
        "aliases": ["sting", "sting energy", "sting drink", "sting red", "sting bottle", "sting can"]
    },
    {
        "name": "Hell Energy Drink", "category": "drink", "region": "Global", "unit": "can", "srv_g": 250.0,
        "cals": 46.0, "p": 0.0, "c": 11.0, "f": 0.0, "fib": 0.0,
        "gu_script": "હેલ એનર્જી", "gu_translit": "hell energy",
        "hi_script": "हेल एनर्जी", "hi_translit": "hell energy",
        "aliases": ["hell energy", "hell drink", "hell can", "hell energy can"]
    },

    # ── 4. Fresh & Packaged Juices ────────────────────────────────────────────
    {
        "name": "Sugarcane Juice", "category": "drink", "region": "India", "unit": "glass", "srv_g": 250.0,
        "cals": 80.0, "p": 0.2, "c": 20.0, "f": 0.1, "fib": 0.5,
        "gu_script": "શેરડીનો રસ", "gu_translit": "sherdi no ras",
        "hi_script": "गन्ने का रस", "hi_translit": "ganne ka ras",
        "aliases": ["ganne ka ras", "sherdi no ras", "fresh sugarcane juice", "ganne ka juice", "sherdi ras"]
    },
    {
        "name": "Orange Juice", "category": "drink", "region": "Global", "unit": "glass", "srv_g": 250.0,
        "cals": 45.0, "p": 0.7, "c": 10.4, "f": 0.2, "fib": 0.4,
        "gu_script": "સંતરાનો રસ", "gu_translit": "santrano ras",
        "hi_script": "संतरे का जूस", "hi_translit": "santre ka juice",
        "aliases": ["fresh orange juice", "santre ka juice", "santre no ras", "mosambi juice", "sweet lime juice", "orange juice glass"]
    },
    {
        "name": "Apple Juice", "category": "drink", "region": "Global", "unit": "glass", "srv_g": 250.0,
        "cals": 46.0, "p": 0.1, "c": 11.3, "f": 0.1, "fib": 0.2,
        "gu_script": "સફરજનનો રસ", "gu_translit": "safarjanno ras",
        "hi_script": "सेब का जूस", "hi_translit": "seb ka juice",
        "aliases": ["fresh apple juice", "seb ka juice", "apple drink", "apple juice glass"]
    },
    {
        "name": "Real Fruit Juice", "category": "drink", "region": "India", "unit": "glass", "srv_g": 200.0,
        "cals": 54.0, "p": 0.4, "c": 13.0, "f": 0.1, "fib": 0.3,
        "gu_script": "રીયલ ફ્રૂટ જ્યુસ", "gu_translit": "real fruit juice",
        "hi_script": "रियल फ्रूट जूस", "hi_translit": "real fruit juice",
        "aliases": ["real mixed fruit juice", "packaged fruit juice", "tropicana juice", "real juice", "tropicana mixed fruit"]
    },
    {
        "name": "Frooti", "category": "drink", "region": "India", "unit": "bottle", "srv_g": 250.0,
        "cals": 65.0, "p": 0.0, "c": 16.2, "f": 0.0, "fib": 0.1,
        "gu_script": "ફ્રૂટી", "gu_translit": "frooti",
        "hi_script": "फ्रूटी", "hi_translit": "frooti",
        "aliases": ["frooti mango drink", "frooti bottle", "frooti tetrapak", "frooti drink"]
    },
    {
        "name": "Maaza", "category": "drink", "region": "India", "unit": "bottle", "srv_g": 250.0,
        "cals": 62.0, "p": 0.1, "c": 15.5, "f": 0.0, "fib": 0.1,
        "gu_script": "માઝા", "gu_translit": "maaza",
        "hi_script": "माज़ा", "hi_translit": "maaza",
        "aliases": ["maaza mango drink", "maaza bottle", "maaza juice", "maaza tetrapak"]
    },
    {
        "name": "Slice", "category": "drink", "region": "India", "unit": "bottle", "srv_g": 250.0,
        "cals": 64.0, "p": 0.1, "c": 16.0, "f": 0.0, "fib": 0.1,
        "gu_script": "સ્લાઇસ", "gu_translit": "slice",
        "hi_script": "स्लाइस", "hi_translit": "slice",
        "aliases": ["slice mango drink", "slice bottle", "slice juice"]
    },
    {
        "name": "Mango Drink", "category": "drink", "region": "India", "unit": "bottle", "srv_g": 250.0,
        "cals": 60.0, "p": 0.2, "c": 15.0, "f": 0.1, "fib": 0.2,
        "gu_script": "મેંગો ડ્રિન્ક", "gu_translit": "mango drink",
        "hi_script": "मैंगो ड्रिंक", "hi_translit": "mango drink",
        "aliases": ["mango juice drink", "packaged mango juice", "mango pulp drink"]
    },

    # ── 5. Indian Traditional Drinks ──────────────────────────────────────────
    {
        "name": "Masala Chaas", "category": "drink", "region": "Gujarat", "unit": "glass", "srv_g": 250.0,
        "cals": 22.0, "p": 1.2, "c": 2.0, "f": 1.0, "fib": 0.1,
        "gu_script": "મસાલા છાશ", "gu_translit": "masala chaas",
        "hi_script": "मसाला छाछ", "hi_translit": "masala chaach",
        "aliases": ["chaas", "chhas", "buttermilk", "masala buttermilk", "spiced buttermilk", "salted chaas", "gujarati chaas", "jeera chaas"]
    },
    {
        "name": "Sweet Lassi", "category": "drink", "region": "Punjab", "unit": "glass", "srv_g": 250.0,
        "cals": 85.0, "p": 3.0, "c": 14.0, "f": 2.5, "fib": 0.0,
        "gu_script": "મીઠી લસ્સી", "gu_translit": "meethi lassi",
        "hi_script": "मीठी लस्सी", "hi_translit": "meethi lassi",
        "aliases": ["lassi", "punjabi lassi", "sweet curd drink", "meethi lassi", "plain lassi", "creamy sweet lassi"]
    },
    {
        "name": "Mango Lassi", "category": "drink", "region": "Punjab", "unit": "glass", "srv_g": 250.0,
        "cals": 95.0, "p": 2.8, "c": 16.5, "f": 2.2, "fib": 0.5,
        "gu_script": "મેંગો લસ્સી", "gu_translit": "mango lassi",
        "hi_script": "मैंगो लस्सी", "hi_translit": "mango lassi",
        "aliases": ["aam lassi", "mango curd drink", "keri lassi", "fresh mango lassi"]
    },
    {
        "name": "Salted Lassi", "category": "drink", "region": "North India", "unit": "glass", "srv_g": 250.0,
        "cals": 35.0, "p": 2.5, "c": 3.2, "f": 1.5, "fib": 0.0,
        "gu_script": "નમકીન લસ્સી", "gu_translit": "namkeen lassi",
        "hi_script": "नमकीन लस्सी", "hi_translit": "namkeen lassi",
        "aliases": ["khari lassi", "salted curd drink", "jeera lassi"]
    },
    {
        "name": "Nimbu Pani", "category": "drink", "region": "India", "unit": "glass", "srv_g": 250.0,
        "cals": 28.0, "p": 0.2, "c": 7.0, "f": 0.0, "fib": 0.1,
        "gu_script": "લીંબુ પાણી", "gu_translit": "limbu pani",
        "hi_script": "नींबू पानी", "hi_translit": "nimbu pani",
        "aliases": ["shikanji", "lemonade", "fresh lime water", "nimbu shikanji", "limbu pani", "fresh nimbu pani", "sweet nimbu pani"]
    },
    {
        "name": "Jaljeera", "category": "drink", "region": "North India", "unit": "glass", "srv_g": 250.0,
        "cals": 18.0, "p": 0.4, "c": 4.0, "f": 0.1, "fib": 0.2,
        "gu_script": "જીરા પાણી", "gu_translit": "jeera pani",
        "hi_script": "जलजीरा", "hi_translit": "jaljeera",
        "aliases": ["jal jeera", "jaljira", "cumin drink", "spiced cumin water", "jaljeera drink"]
    },
    {
        "name": "Rooh Afza", "category": "drink", "region": "India", "unit": "glass", "srv_g": 250.0,
        "cals": 40.0, "p": 0.1, "c": 10.0, "f": 0.0, "fib": 0.0,
        "gu_script": "રૂહ અફઝા", "gu_translit": "rooh afza",
        "hi_script": "रूह अफ़ज़ा", "hi_translit": "rooh afza",
        "aliases": ["roohafza", "gulab sharbat", "rose sharbat", "rooh afza sharbat", "rooh afza milk"]
    },
    {
        "name": "Kokum Sharbat", "category": "drink", "region": "Maharashtra", "unit": "glass", "srv_g": 250.0,
        "cals": 35.0, "p": 0.2, "c": 8.5, "f": 0.0, "fib": 0.2,
        "gu_script": "કોકમ શરબત", "gu_translit": "kokum sharbat",
        "hi_script": "कोकम शरबत", "hi_translit": "kokum sharbat",
        "aliases": ["kokum juice", "kokum drink", "maharashtrian kokum sharbat", "sweet kokum juice"]
    },
    {
        "name": "Khus Sharbat", "category": "drink", "region": "North India", "unit": "glass", "srv_g": 250.0,
        "cals": 38.0, "p": 0.1, "c": 9.5, "f": 0.0, "fib": 0.0,
        "gu_script": "ખસ શરબત", "gu_translit": "khus sharbat",
        "hi_script": "खस का शरबत", "hi_translit": "khus sharbat",
        "aliases": ["khus drink", "vetiver drink", "green sharbat", "khus syrup drink"]
    },
    {
        "name": "Coconut Water", "category": "drink", "region": "South India", "unit": "glass", "srv_g": 250.0,
        "cals": 19.0, "p": 0.7, "c": 3.7, "f": 0.2, "fib": 1.1,
        "gu_script": "લીલું નાળિયેર પાણી", "gu_translit": "lilu nariyal pani",
        "hi_script": "नारियल पानी", "hi_translit": "nariyal pani",
        "aliases": ["nariyal pani", "tender coconut water", "elaneer", "fresh coconut water", "nariyel pani"]
    },
    {
        "name": "Sattu Sharbat", "category": "drink", "region": "Bihar", "unit": "glass", "srv_g": 250.0,
        "cals": 55.0, "p": 3.5, "c": 8.5, "f": 1.0, "fib": 1.5,
        "gu_script": "સત્તુ શરબત", "gu_translit": "sattu sharbat",
        "hi_script": "सत्तू का शरबत", "hi_translit": "sattu sharbat",
        "aliases": ["sattu drink", "bihari sattu sharbat", "sattu namkeen drink", "roasted gram flour drink"]
    },
    {
        "name": "Aam Panna", "category": "drink", "region": "North India", "unit": "glass", "srv_g": 250.0,
        "cals": 45.0, "p": 0.4, "c": 11.0, "f": 0.1, "fib": 0.4,
        "gu_script": "કેરી પન્ના", "gu_translit": "keri panna",
        "hi_script": "आम पन्ना", "hi_translit": "aam panna",
        "aliases": ["kairi panna", "raw mango drink", "aam ka panna", "kacchi keri panna"]
    },
    {
        "name": "Solkadhi", "category": "drink", "region": "Maharashtra", "unit": "glass", "srv_g": 200.0,
        "cals": 48.0, "p": 1.0, "c": 3.5, "f": 3.5, "fib": 0.3,
        "gu_script": "સોલકઢી", "gu_translit": "solkadhi",
        "hi_script": "सोलकढ़ी", "hi_translit": "solkadhi",
        "aliases": ["sol kadhi", "kokum coconut milk drink", "solkadi", "konkani solkadhi"]
    },
    {
        "name": "Thandai", "category": "drink", "region": "North India", "unit": "glass", "srv_g": 250.0,
        "cals": 95.0, "p": 3.8, "c": 12.5, "f": 3.6, "fib": 0.8,
        "gu_script": "ઠંડાઈ", "gu_translit": "thandai",
        "hi_script": "ठंडाई", "hi_translit": "thandai",
        "aliases": ["kesar thandai", "holi thandai", "badam thandai", "banarasi thandai"]
    },
    {
        "name": "Kashmiri Kahwa", "category": "drink", "region": "Kashmir", "unit": "cup", "srv_g": 150.0,
        "cals": 22.0, "p": 0.5, "c": 4.5, "f": 0.3, "fib": 0.1,
        "gu_script": "કાહવા", "gu_translit": "kahwa",
        "hi_script": "काहवा", "hi_translit": "kahwa",
        "aliases": ["kahwa", "kashmiri tea kahwa", "kahwah", "kashmiri green tea"]
    },

    # ── 6. Teas & Coffees ─────────────────────────────────────────────────────
    {
        "name": "Masala Chai", "category": "drink", "region": "India", "unit": "cup", "srv_g": 150.0,
        "cals": 65.0, "p": 2.0, "c": 9.0, "f": 2.5, "fib": 0.0,
        "gu_script": "મસાલા ચા", "gu_translit": "masala cha",
        "hi_script": "मसाला चाय", "hi_translit": "masala chai",
        "aliases": ["chai", "tea", "indian tea", "milk tea", "adrak chai", "elaichi chai", "masala tea", "hot tea", "adrak wali chai"]
    },
    {
        "name": "Green Tea", "category": "drink", "region": "Global", "unit": "cup", "srv_g": 150.0,
        "cals": 1.0, "p": 0.1, "c": 0.2, "f": 0.0, "fib": 0.0,
        "gu_script": "ગ્રીન ટી", "gu_translit": "green tea",
        "hi_script": "ग्रीन टी", "hi_translit": "green tea",
        "aliases": ["plain green tea", "lemon green tea", "organic green tea", "green tea no sugar"]
    },
    {
        "name": "Black Tea", "category": "drink", "region": "Global", "unit": "cup", "srv_g": 150.0,
        "cals": 1.0, "p": 0.1, "c": 0.3, "f": 0.0, "fib": 0.0,
        "gu_script": "બ્લેક ટી", "gu_translit": "black tea",
        "hi_script": "काली चाय", "hi_translit": "kali chai",
        "aliases": ["lal chai", "lemon tea without sugar", "black tea without sugar", "plain black tea"]
    },
    {
        "name": "Filter Coffee", "category": "drink", "region": "South India", "unit": "cup", "srv_g": 150.0,
        "cals": 70.0, "p": 2.2, "c": 8.0, "f": 3.2, "fib": 0.0,
        "gu_script": "ફિલ્ટર કોફી", "gu_translit": "filter coffee",
        "hi_script": "फिल्टर कॉफी", "hi_translit": "filter coffee",
        "aliases": ["south indian filter coffee", "kaapi", "degree coffee", "filter kaapi", "madras filter coffee"]
    },
    {
        "name": "Black Coffee", "category": "drink", "region": "Global", "unit": "cup", "srv_g": 150.0,
        "cals": 2.0, "p": 0.2, "c": 0.3, "f": 0.0, "fib": 0.0,
        "gu_script": "બ્લેક કોફી", "gu_translit": "black coffee",
        "hi_script": "काली कॉफी", "hi_translit": "kali coffee",
        "aliases": ["americano", "black coffee no sugar", "espresso shot", "plain black coffee"]
    },
    {
        "name": "Cold Coffee", "category": "drink", "region": "Global", "unit": "glass", "srv_g": 250.0,
        "cals": 85.0, "p": 2.8, "c": 12.0, "f": 2.9, "fib": 0.0,
        "gu_script": "કોલ્ડ કોફી", "gu_translit": "cold coffee",
        "hi_script": "कोल्ड कॉफी", "hi_translit": "cold coffee",
        "aliases": ["iced coffee", "frappe", "cold coffee with milk", "cafe frappe"]
    },
    {
        "name": "Cappuccino", "category": "drink", "region": "Global", "unit": "cup", "srv_g": 200.0,
        "cals": 55.0, "p": 2.6, "c": 5.5, "f": 2.5, "fib": 0.0,
        "gu_script": "કેપુચીનો", "gu_translit": "cappuccino",
        "hi_script": "कैपुचीनो", "hi_translit": "cappuccino",
        "aliases": ["latte", "cafe latte", "hot coffee with foam", "cafe cappuccino"]
    },

    # ── 7. Protein & Sports Drinks ────────────────────────────────────────────
    {
        "name": "Whey Protein Shake", "category": "drink", "region": "Global", "unit": "glass", "srv_g": 300.0,
        "cals": 40.0, "p": 8.0, "c": 1.0, "f": 0.5, "fib": 0.2,
        "gu_script": "પ્રોટીન શેક", "gu_translit": "protein shake",
        "hi_script": "प्रोटीन शेक", "hi_translit": "protein shake",
        "aliases": ["protein shake", "whey shake", "protein drink", "whey protein in water", "isolate protein shake"]
    },
    {
        "name": "Gatorade", "category": "drink", "region": "Global", "unit": "bottle", "srv_g": 500.0,
        "cals": 24.0, "p": 0.0, "c": 6.0, "f": 0.0, "fib": 0.0,
        "gu_script": "ગેટોરેડ", "gu_translit": "gatorade",
        "hi_script": "गैटोरेड", "hi_translit": "gatorade",
        "aliases": ["gatorade sports drink", "sports drink", "electrolyte drink", "gatorade blue", "gatorade bottle"]
    },
    {
        "name": "Electral Solution", "category": "drink", "region": "India", "unit": "glass", "srv_g": 250.0,
        "cals": 16.0, "p": 0.0, "c": 4.0, "f": 0.0, "fib": 0.0,
        "gu_script": "ઓઆરએસ", "gu_translit": "ors",
        "hi_script": "ओआरएस", "hi_translit": "ors",
        "aliases": ["electral", "electral ors", "ors", "ors drink", "ors water", "electral water", "oral rehydration salts", "electral powder water"]
    },

    # ── 8. Mocktails & Sparkling Water ────────────────────────────────────────
    {
        "name": "Virgin Mojito", "category": "drink", "region": "Global", "unit": "glass", "srv_g": 250.0,
        "cals": 35.0, "p": 0.2, "c": 8.5, "f": 0.0, "fib": 0.2,
        "gu_script": "વર્જિન મોજિટો", "gu_translit": "virgin mojito",
        "hi_script": "वर्जिन मोजितो", "hi_translit": "virgin mojito",
        "aliases": ["mojito", "mint mojito mocktail", "lemon mint mojito", "virgin mojito mocktail", "mint mojito"]
    },
    {
        "name": "Blue Lagoon Mocktail", "category": "drink", "region": "Global", "unit": "glass", "srv_g": 250.0,
        "cals": 40.0, "p": 0.1, "c": 10.0, "f": 0.0, "fib": 0.0,
        "gu_script": "બ્લૂ લગૂન", "gu_translit": "blue lagoon",
        "hi_script": "ब्लू लगून", "hi_translit": "blue lagoon",
        "aliases": ["blue lagoon", "blue mocktail", "blue curacao mocktail", "blue lagoon drink"]
    },
    {
        "name": "Club Soda", "category": "drink", "region": "Global", "unit": "glass", "srv_g": 250.0,
        "cals": 0.0, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "સોડા", "gu_translit": "soda",
        "hi_script": "सोडा वॉटर", "hi_translit": "soda water",
        "aliases": ["soda", "soda water", "sparkling water", "plain soda", "carbonated water", "kinley soda", "perrier plain"]
    },
    {
        "name": "Flavored Sparkling Water", "category": "drink", "region": "Global", "unit": "can", "srv_g": 330.0,
        "cals": 0.5, "p": 0.0, "c": 0.0, "f": 0.0, "fib": 0.0,
        "gu_script": "ફ્લેવર્ડ સ્પાર્કલિંગ વોટર", "gu_translit": "flavored sparkling water",
        "hi_script": "फ्लेवर्ड स्पार्कलिंग वॉटर", "hi_translit": "flavored sparkling water",
        "aliases": ["lemon sparkling water", "flavored soda zero cal", "perrier flavored", "peach sparkling water"]
    }
]


def ingest_authentic_drinks():
    settings = get_settings()
    # Prioritize fast local MongoDB connection
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
        db = client[settings.MONGODB_DB_NAME]
    except Exception:
        client = MongoClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)
        db = client[settings.MONGODB_DB_NAME]

    foods_coll = db["foods"]
    aliases_coll = db["food_aliases"]

    initial_foods = foods_coll.count_documents({})
    initial_aliases = aliases_coll.count_documents({})
    print(f"Initial: {initial_foods} foods, {initial_aliases} aliases")

    inserted_foods = 0
    updated_aliases = 0
    now_iso = datetime.datetime.now().isoformat()

    for item in AUTHENTIC_DRINKS_CATALOG:
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
            # Update nutrition fields to authoritative values
            foods_coll.update_one(
                {"food_id": food_id},
                {"$set": {
                    "serving_size_g": srv_g,
                    "serving_unit": p_unit,
                    "calories_per_100g": cals_100g,
                    "calories_kcal": cals_serving,
                    "protein_g": item["p"],
                    "carbs_g": item["c"],
                    "fat_g": item["f"],
                    "fiber_g": item["fib"],
                    "is_verified": True,
                    "updated_at": now_iso
                }}
            )
        else:
            food_id = f"food_drk_{re.sub(r'[^\w]', '_', food_name_clean)}"
            schema = generate_food_schema_fields(canonical, category, "", p_unit)

            food_doc = {
                "food_id": food_id,
                "food_name": food_name_clean,
                "food_name_display": canonical,
                "aliases": [],
                "category": category,
                "subcategory": "authentic_beverage",
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
                "quantity_units": schema.get("quantity_units", ["glass", "cup", "can", "bottle", "ml", "litre"]),
                "quantity_options": schema.get("quantity_options", ["1 glass", "1 can", "1 bottle", "250ml", "500ml"]),
                "preparation_variants": schema.get("preparation_variants", ["Normal"]),
                "variant_nutrition": schema.get("variant_nutrition", {}),
                "is_verified": True,
                "data_source": "Authentic Drink Master",
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
    print(f"[DRINK INGESTION COMPLETE]")
    print(f"  Foods: {initial_foods} -> {final_foods} (+{inserted_foods} new drinks)")
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
    ingest_authentic_drinks()
