"""
Generator for 10,000+ Real Indian Food Scenarios
Covering:
1. Gujarati (Native script + Roman transliterations)
2. Hindi (Devanagari script + Hinglish)
3. English (Canonical Indian English)
4. Joined words (concatenated foods: dalrice, rotishak, masaladosa, etc.)
5. Typos & spelling mistakes (phonetic, character repetition, missing char, keyboard slips)
6. Compound foods (dal chawal, rajma chawal, chole bhature, idli sambar, etc.)
7. Regional names (Gujarati, Kathiyawadi, Punjabi, Maharashtrian, South Indian, Rajasthani, Bengali, Kashmiri)
8. Snacks (Farsan, chaat, street food, namkeen)
9. Drinks (Chaas, lassi, cutting chai, filter coffee, sharbat, badam milk, etc.)
10. Preparation variants (with ghee, roasted, fried, boiled, steamed, sugar-free, etc.)
11. Different scenarios (single items, quantities, composite meal logging in EN/GU/HI)
"""

import os
import json
import csv
import itertools
from typing import List, Dict, Any

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Clean dataset")

# ─────────────────────────────────────────────────────────────────────────────
# CORE AUTHENTIC INDIAN FOODS DATABASE (120+ Authentic Foods)
# ─────────────────────────────────────────────────────────────────────────────
BASE_FOODS = [
    # Breads / Flatbreads
    {
        "canonical": "Roti",
        "category": "bread_flatbread",
        "region": "Pan-India",
        "gu_script": "રોટલી",
        "gu_translit": "rotli",
        "hi_script": "रोटी",
        "hi_translit": "roti",
        "en_name": "Roti / Chapati",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["with ghee", "without ghee", "butter", "plain", "tandoori", "rumali"]
    },
    {
        "canonical": "Chapati",
        "category": "bread_flatbread",
        "region": "Pan-India",
        "gu_script": "ચપાતી",
        "gu_translit": "chapati",
        "hi_script": "चपाती",
        "hi_translit": "chapati",
        "en_name": "Wheat Chapati",
        "default_unit": "piece",
        "typical_qty": "3 pieces",
        "variants": ["with ghee", "without ghee", "oil roasted", "soft phulka"]
    },
    {
        "canonical": "Bhakri",
        "category": "bread_flatbread",
        "region": "Gujarat",
        "gu_script": "ભાખરી",
        "gu_translit": "bhakhri",
        "hi_script": "भाखरी",
        "hi_translit": "bhakri",
        "en_name": "Gujarati Wheat Bhakri",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["with ghee", "biscuit style", "crispy kadak", "masala bhakri", "jeera bhakri"]
    },
    {
        "canonical": "Bajra Rotlo",
        "category": "bread_flatbread",
        "region": "Gujarat",
        "gu_script": "બાજરીનો રોટલો",
        "gu_translit": "bajra no rotlo",
        "hi_script": "बाजरे की रोटी",
        "hi_translit": "bajra roti",
        "en_name": "Pearl Millet Flatbread (Bajra Rotlo)",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["with white butter (makhan)", "with ghee", "thick roasted", "garlic tadka"]
    },
    {
        "canonical": "Methi Thepla",
        "category": "bread_flatbread",
        "region": "Gujarat",
        "gu_script": "મેથી થેપલા",
        "gu_translit": "methi thepla",
        "hi_script": "मेथी थेपला",
        "hi_translit": "methi thepla",
        "en_name": "Fenugreek Spiced Flatbread (Thepla)",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["oil roasted", "with curd", "with chhas", "soft rolled", "travel pack"]
    },
    {
        "canonical": "Dudhi Thepla",
        "category": "bread_flatbread",
        "region": "Gujarat",
        "gu_script": "દૂધી થેપલા",
        "gu_translit": "dudhi thepla",
        "hi_script": "लौकी थेपला",
        "hi_translit": "lauki thepla",
        "en_name": "Bottle Gourd Thepla",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["oil roasted", "mild spiced", "with pickle", "soft"]
    },
    {
        "canonical": "Aloo Paratha",
        "category": "bread_flatbread",
        "region": "Punjab",
        "gu_script": "બટાકા પરાઠા",
        "gu_translit": "bataka paratha",
        "hi_script": "आलू पराठा",
        "hi_translit": "aloo paratha",
        "en_name": "Stuffed Potato Paratha",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["with amul butter", "with curd", "with pickle", "tawa fried", "less oil"]
    },
    {
        "canonical": "Paneer Paratha",
        "category": "bread_flatbread",
        "region": "Punjab",
        "gu_script": "પનીર પરાઠા",
        "gu_translit": "paneer paratha",
        "hi_script": "पनीर पराठा",
        "hi_translit": "paneer paratha",
        "en_name": "Cottage Cheese Stuffed Paratha",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["with butter", "with dahi", "spicy mint", "tawa roasted"]
    },
    {
        "canonical": "Gobi Paratha",
        "category": "bread_flatbread",
        "region": "Punjab",
        "gu_script": "કોબીજ પરાઠા",
        "gu_translit": "kobij paratha",
        "hi_script": "गोभी पराठा",
        "hi_translit": "gobi paratha",
        "en_name": "Cauliflower Stuffed Paratha",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["with butter", "with green chutney", "crispy tawa"]
    },
    {
        "canonical": "Puri",
        "category": "bread_flatbread",
        "region": "Pan-India",
        "gu_script": "પૂરી",
        "gu_translit": "puri",
        "hi_script": "पूरी",
        "hi_translit": "puri",
        "en_name": "Deep Fried Puffed Bread (Puri)",
        "default_unit": "piece",
        "typical_qty": "4 pieces",
        "variants": ["deep fried", "masala puri", "palak puri", "soft puffed"]
    },
    {
        "canonical": "Bhatura",
        "category": "bread_flatbread",
        "region": "Punjab",
        "gu_script": "ભટૂરા",
        "gu_translit": "bhatura",
        "hi_script": "भटूरा",
        "hi_translit": "bhatura",
        "en_name": "Fluffy Deep Fried Bhatura",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["deep fried", "paneer stuffed", "crispy golden"]
    },
    {
        "canonical": "Butter Naan",
        "category": "bread_flatbread",
        "region": "North India",
        "gu_script": "બટર નાન",
        "gu_translit": "butter naan",
        "hi_script": "बटर नान",
        "hi_translit": "butter naan",
        "en_name": "Tandoori Butter Naan",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["with butter", "garlic butter", "plain tandoori", "cheese garlic"]
    },
    {
        "canonical": "Kulcha",
        "category": "bread_flatbread",
        "region": "Punjab",
        "gu_script": "કુલચા",
        "gu_translit": "kulcha",
        "hi_script": "कुलचा",
        "hi_translit": "kulcha",
        "en_name": "Amritsari Stuffed Kulcha",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["amritsari aloo pyaz", "chur chur", "paneer stuffed", "with butter"]
    },
    {
        "canonical": "Makki Di Roti",
        "category": "bread_flatbread",
        "region": "Punjab",
        "gu_script": "મકાઈ રોટલી",
        "gu_translit": "makai rotli",
        "hi_script": "मक्के की रोटी",
        "hi_translit": "makki ki roti",
        "en_name": "Cornmeal Flatbread (Makki Roti)",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["with white butter", "with jaggery (gud)", "tandoor roasted", "tawa made"]
    },
    {
        "canonical": "Jowar Bhakri",
        "category": "bread_flatbread",
        "region": "Maharashtra",
        "gu_script": "જુવાર ભાખરી",
        "gu_translit": "juvar bhakhri",
        "hi_script": "ज्वार भाकरी",
        "hi_translit": "jowar bhakri",
        "en_name": "Sorghum Flatbread (Jowar Bhakri)",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["traditional flame roasted", "thin soft", "with ghee"]
    },
    {
        "canonical": "Puran Poli",
        "category": "sweet_dessert",
        "region": "Maharashtra",
        "gu_script": "પુરણ પોળી",
        "gu_translit": "puran poli",
        "hi_script": "पूरन पोली",
        "hi_translit": "puran poli",
        "en_name": "Sweet Stuffed Lentil Flatbread",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["with pure ghee", "with warm milk", "gujarati vedmi style", "jaggery sweet"]
    },

    # Dals & Kadhis
    {
        "canonical": "Dal Tadka",
        "category": "dal_lentil",
        "region": "Pan-India",
        "gu_script": "દાળ તડકા",
        "gu_translit": "dal tadka",
        "hi_script": "दाल तड़का",
        "hi_translit": "dal tadka",
        "en_name": "Yellow Dal Tadka",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["ghee tadka", "garlic tadka", "double tadka", "spicy jeera", "jeera fry"]
    },
    {
        "canonical": "Dal Fry",
        "category": "dal_lentil",
        "region": "Pan-India",
        "gu_script": "દાળ ફ્રાય",
        "gu_translit": "dal fry",
        "hi_script": "दाल फ्राई",
        "hi_translit": "dal fry",
        "en_name": "Spiced Toor Dal Fry",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["butter dal fry", "kolhapuri spicy", "dhaba style", "onion tomato masala"]
    },
    {
        "canonical": "Dal Makhani",
        "category": "dal_lentil",
        "region": "Punjab",
        "gu_script": "દાળ મખની",
        "gu_translit": "dal makhani",
        "hi_script": "दाल मखनी",
        "hi_translit": "dal makhani",
        "en_name": "Creamy Black Lentil Dal Makhani",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["with extra butter", "with fresh cream", "slow cooked overnight", "restaurant rich"]
    },
    {
        "canonical": "Gujarati Dal",
        "category": "dal_lentil",
        "region": "Gujarat",
        "gu_script": "ગુજરાતી દાળ",
        "gu_translit": "gujarati dal",
        "hi_script": "गुजराती दाल",
        "hi_translit": "gujarati dal",
        "en_name": "Sweet & Tangy Gujarati Tuver Dal",
        "default_unit": "katori",
        "typical_qty": "1 katori (120ml)",
        "variants": ["sweet with jaggery", "kokum sour", "peanuts added", "thin consistency"]
    },
    {
        "canonical": "Moong Dal",
        "category": "dal_lentil",
        "region": "Pan-India",
        "gu_script": "મગની દાળ",
        "gu_translit": "mag ni dal",
        "hi_script": "मूंग दाल",
        "hi_translit": "moong dal",
        "en_name": "Yellow Split Moong Dal",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["light boiled", "hing jeera tadka", "green whole moong", "khichdi accompaniment"]
    },
    {
        "canonical": "Chana Dal",
        "category": "dal_lentil",
        "region": "North India",
        "gu_script": "ચણાની દાળ",
        "gu_translit": "chana ni dal",
        "hi_script": "चना दाल",
        "hi_translit": "chana dal",
        "en_name": "Bengal Gram Dal",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["tadka fried", "with lauki", "with palak", "dhaba style"]
    },
    {
        "canonical": "Gujarati Kadhi",
        "category": "curry_gravy",
        "region": "Gujarat",
        "gu_script": "ગુજરાતી કઢી",
        "gu_translit": "gujarati kadhi",
        "hi_script": "गुजराती कढ़ी",
        "hi_translit": "gujarati kadhi",
        "en_name": "Sweet & Spiced Gujarati Yogurt Kadhi",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150ml)",
        "variants": ["sweet with jaggery", "cinnamon clove tempered", "ginger green chili", "thin creamy"]
    },
    {
        "canonical": "Punjabi Kadhi Pakora",
        "category": "curry_gravy",
        "region": "Punjab",
        "gu_script": "પંજાબી કઢી પકોડા",
        "gu_translit": "punjabi kadhi pakoda",
        "hi_script": "पंजाबी कढ़ी पकोड़ा",
        "hi_translit": "punjabi kadhi pakora",
        "en_name": "Punjabi Yogurt Curry with Crispy Pakoras",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (200g)",
        "variants": ["onion pakora", "sour curd based", "methi dana tadka", "thick gravy"]
    },
    {
        "canonical": "Sambar",
        "category": "dal_lentil",
        "region": "South India",
        "gu_script": "સાંભાર",
        "gu_translit": "sambhar",
        "hi_script": "सांभर",
        "hi_translit": "sambar",
        "en_name": "South Indian Vegetable Sambar",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150ml)",
        "variants": ["drumstick sambar", "shallot onion sambar", "spicy udipi style", "tamarind tangy"]
    },
    {
        "canonical": "Rasam",
        "category": "soup",
        "region": "South India",
        "gu_script": "રસમ",
        "gu_translit": "rasam",
        "hi_script": "रसम",
        "hi_translit": "rasam",
        "en_name": "Spiced Tamarind Tomato Broth (Rasam)",
        "default_unit": "cup",
        "typical_qty": "1 cup (150ml)",
        "variants": ["tomato rasam", "pepper garlic rasam", "mysore rasam", "hot digestive soup"]
    },

    # Rice Dishes & Khichdi
    {
        "canonical": "Jeera Rice",
        "category": "rice_grain",
        "region": "Pan-India",
        "gu_script": "જીરા રાઈસ",
        "gu_translit": "jeera rice",
        "hi_script": "जीरा राइस",
        "hi_translit": "jeera rice",
        "en_name": "Cumin Spiced Basmati Rice",
        "default_unit": "plate",
        "typical_qty": "1 plate (180g)",
        "variants": ["with pure ghee", "fragrant long grain", "coriander garnished"]
    },
    {
        "canonical": "Khichdi",
        "category": "rice_grain",
        "region": "Gujarat",
        "gu_script": "ખીચડી",
        "gu_translit": "khichdi",
        "hi_script": "खिचड़ी",
        "hi_translit": "khichdi",
        "en_name": "Moong Dal Rice Khichdi",
        "default_unit": "plate",
        "typical_qty": "1 plate (200g)",
        "variants": ["with desi ghee", "vaghareli (spiced tadka)", "moong chilkawali", "soft soothing"]
    },
    {
        "canonical": "Veg Biryani",
        "category": "rice_grain",
        "region": "Hyderabad",
        "gu_script": "વેજ બિરયાની",
        "gu_translit": "veg biryani",
        "hi_script": "वेज बिरयानी",
        "hi_translit": "veg biryani",
        "en_name": "Hyderabadi Vegetable Dum Biryani",
        "default_unit": "plate",
        "typical_qty": "1 plate (250g)",
        "variants": ["dum cooked", "with mix veg raita", "spicy salan", "fried onion garnish"]
    },
    {
        "canonical": "Chicken Biryani",
        "category": "rice_grain",
        "region": "Hyderabad",
        "gu_script": "ચિકન બિરયાની",
        "gu_translit": "chicken biryani",
        "hi_script": "चिकन बिरयानी",
        "hi_translit": "chicken biryani",
        "en_name": "Hyderabadi Chicken Biryani",
        "default_unit": "plate",
        "typical_qty": "1 plate (300g)",
        "variants": ["spicy dum cooked", "with mirchi ka salan", "boneless pieces", "boiled egg garnish"]
    },
    {
        "canonical": "Curd Rice",
        "category": "rice_grain",
        "region": "South India",
        "gu_script": "દહીં ભાત",
        "gu_translit": "dahi bhat",
        "hi_script": "दही चावल",
        "hi_translit": "curd rice",
        "en_name": "South Indian Tempered Curd Rice",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["mustard curry leaf tadka", "pomegranate garnish", "cooling summer meal", "urad dal crunch"]
    },
    {
        "canonical": "Pulao",
        "category": "rice_grain",
        "region": "Pan-India",
        "gu_script": "પુલાવ",
        "gu_translit": "pulao",
        "hi_script": "पुलाव",
        "hi_translit": "pulao",
        "en_name": "Garden Vegetable Pulao",
        "default_unit": "plate",
        "typical_qty": "1 plate (200g)",
        "variants": ["matar pulao", "paneer veg pulao", "kashmiri sweet pulao", "tawa pulao"]
    },

    # Vegetable Curries & Shaak
    {
        "canonical": "Sev Tameta Nu Shaak",
        "category": "vegetable_dry",
        "region": "Gujarat",
        "gu_script": "સેવ ટમેટાનું શાક",
        "gu_translit": "sev tameta nu shaak",
        "hi_script": "सेव टमाटर की सब्जी",
        "hi_translit": "sev tamatar sabzi",
        "en_name": "Kathiyawadi Sev Tameta Sabzi",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["sweet and spicy gravy", "with spicy ratlami sev", "kathiyawadi style", "coriander garnish"]
    },
    {
        "canonical": "Undhiyu",
        "category": "vegetable_dry",
        "region": "Gujarat",
        "gu_script": "ઊંધિયું",
        "gu_translit": "undhiyu",
        "hi_script": "उंधियू",
        "hi_translit": "undhiyu",
        "en_name": "Surti Undhiyu Mixed Vegetable",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["surti style with green muthia", "kathiyawadi spicy red", "winter delicacy", "slow cooked"]
    },
    {
        "canonical": "Lasaniya Bataka",
        "category": "vegetable_dry",
        "region": "Gujarat",
        "gu_script": "લસણીયા બટાકા",
        "gu_translit": "lasaniya bataka",
        "hi_script": "लहसुनी आलू",
        "hi_translit": "lahsuni aloo",
        "en_name": "Kathiyawadi Garlic Spiced Baby Potatoes",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["spicy red garlic paste", "kathiyawadi dhaba style", "baby potatoes dry"]
    },
    {
        "canonical": "Ringna No Oro",
        "category": "vegetable_dry",
        "region": "Gujarat",
        "gu_script": "રીંગણાનો ઓળો",
        "gu_translit": "ringna no oro",
        "hi_script": "बैंगन का भर्ता",
        "hi_translit": "baingan bharta",
        "en_name": "Kathiyawadi Smoked Eggplant Mash (Ringna Oro)",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["with spring garlic", "charcoal roasted", "green chili garlic tadka", "oil simmered"]
    },
    {
        "canonical": "Paneer Butter Masala",
        "category": "curry_gravy",
        "region": "North India",
        "gu_script": "પનીર બટર મસાલા",
        "gu_translit": "paneer butter masala",
        "hi_script": "पनीर बटर मसाला",
        "hi_translit": "paneer butter masala",
        "en_name": "Rich Cottage Cheese in Butter Tomato Gravy",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (200g)",
        "variants": ["extra butter", "fresh cream rich", "kasuri methi infused", "mild sweet tangy"]
    },
    {
        "canonical": "Palak Paneer",
        "category": "curry_gravy",
        "region": "North India",
        "gu_script": "પાલક પનીર",
        "gu_translit": "palak paneer",
        "hi_script": "पालक पनीर",
        "hi_translit": "palak paneer",
        "en_name": "Fresh Spinach Puree with Cottage Cheese",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["garlic tadka", "pan-fried paneer", "homestyle light", "restaurant creamy"]
    },
    {
        "canonical": "Kadhai Paneer",
        "category": "curry_gravy",
        "region": "North India",
        "gu_script": "કડાઈ પનીર",
        "gu_translit": "kadhai paneer",
        "hi_script": "कढ़ाई पनीर",
        "hi_translit": "kadhai paneer",
        "en_name": "Wok-Tossed Cottage Cheese with Bell Peppers",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["fresh pounded kadhai spices", "crisp bell peppers", "spicy thick masala"]
    },
    {
        "canonical": "Chole",
        "category": "curry_gravy",
        "region": "Punjab",
        "gu_script": "છોલે ચણા",
        "gu_translit": "chole chana",
        "hi_script": "छोले",
        "hi_translit": "chole",
        "en_name": "Amritsari Spiced Chickpea Curry",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["pindi chole dark", "amritsari style", "onion garlic free jain", "anardana sour"]
    },
    {
        "canonical": "Rajma",
        "category": "curry_gravy",
        "region": "North India",
        "gu_script": "રાજમા",
        "gu_translit": "rajma",
        "hi_script": "राजमा",
        "hi_translit": "rajma",
        "en_name": "Punjabi Red Kidney Bean Curry",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["chitra rajma", "thick onion tomato gravy", "ginger julienne garnish", "homestyle slow cooked"]
    },
    {
        "canonical": "Aloo Gobi",
        "category": "vegetable_dry",
        "region": "Pan-India",
        "gu_script": "બટાકા ફ્લાવર શાક",
        "gu_translit": "bataka flower nu shaak",
        "hi_script": "आलू गोभी",
        "hi_translit": "aloo gobi",
        "en_name": "Spiced Potato & Cauliflower Stir Fry",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["sukhi dry sabzi", "dhaba style semi gravy", "haldi jeera spiced", "ginger infused"]
    },
    {
        "canonical": "Bhindi Masala",
        "category": "vegetable_dry",
        "region": "Pan-India",
        "gu_script": "ભીંડાનું શાક",
        "gu_translit": "bhinda nu shaak",
        "hi_script": "भिंडी मसाला",
        "hi_translit": "bhindi masala",
        "en_name": "Okra Stir Fry with Onions & Spices",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (140g)",
        "variants": ["crispy fried", "besan stuffed", "onion masala (do pyaza)", "homestyle dry"]
    },
    {
        "canonical": "Sarson Ka Saag",
        "category": "vegetable_dry",
        "region": "Punjab",
        "gu_script": "સરસવનું શાક",
        "gu_translit": "sarsav nu shaak",
        "hi_script": "सरसों का साग",
        "hi_translit": "sarson ka saag",
        "en_name": "Punjabi Mustard Greens Puree",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (180g)",
        "variants": ["with white butter (makhan)", "slow cooked hand churned", "garlic ginger tadka"]
    },
    {
        "canonical": "Butter Chicken",
        "category": "curry_gravy",
        "region": "Punjab",
        "gu_script": "બટર ચિકન",
        "gu_translit": "butter chicken",
        "hi_script": "बटर चिकन",
        "hi_translit": "butter chicken",
        "en_name": "Tandoori Chicken in Silky Butter Gravy",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (220g)",
        "variants": ["tandoori roasted chicken", "rich buttery makhani sauce", "boneless pieces"]
    },

    # Snacks, Farsan & Street Food
    {
        "canonical": "Pani Puri",
        "category": "street_food_chaat",
        "region": "Pan-India",
        "gu_script": "પાણીપુરી",
        "gu_translit": "panipuri",
        "hi_script": "पानी पूरी",
        "hi_translit": "pani puri",
        "en_name": "Crispy Hollow Puris with Spiced Mint Water",
        "default_unit": "plate",
        "typical_qty": "1 plate (6 pieces)",
        "variants": ["spicy teekha pani", "sweet meetha pani", "ragda stuffing", "moong boondi mix"]
    },
    {
        "canonical": "Sev Puri",
        "category": "street_food_chaat",
        "region": "Maharashtra",
        "gu_script": "સેવ પુરી",
        "gu_translit": "sev puri",
        "hi_script": "सेव पूरी",
        "hi_translit": "sev puri",
        "en_name": "Crisp Puris Topped with Potatoes & Nylon Sev",
        "default_unit": "plate",
        "typical_qty": "1 plate (6 pieces)",
        "variants": ["with sweet chutney", "extra spicy garlic", "onion raw mango topping"]
    },
    {
        "canonical": "Bhel Puri",
        "category": "street_food_chaat",
        "region": "Maharashtra",
        "gu_script": "ભેળ પુરી",
        "gu_translit": "bhel puri",
        "hi_script": "भेल पूरी",
        "hi_translit": "bhel puri",
        "en_name": "Puffed Rice Chaat with Chutneys",
        "default_unit": "plate",
        "typical_qty": "1 plate (150g)",
        "variants": ["sukha bhel (dry)", "geela bhel (chutney wet)", "jain bhel", "spicy kolkata style"]
    },
    {
        "canonical": "Pav Bhaji",
        "category": "street_food_chaat",
        "region": "Maharashtra",
        "gu_script": "પાવ ભાજી",
        "gu_translit": "pav bhaji",
        "hi_script": "पाव भाजी",
        "hi_translit": "pav bhaji",
        "en_name": "Spiced Vegetable Mash with Butter Toasted Pav",
        "default_unit": "plate",
        "typical_qty": "1 plate (2 pav + bhaji)",
        "variants": ["amul butter pav", "cheese pav bhaji", "jain pav bhaji", "khada bhaji chunk style"]
    },
    {
        "canonical": "Vada Pav",
        "category": "street_food_chaat",
        "region": "Maharashtra",
        "gu_script": "વડા પાવ",
        "gu_translit": "vada pav",
        "hi_script": "वड़ा पाव",
        "hi_translit": "vada pav",
        "en_name": "Spiced Potato Fritter in Bread Roll",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["dry red garlic chutney", "fried salted green chili", "butter grilled pav"]
    },
    {
        "canonical": "Samosa",
        "category": "snack",
        "region": "Pan-India",
        "gu_script": "સમોસા",
        "gu_translit": "samosa",
        "hi_script": "समोसा",
        "hi_translit": "samosa",
        "en_name": "Crispy Spiced Potato Filled Pastry",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["potato peas filling", "cocktail mini samosa", "paneer samosa", "with mint chutney"]
    },
    {
        "canonical": "Kachori",
        "category": "snack",
        "region": "Rajasthan",
        "gu_script": "કચોરી",
        "gu_translit": "kachori",
        "hi_script": "कचौरी",
        "hi_translit": "kachori",
        "en_name": "Flaky Spiced Dal / Onion Pastry",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["pyaaz ki kachori", "moong dal khasta", "lilva kachori (gujarati)", "with sweet tamarind"]
    },
    {
        "canonical": "Khaman",
        "category": "snack",
        "region": "Gujarat",
        "gu_script": "ખમણ",
        "gu_translit": "khaman",
        "hi_script": "खमण",
        "hi_translit": "khaman",
        "en_name": "Steamed Spongy Gram Flour Khaman",
        "default_unit": "plate",
        "typical_qty": "1 plate (150g)",
        "variants": ["nylon khaman juicy", "surti sev khamani", "mustard green chili vaghar", "with raw papaya relish"]
    },
    {
        "canonical": "Dhokla",
        "category": "snack",
        "region": "Gujarat",
        "gu_script": "ઢોકળાં",
        "gu_translit": "dhokla",
        "hi_script": "ढोकला",
        "hi_translit": "dhokla",
        "en_name": "Steamed Fermented Rice & Dal Dhokla",
        "default_unit": "plate",
        "typical_qty": "1 plate (150g)",
        "variants": ["white khatta dhokla", "rava dhokla", "sandwich dhokla", "sprinkled black pepper"]
    },
    {
        "canonical": "Khandvi",
        "category": "snack",
        "region": "Gujarat",
        "gu_script": "ખાંડવી",
        "gu_translit": "khandvi",
        "hi_script": "खांडवी",
        "hi_translit": "khandvi",
        "en_name": "Rolled Gram Flour & Buttermilk Pinwheels",
        "default_unit": "plate",
        "typical_qty": "1 plate (120g)",
        "variants": ["coconut coriander tempered", "mustard sesame vaghar", "delicate smooth rolls"]
    },
    {
        "canonical": "Handvo",
        "category": "snack",
        "region": "Gujarat",
        "gu_script": "હાંડવો",
        "gu_translit": "handvo",
        "hi_script": "हांडवो",
        "hi_translit": "handvo",
        "en_name": "Baked Savory Lentil & Vegetable Cake",
        "default_unit": "piece",
        "typical_qty": "2 pieces (150g)",
        "variants": ["sesame crispy crust", "bottle gourd stuffed", "pan baked crunchy", "with green chutney"]
    },
    {
        "canonical": "Fafda",
        "category": "snack",
        "region": "Gujarat",
        "gu_script": "ફાફડા",
        "gu_translit": "fafda",
        "hi_script": "फाफड़ा",
        "hi_translit": "fafda",
        "en_name": "Crisp Gram Flour Strips (Fafda)",
        "default_unit": "plate",
        "typical_qty": "1 plate (100g)",
        "variants": ["with sweet jalebi", "with besan kadhi", "with papaya sambharo", "fried green chili"]
    },
    {
        "canonical": "Khakhra",
        "category": "snack",
        "region": "Gujarat",
        "gu_script": "ખાખરા",
        "gu_translit": "khakhra",
        "hi_script": "खाखरा",
        "hi_translit": "khakhra",
        "en_name": "Crispy Thin Whole Wheat Crackers",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["methi khakhra", "jeera khakhra", "masala khakhra", "plain ghee roasted", "chorafali"]
    },
    {
        "canonical": "Dabeli",
        "category": "street_food_chaat",
        "region": "Gujarat",
        "gu_script": "કચ્છી દાબેલી",
        "gu_translit": "kacchi dabeli",
        "hi_script": "दाबेली",
        "hi_translit": "dabeli",
        "en_name": "Spiced Potato Burger with Peanuts & Pomegranate",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["butter grilled pav", "spiced masala peanuts", "sev pomegranate topping", "garlic red chutney"]
    },
    {
        "canonical": "Poha",
        "category": "breakfast",
        "region": "Maharashtra",
        "gu_script": "પૌંઆ",
        "gu_translit": "pauva",
        "hi_script": "पोहा",
        "hi_translit": "poha",
        "en_name": "Flattened Rice with Onions & Peanuts",
        "default_unit": "plate",
        "typical_qty": "1 plate (150g)",
        "variants": ["kanda poha (onion)", "batata poha (potato)", "lemon coriander tempered", "indori steamed poha"]
    },
    {
        "canonical": "Idli",
        "category": "breakfast",
        "region": "South India",
        "gu_script": "ઈડલી",
        "gu_translit": "idli",
        "hi_script": "इडली",
        "hi_translit": "idli",
        "en_name": "Steamed Fermented Rice Cakes",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["steamed soft", "mini podi idli", "rava idli", "with sambar and coconut chutney"]
    },
    {
        "canonical": "Masala Dosa",
        "category": "breakfast",
        "region": "South India",
        "gu_script": "મસાલા ઢોસા",
        "gu_translit": "masala dosa",
        "hi_script": "मसाला डोसा",
        "hi_translit": "masala dosa",
        "en_name": "Crispy Crepe with Spiced Potato Filling",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["ghee roast", "mysore spicy red chutney", "paper thin crisp", "onion rava dosa"]
    },
    {
        "canonical": "Medu Vada",
        "category": "breakfast",
        "region": "South India",
        "gu_script": "મેદુ વડા",
        "gu_translit": "medu vada",
        "hi_script": "मेदु वड़ा",
        "hi_translit": "medu vada",
        "en_name": "Crispy Fried Lentil Doughnuts",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["crisp fried", "sambar dipped", "with coconut chutney", "peppercorn ginger spiced"]
    },
    {
        "canonical": "Uttapam",
        "category": "breakfast",
        "region": "South India",
        "gu_script": "ઉત્તપમ",
        "gu_translit": "uttapam",
        "hi_script": "उत्तपम",
        "hi_translit": "uttapam",
        "en_name": "Thick Savory Rice Pancake with Veggies",
        "default_unit": "piece",
        "typical_qty": "1 piece",
        "variants": ["onion tomato topping", "mixed veg chili", "podi sprinkled", "soft spongy"]
    },

    # Drinks & Traditional Beverages
    {
        "canonical": "Masala Chai",
        "category": "drink",
        "region": "Pan-India",
        "gu_script": "મસાલા ચા",
        "gu_translit": "masala chai",
        "hi_script": "मसाला चाय",
        "hi_translit": "masala chai",
        "en_name": "Spiced Indian Milk Tea",
        "default_unit": "cup",
        "typical_qty": "1 cup (150ml)",
        "variants": ["with adrak (ginger)", "with elaichi (cardamom)", "without sugar", "cutting chai", "kulhad chai"]
    },
    {
        "canonical": "Cutting Chai",
        "category": "drink",
        "region": "Maharashtra",
        "gu_script": "કટિંગ ચા",
        "gu_translit": "cutting chai",
        "hi_script": "कटिंग चाय",
        "hi_translit": "cutting chai",
        "en_name": "Mumbai Half-Measure Strong Spiced Tea",
        "default_unit": "cup",
        "typical_qty": "1 small glass (80ml)",
        "variants": ["strong brew (kadak)", "extra ginger", "tapri style"]
    },
    {
        "canonical": "Chaas",
        "category": "drink",
        "region": "Gujarat",
        "gu_script": "છાસ",
        "gu_translit": "chhas",
        "hi_script": "छाछ",
        "hi_translit": "chaas",
        "en_name": "Spiced Buttermilk (Chaas / Mattha)",
        "default_unit": "glass",
        "typical_qty": "1 glass (250ml)",
        "variants": ["roasted jeera (cumin)", "masala mint", "plain salted", "chilled summer digestive"]
    },
    {
        "canonical": "Sweet Lassi",
        "category": "drink",
        "region": "Punjab",
        "gu_script": "મીઠી લસ્સી",
        "gu_translit": "meethi lassi",
        "hi_script": "मीठी लस्सी",
        "hi_translit": "sweet lassi",
        "en_name": "Sweetened Churned Yogurt Lassi",
        "default_unit": "glass",
        "typical_qty": "1 glass (300ml)",
        "variants": ["with malai top", "mango lassi", "rose flavored", "sugar free", "kesar pista"]
    },
    {
        "canonical": "Filter Coffee",
        "category": "drink",
        "region": "South India",
        "gu_script": "ફિલ્ટર કોફી",
        "gu_translit": "filter coffee",
        "hi_script": "फ़िल्टर कॉफ़ी",
        "hi_translit": "filter coffee",
        "en_name": "South Indian Frothy Filter Kaapi",
        "default_unit": "cup",
        "typical_qty": "1 cup (120ml)",
        "variants": ["strong decoction", "dabarah tumbler served", "frothy hot milk", "without sugar"]
    },
    {
        "canonical": "Nimbu Pani",
        "category": "drink",
        "region": "Pan-India",
        "gu_script": "લીંબુ પાણી",
        "gu_translit": "limbu pani",
        "hi_script": "नींबू पानी",
        "hi_translit": "nimbu pani",
        "en_name": "Fresh Spiced Lemonade",
        "default_unit": "glass",
        "typical_qty": "1 glass (250ml)",
        "variants": ["sweet & salted", "shikanji with black salt", "soda mixed", "mint infused"]
    },
    {
        "canonical": "Badam Milk",
        "category": "drink",
        "region": "North India",
        "gu_script": "બદામ દૂધ",
        "gu_translit": "badam doodh",
        "hi_script": "बादाम दूध",
        "hi_translit": "badam milk",
        "en_name": "Almond Saffron Flavored Milk",
        "default_unit": "glass",
        "typical_qty": "1 glass (200ml)",
        "variants": ["warm winter milk", "chilled summer shake", "kesar saffron garnished", "pistachio crunch"]
    },
    {
        "canonical": "Sugarcane Juice",
        "category": "drink",
        "region": "Pan-India",
        "gu_script": "શેરડીનો રસ",
        "gu_translit": "sherdi no ras",
        "hi_script": "गन्ने का रस",
        "hi_translit": "ganne ka ras",
        "en_name": "Fresh Sugarcane Juice",
        "default_unit": "glass",
        "typical_qty": "1 glass (300ml)",
        "variants": ["with lemon & ginger", "without ice", "mint infused", "chilled freshly pressed"]
    },
    {
        "canonical": "Jaljeera",
        "category": "drink",
        "region": "North India",
        "gu_script": "જલજીરા",
        "gu_translit": "jaljeera",
        "hi_script": "जलजीरा",
        "hi_translit": "jaljeera",
        "en_name": "Spiced Cumin & Mint Cooler",
        "default_unit": "glass",
        "typical_qty": "1 glass (250ml)",
        "variants": ["with boondi pearls", "black salt digestive", "lemon mint iced"]
    },
    {
        "canonical": "Thandai",
        "category": "drink",
        "region": "North India",
        "gu_script": "ઠંડાઈ",
        "gu_translit": "thandai",
        "hi_script": "ठंडाई",
        "hi_translit": "thandai",
        "en_name": "Spiced Nut & Poppy Seed Festive Milk Drink",
        "default_unit": "glass",
        "typical_qty": "1 glass (250ml)",
        "variants": ["holi special", "fennel rose flavored", "rich cashew almond paste"]
    },

    # Sweets & Desserts
    {
        "canonical": "Gulab Jamun",
        "category": "sweet_dessert",
        "region": "Pan-India",
        "gu_script": "ગુલાબ જાંબુ",
        "gu_translit": "gulab jambu",
        "hi_script": "गुलाब जामुन",
        "hi_translit": "gulab jamun",
        "en_name": "Fried Milk Dumplings in Rose Sugar Syrup",
        "default_unit": "piece",
        "typical_qty": "2 pieces",
        "variants": ["warm in cardamom syrup", "stuffed dry fruit", "kala jamun"]
    },
    {
        "canonical": "Jalebi",
        "category": "sweet_dessert",
        "region": "Pan-India",
        "gu_script": "જલેબી",
        "gu_translit": "jalebi",
        "hi_script": "जलेबी",
        "hi_translit": "jalebi",
        "en_name": "Crispy Spiral Fritters in Sugar Syrup",
        "default_unit": "plate",
        "typical_qty": "100g",
        "variants": ["pure ghee fried", "thin crispy", "with fafda", "with chilled rabdi"]
    },
    {
        "canonical": "Sukhdi",
        "category": "sweet_dessert",
        "region": "Gujarat",
        "gu_script": "સુખડી",
        "gu_translit": "sukhdi",
        "hi_script": "सुखड़ी",
        "hi_translit": "sukhdi",
        "en_name": "Traditional Gujarati Whole Wheat & Jaggery Fudge",
        "default_unit": "piece",
        "typical_qty": "2 pieces (60g)",
        "variants": ["soft melt in mouth", "desi ghee jaggery roasted", "mahudha style"]
    },
    {
        "canonical": "Kheer",
        "category": "sweet_dessert",
        "region": "Pan-India",
        "gu_script": "ખીર",
        "gu_translit": "kheer",
        "hi_script": "खीर",
        "hi_translit": "kheer",
        "en_name": "Slow-Simmered Rice & Cardamom Milk Pudding",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["basmati rice kheer", "sevaiyan kheer", "saffron almond rich", "sugar free jaggery"]
    },

    # Fresh Fruits
    {
        "canonical": "Apple",
        "category": "fruit",
        "region": "Pan-India",
        "gu_script": "સફરજન",
        "gu_translit": "safarjan",
        "hi_script": "सेब",
        "hi_translit": "seb",
        "en_name": "Fresh Apple",
        "default_unit": "piece",
        "typical_qty": "1 piece (150g)",
        "variants": ["fresh sliced", "raw with peel", "kashmiri red"]
    },
    {
        "canonical": "Banana",
        "category": "fruit",
        "region": "Pan-India",
        "gu_script": "કેળું",
        "gu_translit": "kelu",
        "hi_script": "केला",
        "hi_translit": "kela",
        "en_name": "Ripe Yellow Banana",
        "default_unit": "piece",
        "typical_qty": "1 piece (100g)",
        "variants": ["fresh ripe", "elaichi banana", "robusta"]
    },
    {
        "canonical": "Mango",
        "category": "fruit",
        "region": "Pan-India",
        "gu_script": "કેરી",
        "gu_translit": "keri",
        "hi_script": "आम",
        "hi_translit": "aam",
        "en_name": "Alphonso Mango",
        "default_unit": "piece",
        "typical_qty": "1 piece (200g)",
        "variants": ["fresh cut cubes", "alphonso hapoos", "kesar mango"]
    },
    {
        "canonical": "Papaya",
        "category": "fruit",
        "region": "Pan-India",
        "gu_script": "પપૈયું",
        "gu_translit": "papaiyu",
        "hi_script": "पपीता",
        "hi_translit": "papita",
        "en_name": "Fresh Papaya Slices",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["fresh ripe diced", "chilled cubes"]
    },
    {
        "canonical": "Watermelon",
        "category": "fruit",
        "region": "Pan-India",
        "gu_script": "તરબૂચ",
        "gu_translit": "tarbuj",
        "hi_script": "तरबूज",
        "hi_translit": "tarbooj",
        "en_name": "Fresh Watermelon",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (200g)",
        "variants": ["fresh diced", "chilled slices", "with chaat masala"]
    },
    {
        "canonical": "Pomegranate",
        "category": "fruit",
        "region": "Pan-India",
        "gu_script": "દાડમ",
        "gu_translit": "dadam",
        "hi_script": "अनार",
        "hi_translit": "anar",
        "en_name": "Fresh Pomegranate Arils",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (100g)",
        "variants": ["fresh seeds", "sweet red arils"]
    },

    # Dairy Products
    {
        "canonical": "Curd",
        "category": "dairy",
        "region": "Pan-India",
        "gu_script": "દહીં",
        "gu_translit": "dahi",
        "hi_script": "दही",
        "hi_translit": "dahi",
        "en_name": "Fresh Plain Curd (Dahi)",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["fresh homestyle", "hung curd", "low fat dahi", "sweet curd"]
    },
    {
        "canonical": "Cow Milk",
        "category": "dairy",
        "region": "Pan-India",
        "gu_script": "ગાયનું દૂધ",
        "gu_translit": "doodh",
        "hi_script": "गाय का दूध",
        "hi_translit": "doodh",
        "en_name": "Fresh Whole Milk",
        "default_unit": "glass",
        "typical_qty": "1 glass (200ml)",
        "variants": ["warm boiled", "skimmed", "without sugar", "turmeric milk (haldi doodh)"]
    },
    {
        "canonical": "Paneer",
        "category": "dairy",
        "region": "Pan-India",
        "gu_script": "પનીર",
        "gu_translit": "paneer",
        "hi_script": "पनीर",
        "hi_translit": "paneer",
        "en_name": "Fresh Cottage Cheese (Paneer)",
        "default_unit": "serving",
        "typical_qty": "100g",
        "variants": ["raw fresh cubes", "pan fried", "low fat paneer", "grilled"]
    },
    {
        "canonical": "Ghee",
        "category": "dairy",
        "region": "Pan-India",
        "gu_script": "ઘી",
        "gu_translit": "ghee",
        "hi_script": "घी",
        "hi_translit": "ghee",
        "en_name": "Pure Desi Cow Ghee",
        "default_unit": "tbsp",
        "typical_qty": "1 tbsp (15g)",
        "variants": ["a2 desi bilona", "clarified butter", "hot melted"]
    },
    {
        "canonical": "White Butter",
        "category": "dairy",
        "region": "North India",
        "gu_script": "માખણ",
        "gu_translit": "makhan",
        "hi_script": "मक्खन",
        "hi_translit": "makhan",
        "en_name": "Fresh Homemade White Butter (Makhan)",
        "default_unit": "tbsp",
        "typical_qty": "1 tbsp (15g)",
        "variants": ["fresh churned", "unsalted homestyle"]
    },

    # Packaged Foods & Healthy Staples
    {
        "canonical": "Oats",
        "category": "packaged_food",
        "region": "Pan-India",
        "gu_script": "ઓટ્સ",
        "gu_translit": "oats",
        "hi_script": "ओट्स",
        "hi_translit": "oats",
        "en_name": "Rolled Oats Porridge",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (150g)",
        "variants": ["cooked in milk", "masala veg oats", "overnight oats", "with fruits"]
    },
    {
        "canonical": "Maggi Noodles",
        "category": "packaged_food",
        "region": "Pan-India",
        "gu_script": "મેગી",
        "gu_translit": "maggi",
        "hi_script": "मैगी",
        "hi_translit": "maggi",
        "en_name": "Instant Masala Maggi Noodles",
        "default_unit": "bowl",
        "typical_qty": "1 bowl (1 pack 70g)",
        "variants": ["classic masala", "vegetable cheese maggi", "soupy style"]
    },
    {
        "canonical": "Brown Bread",
        "category": "packaged_food",
        "region": "Pan-India",
        "gu_script": "બ્રાઉન બ્રેડ",
        "gu_translit": "brown bread",
        "hi_script": "ब्राउन ब्रेड",
        "hi_translit": "brown bread",
        "en_name": "Whole Wheat Brown Bread",
        "default_unit": "slice",
        "typical_qty": "2 slices (60g)",
        "variants": ["toasted crisp", "plain soft", "with peanut butter", "with butter"]
    },
    {
        "canonical": "Peanut Butter",
        "category": "packaged_food",
        "region": "Pan-India",
        "gu_script": "પીનટ બટર",
        "gu_translit": "peanut butter",
        "hi_script": "पीनट बटर",
        "hi_translit": "peanut butter",
        "en_name": "Natural Creamy Peanut Butter",
        "default_unit": "tbsp",
        "typical_qty": "1 tbsp (16g)",
        "variants": ["crunchy", "creamy unsweetened", "high protein"]
    },
    {
        "canonical": "Marie Biscuit",
        "category": "packaged_food",
        "region": "Pan-India",
        "gu_script": "મેરી બિસ્કિટ",
        "gu_translit": "marie biscuit",
        "hi_script": "मैरी बिस्कुट",
        "hi_translit": "marie biscuit",
        "en_name": "Crisp Tea Marie Biscuits",
        "default_unit": "piece",
        "typical_qty": "4 biscuits (30g)",
        "variants": ["plain crisp", "with chai dip"]
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# TYPO RULES GENERATOR ("food name mistack", "speling mistac")
# ─────────────────────────────────────────────────────────────────────────────
def generate_realistic_typos(word: str) -> List[Dict[str, str]]:
    """Generates realistic Indian typo variations for a given word."""
    typos = []
    w = word.lower()

    # 1. Letter repetition (double letter slips)
    for i in range(len(w)):
        if w[i] in "aeiousnmrtd":
            rep = w[:i] + w[i] * 2 + w[i:]
            if rep != w:
                typos.append({"text": rep, "type": "char_repetition"})
            rep3 = w[:i] + w[i] * 3 + w[i:]
            if rep3 != w:
                typos.append({"text": rep3, "type": "char_repetition_extreme"})

    # 2. Missing character (fast typing drop)
    for i in range(1, len(w) - 1):
        dropped = w[:i] + w[i+1:]
        if len(dropped) >= 3:
            typos.append({"text": dropped, "type": "missing_char"})

    # 3. Vowel substitution & phonetic shifts
    vowel_subs = [
        ("ee", "i"), ("i", "ee"),
        ("oo", "u"), ("u", "oo"),
        ("a", "aa"), ("aa", "a"),
        ("kh", "k"), ("k", "kh"),
        ("bh", "b"), ("b", "bh"),
        ("dh", "d"), ("d", "dh"),
        ("th", "t"), ("t", "th"),
        ("sh", "s"), ("s", "sh"),
        ("ch", "chh"), ("chh", "ch"),
        ("v", "w"), ("w", "v"),
        ("y", "i"), ("i", "y")
    ]
    for src, dst in vowel_subs:
        if src in w:
            ph = w.replace(src, dst)
            if ph != w:
                typos.append({"text": ph, "type": "phonetic_transcription"})

    # 4. Keyboard neighbor swap (QWERTY slips)
    adj_map = {
        'a': 's', 's': 'a', 'e': 'w', 'i': 'o', 'o': 'p', 'd': 's',
        'k': 'j', 'p': 'o', 'r': 't', 't': 'r', 'n': 'm', 'm': 'n'
    }
    for i in range(len(w)):
        if w[i] in adj_map:
            swap = w[:i] + adj_map[w[i]] + w[i+1:]
            typos.append({"text": swap, "type": "keyboard_slip"})

    # 5. Transposition of adjacent letters
    for i in range(len(w) - 1):
        if w[i] != w[i+1]:
            transposed = w[:i] + w[i+1] + w[i] + w[i+2:]
            typos.append({"text": transposed, "type": "transposed_letters"})

    return typos

# ─────────────────────────────────────────────────────────────────────────────
# 10,000+ SCENARIO BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_10000_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    seen_inputs = set()
    counter = 1

    def add_record(canonical: str, user_input: str, scenario_type: str, language: str,
                   category: str, region: str, prep_variant: str = "normal",
                   unit: str = "piece", typical_qty: str = "1 piece"):
        nonlocal counter
        inp_clean = user_input.strip()
        if not inp_clean or inp_clean.lower() in seen_inputs:
            return
        seen_inputs.add(inp_clean.lower())

        scenarios.append({
            "id": f"RIF_{counter:05d}",
            "food_name_canonical": canonical,
            "scenario_input": inp_clean,
            "scenario_type": scenario_type,
            "language": language,
            "category": category,
            "region": region,
            "preparation_variant": prep_variant,
            "portion_unit": unit,
            "typical_portion": typical_qty,
            "expected_resolved_food": canonical
        })
        counter += 1

    print("[1/10] Building Canonical Multilingual Entries (Gujarati, Hindi, English)...")
    for f in BASE_FOODS:
        c = f["canonical"]
        cat = f["category"]
        reg = f["region"]
        u = f["default_unit"]
        q = f["typical_qty"]

        # English
        add_record(c, f["canonical"].lower(), "english_canonical", "English", cat, reg, "normal", u, q)
        add_record(c, f["en_name"], "english_canonical", "English", cat, reg, "normal", u, q)

        # Gujarati Script
        add_record(c, f["gu_script"], "gujarati_native_script", "Gujarati", cat, reg, "normal", u, q)
        add_record(c, f["gu_translit"], "gujarati_translit", "Gujlish", cat, reg, "normal", u, q)

        # Hindi Script
        add_record(c, f["hi_script"], "hindi_devanagari_script", "Hindi", cat, reg, "normal", u, q)
        add_record(c, f["hi_translit"], "hindi_translit", "Hinglish", cat, reg, "normal", u, q)

    print(f"       -> Count so far: {len(scenarios)}")

    print("[2/10] Building Joined Words (concatenated compound words: dalrice, rotishak, etc.)...")
    # Common real Indian compound food pairings that users write without spaces
    joined_pairs = [
        ("Dal Tadka", "Steamed Basmati Rice", "dalrice", "Dal Rice"),
        ("Dal Tadka", "Jeera Rice", "daljeerarice", "Dal Jeera Rice"),
        ("Dal Fry", "Jeera Rice", "dalfryrice", "Dal Fry Jeera Rice"),
        ("Dal Makhani", "Butter Naan", "dalmakhaninaan", "Dal Makhani Naan"),
        ("Roti", "Sev Tameta Nu Shaak", "rotishak", "Roti Shak"),
        ("Roti", "Aloo Gobi", "rotisabzi", "Roti Sabzi"),
        ("Roti", "Dal Tadka", "rotidal", "Roti Dal"),
        ("Bhakri", "Ringna No Oro", "bhakrioro", "Bhakri Oro"),
        ("Bajra Rotlo", "Ringna No Oro", "rotlooro", "Rotlo Oro"),
        ("Khichdi", "Gujarati Kadhi", "khichdikadhi", "Khichdi Kadhi"),
        ("Masala Dosa", "Sambar", "masaladosasambar", "Masala Dosa Sambar"),
        ("Idli", "Sambar", "idlisambar", "Idli Sambar"),
        ("Medu Vada", "Sambar", "vadasambar", "Vada Sambar"),
        ("Idli", "Medu Vada", "idlivada", "Idli Vada"),
        ("Chole", "Bhatura", "cholebhature", "Chole Bhature"),
        ("Rajma", "Steamed Basmati Rice", "rajmachawal", "Rajma Chawal"),
        ("Pav Bhaji", "Amul Butter", "pavbhaji", "Pav Bhaji"),
        ("Vada Pav", "Green Chutney", "vadapav", "Vada Pav"),
        ("Pani Puri", "Mint Water", "panipuri", "Pani Puri"),
        ("Sev Puri", "Sweet Chutney", "sevpuri", "Sev Puri"),
        ("Bhel Puri", "Nylon Sev", "bhelpuri", "Bhel Puri"),
        ("Dahi Puri", "Sweet Curd", "dahipuri", "Dahi Puri"),
        ("Dahi Vada", "Sweet Chutney", "dahivada", "Dahi Vada"),
        ("Aloo Tikki", "Chaat Chutney", "alootikki", "Aloo Tikki"),
        ("Samosa", "Chaat Chutney", "samosachaat", "Samosa Chaat"),
        ("Aloo Paratha", "Fresh Curd", "alooparathadahi", "Aloo Paratha Dahi"),
        ("Methi Thepla", "Chaas", "theplachaas", "Thepla Chaas"),
        ("Methi Thepla", "Masala Chai", "theplachai", "Thepla Chai"),
        ("Fafda", "Jalebi", "fafdajalebi", "Fafda Jalebi"),
        ("Khaman", "Sev", "khaman", "Khaman Sev"),
        ("Palak Paneer", "Roti", "palakpaneerroti", "Palak Paneer Roti"),
        ("Paneer Butter Masala", "Butter Naan", "paneerbuttermasalanaan", "Paneer Butter Masala Naan"),
        ("Kadhai Paneer", "Tandoori Roti", "kadhaipaneerroti", "Kadhai Paneer Roti"),
        ("Sarson Ka Saag", "Makki Di Roti", "sarsonkasaag", "Sarson Ka Saag"),
        ("Makki Di Roti", "Sarson Ka Saag", "makkikiroti", "Makki Ki Roti"),
        ("Poha", "Masala Chai", "pohachai", "Poha Chai"),
        ("Upma", "Filter Coffee", "upmacoffee", "Upma Coffee"),
        ("Idli", "Filter Coffee", "idlicoffee", "Idli Coffee"),
        ("Dosa", "Filter Coffee", "dosacoffee", "Dosa Coffee"),
        ("Sabudana Khichdi", "Sweet Curd", "sabudanakhichdi", "Sabudana Khichdi"),
        ("Sabudana Vada", "Curd Dip", "sabudanavada", "Sabudana Vada"),
        ("Kanda Poha", "Sev", "kandapoha", "Kanda Poha"),
        ("Misal Pav", "Farsan", "misalpav", "Misal Pav"),
        ("Usal Pav", "Sev", "usalpav", "Usal Pav"),
        ("Sev Usal", "Spring Onion", "sevusal", "Sev Usal"),
        ("Badam Milk", "Warm Saffron", "badammilk", "Badam Milk"),
        ("Nimbu Pani", "Black Salt", "nimbupani", "Nimbu Pani"),
        ("Cutting Chai", "Ginger Cardamom", "cuttingchai", "Cutting Chai"),
        ("Masala Chai", "Adrak Wali", "masalachai", "Masala Chai"),
        ("Sweet Lassi", "Creamy Malai", "sweetlassi", "Sweet Lassi"),
        ("Mango Lassi", "Pulp Yogurt", "mangolassi", "Mango Lassi")
    ]

    quantities_prefixes = ["", "1 ", "2 ", "3 ", "ek ", "be ", "tran ", "do ", "teen ", "plate ", "bowl ", "glass "]
    for head_food, sub_food, joined_term, display_pair in joined_pairs:
        for pfx in quantities_prefixes:
            term = f"{pfx}{joined_term}".strip()
            add_record(head_food, term, "joined_words", "English/Roman", "composite_meal", "Pan-India", "normal", "serving", "1 serving")
            # Also add hyphenated version
            if pfx:
                add_record(head_food, f"{pfx}{joined_term.replace(' ', '')}", "joined_words", "English/Roman", "composite_meal", "Pan-India", "normal", "serving", "1 serving")

    print(f"       -> Count so far: {len(scenarios)}")

    print("[3/10] Building Realistic Typos & Spelling Mistakes (character drops, repeats, phonetics)...")
    for f in BASE_FOODS:
        c = f["canonical"]
        cat = f["category"]
        reg = f["region"]
        u = f["default_unit"]
        q = f["typical_qty"]

        # Run typo engine on canonical English, Gujlish transliteration, and Hinglish transliteration
        target_words = [f["canonical"], f["gu_translit"], f["hi_translit"]]
        for tw in target_words:
            typos = generate_realistic_typos(tw)
            for t in typos:
                add_record(c, t["text"], f"typo_{t['type']}", "English/Roman", cat, reg, "normal", u, q)

    print(f"       -> Count so far: {len(scenarios)}")

    print("[4/10] Building Compound Foods (Staple + Curry combinations in English, Gujarati, Hindi)...")
    compound_templates = [
        # English
        ("{food1} with {food2}", "English"),
        ("{food1} and {food2}", "English"),
        ("had {food1} plus {food2}", "English"),
        ("ate {qty1} {food1} and {qty2} {food2}", "English"),
        ("plate of {food1} with {food2}", "English"),
        # Gujarati Gujlish
        ("{food1} ane {food2}", "Gujlish"),
        ("{food1} sathe {food2}", "Gujlish"),
        ("{qty1} {food1} sathe {qty2} {food2}", "Gujlish"),
        ("aaje {qty1} {food1} ane {food2} lidhu", "Gujlish"),
        ("bapore {qty1} {food1} ane {food2} khadhi", "Gujlish"),
        # Hindi Hinglish
        ("{food1} aur {food2}", "Hinglish"),
        ("{food1} ke sath {food2}", "Hinglish"),
        ("{qty1} {food1} ke saath {qty2} {food2}", "Hinglish"),
        ("aaj maine {qty1} {food1} aur {food2} khaya", "Hinglish"),
        ("dophar me {qty1} {food1} aur {food2} liya", "Hinglish")
    ]

    food_combos = [
        ("Roti", "Dal Tadka", "2", "1 bowl"),
        ("Chapati", "Bhindi Masala", "3", "1 bowl"),
        ("Methi Thepla", "Chaas", "2", "1 glass"),
        ("Bajra Rotlo", "Ringna No Oro", "1", "1 bowl"),
        ("Bhakri", "Sev Tameta Nu Shaak", "2", "1 bowl"),
        ("Khichdi", "Gujarati Kadhi", "1 plate", "1 bowl"),
        ("Jeera Rice", "Dal Fry", "1 plate", "1 bowl"),
        ("Veg Biryani", "Raita", "1 plate", "1 bowl"),
        ("Chole", "Bhatura", "1 bowl", "2"),
        ("Rajma", "Rice", "1 bowl", "1 plate"),
        ("Aloo Paratha", "Curd", "2", "1 bowl"),
        ("Idli", "Sambar", "2", "1 bowl"),
        ("Masala Dosa", "Filter Coffee", "1", "1 cup"),
        ("Medu Vada", "Sambar", "2", "1 bowl"),
        ("Pav Bhaji", "Masala Chai", "1 plate", "1 cup"),
        ("Vada Pav", "Cutting Chai", "1", "1 cup"),
        ("Pani Puri", "Sev Puri", "1 plate", "1 plate"),
        ("Fafda", "Jalebi", "100g", "50g"),
        ("Khaman", "Chaas", "1 plate", "1 glass"),
        ("Paneer Butter Masala", "Butter Naan", "1 bowl", "2"),
        ("Palak Paneer", "Roti", "1 bowl", "3"),
        ("Sarson Ka Saag", "Makki Di Roti", "1 bowl", "2"),
        ("Poha", "Tea", "1 plate", "1 cup")
    ]

    for f1, f2, q1, q2 in food_combos:
        for tmpl, lang in compound_templates:
            phrase = tmpl.format(food1=f1.lower(), food2=f2.lower(), qty1=q1, qty2=q2)
            add_record(f1, phrase, "compound_food_scenario", lang, "composite_meal", "Pan-India", "normal", "serving", f"{q1} + {q2}")

    print(f"       -> Count so far: {len(scenarios)}")

    print("[5/10] Building Regional Specialties (Gujarat, Punjab, South India, Maharashtra, Rajasthan, Bengal, Kashmir)...")
    regional_dishes = [
        # Gujarat & Kathiyawad
        ("Sev Tameta Nu Shaak", "Gujarat", "vegetable_dry", ["સેવ ટમેટાનું શાક", "sev tameta", "sev tameta nu shaak", "kathiyawadi sev tameta", "spicy sev tameta"]),
        ("Undhiyu", "Gujarat", "vegetable_dry", ["ઊંધિયું", "undhiyu", "surti undhiyu", "kathiyawadi undhiyu", "winter undhiyu with muthia"]),
        ("Lasaniya Bataka", "Gujarat", "vegetable_dry", ["લસણીયા બટાકા", "lasaniya bataka", "garlic baby potato kathiyawadi", "spicy lasaniya bateta"]),
        ("Ringna No Oro", "Gujarat", "vegetable_dry", ["રીંગણાનો ઓળો", "ringna no oro", "ringna oro with bajra rotlo", "baingan bharta gujarati"]),
        ("Dal Dhokli", "Gujarat", "composite_meal", ["દાળ ઢોકળી", "dal dhokli", "kathiyawadi dal dhokli", "sweet spicy dal dhokli with ghee"]),
        ("Handvo", "Gujarat", "snack", ["હાંડવો", "handvo", "crispy tawa handvo", "baked vegetable handvo with sesame"]),
        ("Khandvi", "Gujarat", "snack", ["ખાંડવી", "khandvi", "surti khandvi", "tempered rolled khandvi"]),
        ("Fafda Jalebi", "Gujarat", "snack", ["ફાફડા જલેબી", "fafda jalebi", "dussehra special fafda jalebi with sambharo"]),
        ("Sev Khamani", "Gujarat", "snack", ["સેવ ખમણી", "sev khamani", "surti sev khamani with pomegranate"]),
        ("Locho", "Gujarat", "snack", ["લોચો", "surti locho", "butter garlic locho", "cheese locho"]),
        ("Dabeli", "Gujarat", "street_food_chaat", ["કચ્છી દાબેલી", "kacchi dabeli", "cheese dabeli", "butter toasted dabeli"]),
        ("Patra", "Gujarat", "snack", ["પાતરા", "patra", "taro leaves pinwheel patra", "shallow fried crispy patra"]),
        ("Mag Nu Shaak", "Gujarat", "vegetable_dry", ["મગનું શાક", "mag nu shaak", "whole moong sabzi gujarati"]),
        ("Sukhdi", "Gujarat", "sweet_dessert", ["સુખડી", "sukhdi", "mahudha sukhdi", "gor papdi"]),
        ("Mohanthal", "Gujarat", "sweet_dessert", ["મોહનથાળ", "mohanthal", "traditional besan mohanthal with dry fruits"]),

        # Punjab & North India
        ("Sarson Ka Saag", "Punjab", "vegetable_dry", ["सरसों का साग", "sarson ka saag", "sarson da saag with makhan", "punjabi winter saag"]),
        ("Makki Di Roti", "Punjab", "bread_flatbread", ["मक्के की रोटी", "makki di roti", "crisp makki roti with gur"]),
        ("Dal Makhani", "Punjab", "dal_lentil", ["दाल मखनी", "dal makhani", "bukhara style dal makhani", "creamy slow cooked black dal"]),
        ("Amritsari Kulcha", "Punjab", "bread_flatbread", ["अमृतसरी कुलचा", "amritsari kulcha", "chur chur amritsari aloo kulcha"]),
        ("Paneer Tikka", "Punjab", "snack", ["पनीर टिक्का", "paneer tikka", "tandoori charred paneer tikka with mint chutney"]),
        ("Butter Chicken", "Punjab", "curry_gravy", ["बटर चिकन", "butter chicken", "murgh makhani delhi style"]),
        ("Pinni", "Punjab", "sweet_dessert", ["पिन्नी", "punjabi pinni", "wheat dry fruit pinni"]),

        # Maharashtra
        ("Misal Pav", "Maharashtra", "street_food_chaat", ["मिसळ पाव", "misal pav", "kolhapuri tarri misal", "puneri spicy misal pav"]),
        ("Vada Pav", "Maharashtra", "street_food_chaat", ["वडा पाव", "vada pav", "mumbai vada pav with ghati masala", "kadak pav vada"]),
        ("Pav Bhaji", "Maharashtra", "street_food_chaat", ["पाव भाजी", "pav bhaji", "chowpatty style butter pav bhaji", "cheese pav bhaji"]),
        ("Kanda Poha", "Maharashtra", "breakfast", ["कांदा पोहा", "kanda poha", "dadar style onion poha with roasted peanuts"]),
        ("Thalipeeth", "Maharashtra", "bread_flatbread", ["थालीपीठ", "thalipeeth", "multigrain spiced bhajjani thalipeeth with white butter"]),
        ("Sabudana Khichdi", "Maharashtra", "fasting_food", ["साबूदाना खिचड़ी", "sabudana khichdi", "vrat special tapioca pearl khichdi"]),
        ("Sabudana Vada", "Maharashtra", "snack", ["साबूदाना वड़ा", "sabudana vada", "crispy golden sabudana vada with dahi"]),
        ("Puran Poli", "Maharashtra", "sweet_dessert", ["पूरन पोळी", "puran poli", "maharashtrian puran poli with ghee"]),
        ("Sol Kadhi", "Maharashtra", "drink", ["सोल कढी", "sol kadhi", "konkani kokum coconut milk digestive drink"]),
        ("Pithla Bhakri", "Maharashtra", "composite_meal", ["पिठलं भाकरी", "pithla bhakri", "hot besan pithla with jowar bhakri and thecha"]),

        # South India (Tamil Nadu, Kerala, Karnataka, Andhra)
        ("Bisi Bele Bath", "South India", "rice_grain", ["ಬಿಸಿ ಬೇಳೆ ಬಾತ್", "bisi bele bath", "karnataka spicy lentil rice with boondi"]),
        ("Pongal", "South India", "breakfast", ["ವೆಣ್ ಪೊಂಗಲ್", "ven pongal", "ghee pepper cashew pongal"]),
        ("Appam with Stew", "South India", "composite_meal", ["ആപ്പം", "appam with vegetable stew", "kerala lace appam with coconut milk stew"]),
        ("Puttu Kadala Curry", "South India", "composite_meal", ["പുട്ട്", "puttu and kadala curry", "kerala steamed rice puttu with black chickpea curry"]),
        ("Curd Rice", "South India", "rice_grain", ["தயிர் சாதம்", "thayir sadam", "bagala bath", "tempered south indian curd rice"]),
        ("Rasam Rice", "South India", "rice_grain", ["ರಸಂ ಅನ್ನ", "rasam sadam", "hot pepper rasam with steamed rice and papadum"]),
        ("Pesarattu", "South India", "breakfast", ["పెసరట్టు", "pesarattu upma", "andhra whole green gram dosa with ginger chutney"]),
        ("Ragi Mudde", "South India", "bread_flatbread", ["ರಾಗಿ ಮುದ್ದೆ", "ragi mudde with saaru", "karnataka finger millet balls"]),
        ("Akki Roti", "South India", "bread_flatbread", ["ಅಕ್ಕಿ ರೊಟ್ಟಿ", "akki roti with coconut chutney", "karnataka rice flour flatbread"]),

        # Rajasthan
        ("Dal Baati Churma", "Rajasthan", "composite_meal", ["दाल बाटी चूरमा", "dal baati churma", "jaipuri baked baati with panchmel dal and sweet churma"]),
        ("Gatte Ki Sabzi", "Rajasthan", "curry_gravy", ["गट्टे की सब्जी", "gatte ki sabzi", "spiced gram flour roundels in yogurt gravy"]),
        ("Ker Sangri", "Rajasthan", "vegetable_dry", ["केर सांगरी", "ker sangri", "marwari desert bean and berry delicacy"]),
        ("Pyaaz Kachori", "Rajasthan", "snack", ["प्याज़ कचौड़ी", "pyaaz ki kachori", "jodhpur flaky spiced onion kachori"]),
        ("Mirchi Bada", "Rajasthan", "snack", ["मिर्ची बड़ा", "jodhpur mirchi bada", "large green chili potato batter fritter"]),
        ("Ghevar", "Rajasthan", "sweet_dessert", ["घेवर", "malai ghevar", "teej special honeycomb sweet"]),

        # Bengal & Eastern
        ("Luchi Alur Dom", "Bengal", "composite_meal", ["লুচি আলুর দম", "luchi alur dom", "bengali deep fried refined flour puri with spiced dum aloo"]),
        ("Cholar Dal", "Bengal", "dal_lentil", ["ছোলার ডাল", "cholar dal with coconut", "bengali sweet chana dal with fried coconut bits"]),
        ("Mishti Doi", "Bengal", "sweet_dessert", ["মিষ্টি দই", "mishti doi", "kolkata baked sweet caramel yogurt"]),
        ("Rasgulla", "Bengal", "sweet_dessert", ["রসগোল্লা", "rasgulla", "spongy kolkata roshogolla in light sugar syrup"]),
        ("Sandesh", "Bengal", "sweet_dessert", ["সন্দেশ", "sandesh", "fresh chenna nolen gur sandesh"]),
        ("Litti Chokha", "Bihar", "composite_meal", ["लिट्टी चोखा", "litti chokha", "bihari roasted wheat sattu litti with baingan aloo chokha"]),

        # Kashmir & Awadh
        ("Dum Aloo", "Kashmir", "curry_gravy", ["कश्मीरी दम आलू", "kashmiri dum aloo", "fennel ginger spiced baby potatoes"]),
        ("Kahwa", "Kashmir", "drink", ["कश्मीरी कहवा", "kashmiri kahwa", "saffron green tea with almond flakes and cardamom"]),
        ("Rogan Josh", "Kashmir", "curry_gravy", ["रोगन जोश", "rogan josh", "traditional kashmiri aromatic gravy"]),
        ("Lucknowi Biryani", "Awadh", "rice_grain", ["अवधी बिरयानी", "lucknowi dum biryani", "fragrant mild awadhi pulao biryani"])
    ]

    for dish_name, reg, cat, aliases in regional_dishes:
        for al in aliases:
            add_record(dish_name, al, "regional_name_scenario", "Regional Indian", cat, reg, "normal", "serving", "1 serving")
            # With common regional quantity modifiers
            add_record(dish_name, f"1 plate {al}", "regional_name_scenario", "Regional Indian", cat, reg, "normal", "plate", "1 plate")
            add_record(dish_name, f"2 plates {al}", "regional_name_scenario", "Regional Indian", cat, reg, "normal", "plate", "2 plates")

    print(f"       -> Count so far: {len(scenarios)}")

    print("[6/10] Building Snacks, Farsan, Chaat & Street Food Scenarios...")
    snack_items = [
        "Samosa", "Kachori", "Pani Puri", "Sev Puri", "Bhel Puri", "Dahi Puri",
        "Aloo Tikki", "Ragda Pattice", "Pav Bhaji", "Vada Pav", "Dabeli",
        "Khaman", "Dhokla", "Khandvi", "Handvo", "Fafda", "Khakhra", "Patra",
        "Methi Na Gota", "Batata Vada", "Bread Pakora", "Paneer Pakora",
        "Onion Bhajiya", "Mirchi Pakora", "Chakli", "Gathiya", "Sev", "Aloo Bhujia",
        "Poha Chivda", "Mathri", "Namak Para", "Spring Roll", "Veg Momos",
        "Frankie", "Kathi Roll", "Chana Chaat", "Papdi Chaat", "Dahi Vada"
    ]
    snack_modifiers = [
        "", "crispy", "hot", "fresh", "spicy", "extra sev", "with green chutney",
        "with sweet tamarind chutney", "street style", "haldiram style",
        "savare nasto", "evening snack", "tea time"
    ]
    for snk in snack_items:
        for mod in snack_modifiers:
            text = f"{mod} {snk}".strip()
            add_record(snk, text, "snack_farsan_chaat", "English/Roman", "snack", "Pan-India", "normal", "plate", "1 plate")
            # Add with typical portion sizes
            add_record(snk, f"1 plate {text}", "snack_farsan_chaat", "English/Roman", "snack", "Pan-India", "normal", "plate", "1 plate")
            add_record(snk, f"2 {snk.lower()} with chutney", "snack_farsan_chaat", "English/Roman", "snack", "Pan-India", "normal", "piece", "2 pieces")

    print(f"       -> Count so far: {len(scenarios)}")

    print("[7/10] Building Drinks & Traditional Beverages Scenarios...")
    beverage_items = [
        "Masala Chai", "Cutting Chai", "Adrak Wali Chai", "Elaichi Chai", "Lemon Tea", "Green Tea",
        "Filter Coffee", "Cold Coffee", "Chaas", "Masala Chaas", "Jeera Chaas", "Sweet Lassi",
        "Salted Lassi", "Mango Lassi", "Nimbu Pani", "Shikanji", "Jaljeera", "Badam Milk",
        "Sugarcane Juice", "Aam Panna", "Kokum Sharbat", "Sol Kadhi", "Thandai", "Nariyal Pani",
        "Sattu Sharbat", "Rooh Afza Milk"
    ]
    drink_modifiers = [
        "", "chilled", "hot", "without sugar", "sugar free", "extra sweet",
        "with less ice", "without ice", "strong kadak", "refreshing",
        "1 glass", "2 glasses", "1 cup", "2 cups", "1 bottle", "pitcher"
    ]
    for drk in beverage_items:
        for mod in drink_modifiers:
            term = f"{mod} {drk}".strip()
            add_record(drk, term, "drink_beverage", "English/Roman", "drink", "Pan-India", "normal", "glass/cup", "1 glass")

    print(f"       -> Count so far: {len(scenarios)}")

    print("[8/10] Building Preparation Variants & Dietary Modifications...")
    variants_list = [
        ("with ghee", "with pure desi ghee"),
        ("without ghee", "dry sukha without ghee"),
        ("with butter", "topped with amul butter"),
        ("extra butter", "loaded with extra butter"),
        ("roasted", "tawa roasted crispy"),
        ("boiled", "plain water boiled"),
        ("steamed", "freshly steamed oil free"),
        ("deep fried", "golden deep fried in oil"),
        ("shallow fried", "tawa shallow pan fried"),
        ("air fried", "healthy air fried crisp"),
        ("tadka", "vaghareli double jeera garlic tadka"),
        ("dry", "sukhi dry sabzi"),
        ("gravy", "tari rassa gravy style"),
        ("spicy", "extra teekha hot spicy"),
        ("mild", "feeku non-spicy"),
        ("sweet", "meethu jaggery sweet"),
        ("sugar free", "bina cheeni sugar free"),
        ("with jaggery", "sweetened with organic gud"),
        ("jain", "jain preparation no onion no garlic"),
        ("whole wheat", "100% whole wheat gehun"),
        ("multigrain", "multigrain fiber rich")
    ]
    for f in BASE_FOODS[:35]:
        c = f["canonical"]
        cat = f["category"]
        reg = f["region"]
        u = f["default_unit"]
        q = f["typical_qty"]
        for v_code, v_desc in variants_list:
            text1 = f"{c.lower()} {v_code}"
            text2 = f"{v_desc} {c.lower()}"
            add_record(c, text1, "preparation_variant", "English/Roman", cat, reg, v_code, u, q)
            add_record(c, text2, "preparation_variant", "English/Roman", cat, reg, v_code, u, q)

    print(f"       -> Count so far: {len(scenarios)}")

    print("[9/10] Building Conversational Daily Meal Logging Scenarios (Breakfast, Lunch, Dinner in EN, GU, HI)...")
    meals = [
        ("breakfast", "savare", "subah"),
        ("lunch", "bapore", "dophar"),
        ("snack", "sanju / nasto", "shaam"),
        ("dinner", "raatre", "raat")
    ]

    meal_sentences = [
        # English
        ("had {qty} {food} for {meal_en}", "English"),
        ("ate {qty} {food} in {meal_en}", "English"),
        ("logged {qty} {food} for today's {meal_en}", "English"),
        ("{meal_en}: {qty} {food}", "English"),
        ("finished {qty} {food} at {meal_en}", "English"),
        # Gujarati Gujlish
        ("{meal_gu} {qty} {food_gu} khadhi", "Gujlish"),
        ("{meal_gu} ma {qty} {food_gu} lidhu", "Gujlish"),
        ("aaje {meal_gu} ma {qty} {food_gu}", "Gujlish"),
        ("savare nasto: {qty} {food_gu}", "Gujlish"),
        ("raatre jemva ma {qty} {food_gu} lidhu", "Gujlish"),
        # Hindi Hinglish
        ("{meal_hi} me {qty} {food_hi} khaya", "Hinglish"),
        ("maine {meal_hi} me {qty} {food_hi} liya", "Hinglish"),
        ("aaj {meal_hi} me {qty} {food_hi}", "Hinglish"),
        ("{meal_hi} time: {qty} {food_hi} khaya tha", "Hinglish"),
        ("subah nashte me {qty} {food_hi} liya", "Hinglish")
    ]

    for f in BASE_FOODS:
        c = f["canonical"]
        cat = f["category"]
        reg = f["region"]
        u = f["default_unit"]
        q = f["typical_qty"]
        f_en = f["canonical"].lower()
        f_gu = f["gu_translit"].lower()
        f_hi = f["hi_translit"].lower()

        for m_en, m_gu, m_hi in meals:
            for s_tmpl, lang in meal_sentences:
                phrase = s_tmpl.format(
                    qty=q,
                    food=f_en,
                    food_gu=f_gu,
                    food_hi=f_hi,
                    meal_en=m_en,
                    meal_gu=m_gu,
                    meal_hi=m_hi
                )
                add_record(c, phrase, "conversational_meal_logging", lang, cat, reg, "normal", u, q)

    print(f"       -> Count so far: {len(scenarios)}")

    print("[10/10] Building Specific Indian Portions & Unit Variations (plates, katoris, bowls, glasses, grams)...")
    portion_templates = [
        "1 piece {f}", "2 pieces {f}", "3 pieces {f}", "4 pieces {f}", "5 pieces {f}",
        "1 plate {f}", "half plate {f}", "full plate {f}", "2 plates {f}",
        "1 katori {f}", "2 katori {f}", "1 bowl {f}", "2 bowls {f}", "small bowl {f}", "big bowl {f}",
        "1 glass {f}", "half glass {f}", "2 glasses {f}", "1 cup {f}", "2 cups {f}",
        "50g {f}", "100g {f}", "150g {f}", "200g {f}", "250g {f}", "500g {f}",
        "100ml {f}", "200ml {f}", "250ml {f}", "500ml {f}",
        "ek {f}", "be {f}", "tran {f}", "char {f}", "panch {f}",
        "ek katori {f}", "ek vaati {f}", "be vaati {f}", "ek cup {f}", "ek glass {f}"
    ]
    for f in BASE_FOODS:
        c = f["canonical"]
        cat = f["category"]
        reg = f["region"]
        target = f["canonical"].lower()
        target_gu = f["gu_translit"].lower()
        for pt in portion_templates:
            text = pt.format(f=target)
            add_record(c, text, "portion_unit_variation", "English/Roman", cat, reg, "normal", "specified", "custom")
            if "vaati" in pt or "be" in pt or "tran" in pt or "ek" in pt:
                text_gu = pt.format(f=target_gu)
                add_record(c, text_gu, "portion_unit_variation", "Gujlish", cat, reg, "normal", "specified", "custom")

    print(f"==================================================")
    print(f"TOTAL UNIQUE SCENARIOS GENERATED: {len(scenarios)}")
    print(f"==================================================")

    return scenarios


def save_datasets(scenarios: List[Dict[str, Any]]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "real_indian_foods_10000_scenarios.json")
    csv_path = os.path.join(OUTPUT_DIR, "real_indian_foods_10000_scenarios.csv")

    print(f"Saving JSON dataset to: {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)

    print(f"Saving CSV dataset to: {csv_path}")
    fieldnames = [
        "id", "food_name_canonical", "scenario_input", "scenario_type",
        "language", "category", "region", "preparation_variant",
        "portion_unit", "typical_portion", "expected_resolved_food"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenarios)

    print(f"[SUCCESS] Datasets generated successfully! Total records: {len(scenarios)}")


if __name__ == "__main__":
    records = build_10000_scenarios()
    save_datasets(records)
