"""
1,000-Case Comprehensive Semantic Nutrition vs. Workout Intent Routing Regression Benchmark
=============================================================================================
Guarantees Under Test:
1. Nutrition / meal recommendation requests must NEVER route to workout recommendations (RECOMMEND_WORKOUT).
2. Workout recommendation requests must NEVER route to meal recommendations (RECOMMEND_MEAL).
3. Mixed intent queries are resolved by semantic priority:
   - Queries asking what to eat/drink/consume around workout ("post workout meal", "what to eat after gym")
     MUST route to RECOMMEND_MEAL.
   - Queries asking what physical exercise to do to burn calories or compensate ("workout after heavy dinner")
     MUST route to RECOMMEND_WORKOUT.
4. Multilingual inputs (English, Hindi, Gujarati, Roman Hinglish, Roman Gujlish, Devanagari, Gujarati script),
   transliteration, phonetic slips, and typos are fully resilient and robust.

Target: 1,000 / 1,000 PASS (100%). Kept permanently for future regression testing.
"""
import asyncio
import io
import sys
from typing import Optional

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from llm_service import Intent, classify_recommendation_intent
from chatbot_engine import ChatbotEngine
from database import sync_db, async_db


# ============================================================================
# 1,000 TEST CASES (category, message, expected_intent)
# ============================================================================

CASES = []

# ----------------------------------------------------------------------------
# Category 1: Nutrition / Meal Recommendations (350 cases)
# ----------------------------------------------------------------------------
MEALS = ["breakfast", "lunch", "dinner", "snack", "morning snack", "evening snack", "supper", "brunch"]
ADJECTIVES = ["healthy", "high protein", "low calorie", "low carb", "balanced", "quick", "clean", "light", "nutritious", "fiber rich"]

# 1.1 English meal recommendations (120 cases)
for m in ["breakfast", "lunch", "dinner", "snacks"]:
    for adj in ["high protein", "healthy", "low calorie", "low carb", "light"]:
        CASES.append(("nutrition_en", f"suggest a {adj} {m}", Intent.RECOMMEND_MEAL))
        CASES.append(("nutrition_en", f"recommend {adj} {m} options", Intent.RECOMMEND_MEAL))
        CASES.append(("nutrition_en", f"give me some {adj} {m} ideas", Intent.RECOMMEND_MEAL))
        CASES.append(("nutrition_en", f"what should I eat for {m} if I want {adj} food?", Intent.RECOMMEND_MEAL))
        CASES.append(("nutrition_en", f"what to eat for {m} that is {adj}?", Intent.RECOMMEND_MEAL))
        CASES.append(("nutrition_en", f"best {adj} {m} recipes", Intent.RECOMMEND_MEAL))

# 1.2 Hindi & Hinglish meal recommendations (80 cases)
hindi_templates = [
    "dinner me kya khana chahiye", "aaj khane me kya banaye", "subah ka nashta kya hona chahiye",
    "healthy nashta suggest karo", "dopahar ke khane me kya khaye", "raat ka bhojan kaisa hona chahiye",
    "kuch healthy khana suggest karo", "diet plan batao weight loss ke liye", "high protein diet suggest karo",
    "dinner me kya khau batao", "lunch ke liye healthy option bataiye", "weight loss ke liye kya khana chahiye",
    "muscle gain ke liye diet chart batao", "fat loss ke liye meal plan suggest karo", "aaj lunch me kya le",
    "shaam ke naste me kya khaye", "evening snack ke liye healthy options batao", "subah subah kya khana accha hai",
    "raat ko halka khana suggest karo", "batao dinner me kya khaye", "kuch high protein snack batao",
    "veg high protein food suggest karo", "bina oil ka khana suggest karo", "healthy khane ke ideas do",
    "kya khaye batao na", "aaj rat ka khana kya hona chahiye", "batao lunch me kya banau",
    "weight loss diet suggest karo", "kya khana chahiye fat kam karne ke liye", "lunch ke liye acche options batao",
    "subah nashte me kya lu", "protein ke liye kya khana chahiye", "healthy breakfast ideas batao",
    "dinner ke liye low calorie khana suggest karo", "sugar free snacks bataiye", "dopahar me kya khana theek rahega",
    "raat ko kya khaye jisse weight na badhe", "paneer ke alawa protein food suggest karo",
    "dal chawal ke sath kya khaye", "healthy dinner recipes suggest karo",
]
for text in hindi_templates:
    CASES.append(("nutrition_hi", text, Intent.RECOMMEND_MEAL))
    CASES.append(("nutrition_hi", f"mujhe {text}", Intent.RECOMMEND_MEAL))

# 1.3 Gujarati & Gujlish meal recommendations (80 cases)
guj_templates = [
    "bapore shu jamvu?", "dinner ma su banavu?", "savare nasto su karvo?", "su khavu joiye aaje?",
    "ratna bhojan mate su saru rahe?", "vajan ghadadva mate su khavu?", "healthy nasto suchavjo",
    "bapor na jamva ma su levu?", "shak ane rotli sivay su khavu?", "protein valo khorak suchavo",
    "dinner mate halku bhojan suchavo", "savar na nasta mate options aapo", "shu jamvu aaje bapore?",
    "aaje ratre su banavu joiye?", "vajan utarva mate diet plan aapo", "high protein shak suchavo",
    "bapore dal bhaat sivay su khavu?", "healthy snack mate options suchavo", "sanje nasto su karvo?",
    "khorak ma protein vadharva su khavu?", "khichdi sivay ratre su khavu?", "bapor nu bhojan kaisa hona chahiye",
    "mane ek saru diet plan suchavo", "fat loss mate su jamvu?", "muscle vadharva su khavu joiye?",
    "savare nasta ma su lidhu joiye?", "dinner ma su saru?", "bapore su khavu?", "ratre su jamvu?",
    "aavti kale khava ma su banavu?", "healthy lunch suchavjo", "protein nasta suchavo",
    "chaas sathe su khavu?", "dudh sathe su levu?", "sanje bhukh lage to su khavu?",
    "savare uthine pahela su khavu?", "ratna khavanu suchavo", "bapor na jamvanu suchavo",
    "vajan ocharu karva su khavu?", "gym javawala mate khorak suchavo",
]
for text in guj_templates:
    CASES.append(("nutrition_gu", text, Intent.RECOMMEND_MEAL))
    CASES.append(("nutrition_gu", f"mane {text}", Intent.RECOMMEND_MEAL))

# 1.4 Native Indic Script meal recommendations (70 cases)
indic_templates = [
    # Hindi (Devanagari)
    "रात के खाने में क्या खाऊं?", "नाश्ता में क्या खाना चाहिए?", "दोपहर के भोजन के लिए सुझाव दें",
    "स्वस्थ नाश्ता क्या होना चाहिए?", "वजन घटाने के लिए डाइट प्लान बताओ", "उच्च प्रोटीन भोजन का सुझाव दें",
    "शाम के नाश्ते में क्या खाएं?", "रात में हल्का खाना क्या खाएं?", "फैट लॉस के लिए क्या खाना चाहिए?",
    "स्वस्थ भोजन के विकल्प बताएं", "दोपहर के खाने में क्या बनाएं?", "प्रोटीन युक्त नाश्ता बताएं",
    "कम कैलोरी वाला रात का खाना", "डाइट चार्ट का सुझाव दें", "सुबह खाली पेट क्या खाएं?",
    "स्वस्थ आहार योजना बताइए", "शाकाहारी प्रोटीन खाना क्या है?", "बिना तेल का खाना बताएं",
    # Gujarati script
    "બપોરે શું જમવું જોઈએ?", "રાત્રે શું બનાવવું?", "સવારના નાસ્તા માટે સ્વસ્થ વિકલ્પો સૂચવો",
    "વજન ઘટાડવા માટે શું ખાવું?", "પ્રોટીન-યુક્ત આહાર સૂચવો", "સાંજના નાસ્તામાં શું લેવું?",
    "રાત્રિ ભોજન માટે હલકો ખોરાક સૂચવો", "સ્વસ્થ નાસ્તો શું હોવો જોઈએ?", "ડાયેટ પ્લાન સૂચવજો",
    "બપોરના ભોજનમાં શું ખાવું?", "ચરબી ઘટાડવા શું ખાવું જોઈએ?", "પૌષ્ટિક આહારના વિકલ્પો આપો",
    "સ્વસ્થ ભોજન યોજના જણાવો", "શાકાહારી પ્રોટીન ખોરાક સૂચવો", "સવારે ભૂખ્યા પેટે શું ખાવું?",
    "રાત્રે શું જમવું સારું રહેશે?", "બપોરે શું જમવું?",
]
for text in indic_templates:
    CASES.append(("nutrition_indic", text, Intent.RECOMMEND_MEAL))
    prefix = "મને " if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in text) else "कृपया "
    CASES.append(("nutrition_indic", f"{prefix}{text}", Intent.RECOMMEND_MEAL))
assert sum(1 for c in CASES if c[0].startswith("nutrition")) == 350, f"Cat 1 should be 350, got {sum(1 for c in CASES if c[0].startswith('nutrition'))}"

# ----------------------------------------------------------------------------
# Category 2: Workout Recommendations (350 cases)
# ----------------------------------------------------------------------------
SPLITS = ["chest", "back", "legs", "shoulders", "biceps", "triceps", "abs", "core", "arms", "full body"]
GOALS = ["fat loss", "muscle gain", "weight loss", "strength", "endurance", "flexibility", "stamina"]

# 2.1 English workout recommendations (120 cases)
for sp in ["chest", "back", "legs", "arms"]:
    for g in ["fat loss", "muscle gain", "strength", "endurance", "toning"]:
        CASES.append(("workout_en", f"suggest a {sp} workout for {g}", Intent.RECOMMEND_WORKOUT))
        CASES.append(("workout_en", f"recommend a {sp} routine for {g}", Intent.RECOMMEND_WORKOUT))
        CASES.append(("workout_en", f"what {sp} exercises should I do for {g}?", Intent.RECOMMEND_WORKOUT))
        CASES.append(("workout_en", f"give me a {sp} training plan for {g}", Intent.RECOMMEND_WORKOUT))
        CASES.append(("workout_en", f"best {sp} workout for {g}", Intent.RECOMMEND_WORKOUT))
        CASES.append(("workout_en", f"what workout should I do tomorrow for {sp}?", Intent.RECOMMEND_WORKOUT))

# 2.2 Hindi & Hinglish workout recommendations (80 cases)
hindi_work_templates = [
    "kal gym me kya karein", "workout plan suggest karo", "kal kaunsi exercise karni chahiye",
    "chest workout batao", "leg day routine suggest karo", "weight loss ke liye konsi exercise karein",
    "muscle build karne ke liye workout routine batao", "kal kya exercise karein batao",
    "arms workout plan bataiye", "biceps badhane ke liye kya exercise karein", "full body workout suggest karo",
    "home workout plan batao", "bina equipment ke konsi exercise karein", "kal gym me kya karna chahiye",
    "fat loss ke liye best exercise batao", "pet ki charbi kam karne ke liye exercise batao",
    "shoulder workout routine bataiye", "back workout me kya karein", "triceps ke liye exercise batao",
    "daily workout routine kaisa hona chahiye", "aaj konsa workout karein", "workout suggest karo please",
    "gym routine batao bhai", "dand baithak ka schedule batao", "running schedule suggest karo",
    "kal kya exercise kare", "chest aur triceps ka workout batao", "back aur biceps routine batao",
    "cardio plan suggest karo", "hiit workout plan bataiye", "beginner workout routine batao",
    "gym me pehle kya kare", "stamina badhane ke liye exercise batao", "stretching routine suggest karo",
    "kal ka workout plan batao", "subah konsi exercise karein", "gym schedule suggest kijiye",
    "kasrat plan batao", "konsi kasrat karni chahiye", "dumbbell workout suggest karo",
]
for text in hindi_work_templates:
    CASES.append(("workout_hi", text, Intent.RECOMMEND_WORKOUT))
    CASES.append(("workout_hi", f"mujhe {text}", Intent.RECOMMEND_WORKOUT))

# 2.3 Gujarati & Gujlish workout recommendations (80 cases)
guj_work_templates = [
    "kale gym ma su karvu?", "aavti kale exercise suchavo", "kasrat plan suchavjo",
    "vajan utarva mate kai kasrat karvi?", "chest workout suchavo", "leg day mate su karvu?",
    "biceps vadharva mate kai exercise karvi?", "mane workout plan aapo", "kale su exercise karvi joiye?",
    "ghar par thai sake evi kasrat suchavo", "muscle vadharva workout plan", "pet ni charbi ghatadva kasrat",
    "shoulder exercise suchavo", "back workout mate guidance aapo", "aavti kal no exercise plan",
    "daily kasrat ma su karvu?", "full body exercise suchavjo", "running mate schedule aapo",
    "gym ma aaje su karvu?", "stamina vadharva kai kasrat sarie?", "kale gym ma kai exercise karvi?",
    "mane workout suggest karo", "exercise routine suchavjo", "kasarat plan aapo",
    "dosh ane dand schedule suchavo", "triceps mate kasrat suchavo", "cardio workout suchavo",
    "vajan ghatadva mate roje su kasrat karvi?", "hiit exercise suchavo", "beginner mate workout plan",
    "dumble sathe kai kasrat karvi?", "savare kai exercise karvi sarie?", "kale su karvu gym ma?",
    "kasrat suchavjo please", "aavti kal gym schedule", "body banavva mate kasrat suchavo",
    "core exercise routine suchavo", "abs mate kai kasrat karvi?", "stretching routine aapo",
    "gym nu schedule suchavo",
]
for text in guj_work_templates:
    CASES.append(("workout_gu", text, Intent.RECOMMEND_WORKOUT))
    CASES.append(("workout_gu", f"mane {text}", Intent.RECOMMEND_WORKOUT))

# 2.4 Native Indic Script workout recommendations (70 cases)
indic_work_templates = [
    # Hindi (Devanagari)
    "कल कौन सी कसरत करें?", "वर्कआउट प्लान बताओ", "जिम में कल क्या करना चाहिए?",
    "छाती (चेस्ट) के लिए व्यायाम बताएं", "वजन घटाने के लिए कौन सी कसरत करें?", "मांसपेशियों के निर्माण के लिए वर्कआउट बताएं",
    "लेग डे रूटीन का सुझाव दें", "पेट की चर्बी कम करने के लिए व्यायाम", "फुल बॉडी वर्कआउट प्लान बताएं",
    "घर पर करने योग्य कसरत बताएं", "बाइसेप्स के लिए बेहतरीन व्यायाम", "दैनिक वर्कआउट रूटीन सुझाइए",
    "कार्डियो एक्सरसाइज का सुझाव दें", "स्टैमिना बढ़ाने के लिए कौन सा वर्कआउट करें?", "पीठ (बैक) के लिए कसरत बताएं",
    "शुरुआती लोगों के लिए वर्कआउट प्लान", "डंबल के साथ व्यायाम बताएं", "स्ट्रेचिंग रूटीन बताइए",
    # Gujarati script
    "કાલે શું કસરત કરવી?", "વર્કઆઉટ પ્લાન સૂચવો", "જીમમાં કાલે શું કરવું જોઈએ?",
    "છાતી (ચેસ્ટ) માટે કસરત જણાવો", "વજન ઘટાડવા કઈ કસરત કરવી?", "સ્નાયુ વધારવા વર્કઆઉટ પ્લાન આપો",
    "પગ (લેગ્સ) માટે કસરતો સૂચવો", "પેટની ચરબી ઘટાડવા કસરત જણાવો", "આખા શરીરની કસરત પ્લાન સૂચવો",
    "ઘરે કરી શકાય તેવી કસરત બતાવો", "બાઈસેપ્સ વધારવા કઈ કસરત કરવી?", "દૈનિક કસરત રૂટિન આપો",
    "કાર્ડિયો કસરતો સૂચવો", "સ્ટેમિના વધારવા કઈ કસરત કરવી?", "પીઠ (બેક) માટે કસરત જણાવો",
    "શરૂઆત કરનારા માટે કસરત યોજના", "ડંબેલ સાથે કસરત સૂચવો",
]
for text in indic_work_templates:
    CASES.append(("workout_indic", text, Intent.RECOMMEND_WORKOUT))
    prefix = "મને " if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in text) else "कृपया "
    CASES.append(("workout_indic", f"{prefix}{text}", Intent.RECOMMEND_WORKOUT))
assert sum(1 for c in CASES if c[0].startswith("workout")) == 350, f"Cat 2 should be 350, got {sum(1 for c in CASES if c[0].startswith('workout'))}"

# ----------------------------------------------------------------------------
# Category 3: Mixed Intent Disambiguation (150 cases)
# ----------------------------------------------------------------------------
# 3.1 Nutrition around Workout -> MUST BE RECOMMEND_MEAL (100 cases)
mixed_nut_templates = [
    "post workout meal suggest karo", "what should I eat after workout?", "pre workout snack recommendation",
    "what to eat after gym", "gym ke baad kya khana chahiye", "workout pachi su khavu?",
    "gym thi aavine su pivu?", "post workout protein shake suggestion", "pre gym snack ideas",
    "what to drink during workout", "workout baad kya khana chahiye", "gym ke pehle kya khaye",
    "post workout diet plan", "pre workout me kya khana theek hai", "what can I eat before my workout?",
    "suggest a good post-workout snack", "high protein food after workout", "gym pachi ketlu khavu?",
    "workout pachi dudh pi shakay?", "best food to eat after heavy workout", "pre-workout energy food",
    "post workout breakfast options", "dinner after evening workout", "what to eat after morning run",
    "post exercise recovery food", "post training meal ideas", "suggest pre workout meal",
    "gym baad ande khaye ya paneer", "post workout carbs and protein suggestions", "what should I drink after gym",
    "pre workout banana aur coffee theek hai?", "workout pachi protein powder sivay su khavu?",
    "gym ke baad protein rich food batao", "after workout protein shake recipe", "healthy post workout snacks",
    "pre workout light food ideas", "what to eat 30 mins before workout", "post workout meal under 400 calories",
    "workout pachi su jamvanu?", "gym ke turant baad kya khaye", "post-workout dinner recommendations",
    "pre-workout breakfast ideas", "what is the best meal after running", "food to eat after cycling",
    "post workout snacks for fat loss", "pre gym coffee and snack suggest karo", "workout pachi khorak suchavo",
    "gym baad kya peena chahiye", "post workout smoothie recipe suggest karo", "pre workout oats recipe",
]
for text in mixed_nut_templates:
    CASES.append(("mixed_nutrition", text, Intent.RECOMMEND_MEAL))
    CASES.append(("mixed_nutrition", f"can you {text}" if "what" not in text else f"please tell me {text}", Intent.RECOMMEND_MEAL))
assert sum(1 for c in CASES if c[0] == "mixed_nutrition") == 100, f"Mixed nutrition should be 100, got {sum(1 for c in CASES if c[0] == 'mixed_nutrition')}"

# 3.2 Workout to burn food / after eating -> MUST BE RECOMMEND_WORKOUT (50 cases)
mixed_work_templates = [
    "workout after heavy dinner", "exercise to burn pizza calories", "what workout should I do after overeating?",
    "bahut jyada khana kha liya ab konsi exercise karu?", "heavy lunch pachi kai kasrat karvi?",
    "how to burn dinner calories with exercise", "which workout after eating cake", "exercise plan after cheat meal",
    "cheat day ke baad konsa workout karein", "heavy food khane ke baad konsi kasrat karein",
    "pizza khane ke baad kitni der running karu?", "overeating ho gayi ab konsa workout plan follow karein",
    "calories burn karne ke liye exercise suggest karo", "heavy dinner pachi exercise suchavjo",
    "biryani khane ke baad konsi exercise karein", "how to burn burger calories through workout",
    "post cheat meal workout routine", "sweet khane ke baad konsi kasrat karvi?", "workout to burn fat after eating",
    "exercise to burn 500 food calories", "aaj zyada calories kha li konsa workout karein",
    "heavy lunch ke baad walk ya exercise kya karein?", "what cardio to burn cheat meal calories",
    "heavy khana pachi vajan na vadhe te mate kai kasrat karvi?", "workout routine to burn fast food",
]
for text in mixed_work_templates:
    CASES.append(("mixed_workout", text, Intent.RECOMMEND_WORKOUT))
    CASES.append(("mixed_workout", f"please suggest {text}", Intent.RECOMMEND_WORKOUT))
assert sum(1 for c in CASES if c[0] == "mixed_workout") == 50, f"Mixed workout should be 50, got {sum(1 for c in CASES if c[0] == 'mixed_workout')}"

# ----------------------------------------------------------------------------
# Category 4: Multilingual & Typo Stress Cases (150 cases)
# ----------------------------------------------------------------------------
# 4.1 Nutrition Typos & Transliterations (75 cases)
nut_typo_templates = [
    "sugest a helthy brakfast", "recomended meel for dinr", "wat shud i eet for lunsh",
    "high protien meel plan", "sugest som helthy snaks", "diet plan for fat los",
    "dinner me kya khaye btao", "su jmvu aaje bapore", "ratre su bnavu joiye",
    "btaao kya khaye dopahar me", "helthy nasta suchevjo", "protien valo khorak suchev",
    "sugest a low calori dinner", "recommand a good brekfast", "wat can i eat for snak",
    "proten rich food sugest karo", "recomnded diet plan", "brekfast ma su khavu",
    "lunsh options for tomorrow", "dinr ideas under 500 cals", "post wrkout meel suggest kro",
    "pre wrkout snak idea", "gym ke bad kya khaye", "wrkout pachi su khavu",
    "sugjest healthy snak", "bapore su jamvu", "shu jmvu aaje", "ratre su khvu",
    "btaiye kya khana chahie", "vajan ghadadva su khvu", "fat los diet btao",
    "recomended post workout shake", "sugest evening snak", "helthy lunch ideas",
    "high protin nasta", "dinr ma su bnavu", "breakfast me kya khau",
]
for text in nut_typo_templates:
    CASES.append(("typo_nutrition", text, Intent.RECOMMEND_MEAL))
    if sum(1 for c in CASES if c[0] == "typo_nutrition") < 75:
        CASES.append(("typo_nutrition", f"pls {text}", Intent.RECOMMEND_MEAL))
while sum(1 for c in CASES if c[0] == "typo_nutrition") < 75:
    CASES.append(("typo_nutrition", f"sugest healthy meel option {len(CASES)}", Intent.RECOMMEND_MEAL))
assert sum(1 for c in CASES if c[0] == "typo_nutrition") == 75, f"Typo nutrition should be 75, got {sum(1 for c in CASES if c[0] == 'typo_nutrition')}"

# 4.2 Workout Typos & Transliterations (75 cases)
work_typo_templates = [
    "sugest a wrkout", "wat exersise tomorrow", "recomended gym rotine",
    "kal konsi exrsise kare", "wrkout plan suggest kro", "recommand chest exersice",
    "leg day rotin suggest kro", "exersise for tomorrow", "sugest a traing plan",
    "kale gym ma su karvu", "aavti kale exersise suchvo", "kasrat plan suchav",
    "sugest cardio wrkout", "wat wrkout should i do", "biceps exersice btao",
    "sholder wrkout plan", "back exersice suggest kro", "home wrkout without equipmnt",
    "recomended ful body wrkout", "kal kya kasrat kre", "gym ma su karvu kale",
    "sugjest gym rotine", "wrkout to burn cals", "exrcise plan for weight los",
    "fat los wrkout suggest kro", "abs exersice btao", "hiit wrkout plan",
    "sugest daily exersice", "traing routine for tomorrow", "wrkout plan for beginer",
    "recomnd workout", "konsi exersise kre kal", "su exersise karvi kale",
    "kale kasrat ma su karvu", "chest exersise suchvo", "leg wrkout plan",
    "stretching rotine suggest kro",
]
for text in work_typo_templates:
    CASES.append(("typo_workout", text, Intent.RECOMMEND_WORKOUT))
    if sum(1 for c in CASES if c[0] == "typo_workout") < 75:
        CASES.append(("typo_workout", f"pls {text}", Intent.RECOMMEND_WORKOUT))
while sum(1 for c in CASES if c[0] == "typo_workout") < 75:
    CASES.append(("typo_workout", f"sugest daily wrkout plan {len(CASES)}", Intent.RECOMMEND_WORKOUT))
assert sum(1 for c in CASES if c[0] == "typo_workout") == 75, f"Typo workout should be 75, got {sum(1 for c in CASES if c[0] == 'typo_workout')}"

# Trim or adjust to exactly 1,000 cases
CASES = CASES[:1000]
assert len(CASES) == 1000, f"Expected 1,000 cases, got {len(CASES)}"


# ============================================================================
# RUNNER & VERIFICATION
# ============================================================================

async def run_benchmark():
    print("=" * 80)
    print("1,000-CASE SEMANTIC NUTRITION VS WORKOUT INTENT ROUTING BENCHMARK")
    print("=" * 80)
    print(f"Total Cases Loaded: {len(CASES)}")

    # Breakdown stats
    cat_counts = {}
    for cat, _, exp in CASES:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  • {cat:<20}: {count} cases")
    print("-" * 80)

    # Initialize Engine with sync_db / async_db
    from llm_service import LLMService
    llm = LLMService()
    engine = ChatbotEngine(
        db=async_db,
        user={"user_id": "test_routing_user", "name": "Benchmark Tester", "goal": "maintain"},
        llm=llm,
    )

    passes = 0
    fails = 0
    failures = []

    # Strict isolation metrics
    nutrition_routed_to_workout = 0
    workout_routed_to_nutrition = 0

    for idx, (cat, message, expected_intent) in enumerate(CASES, 1):
        # 1. Test semantic intent classifier directly
        direct_intent = classify_recommendation_intent(message)

        # 2. Test engine deterministic recommendation detection
        looks_meal = engine._looks_like_meal_recommendation(message)
        looks_work = engine._looks_like_workout_recommendation(message)

        passed = True
        reason = []

        if expected_intent == Intent.RECOMMEND_MEAL:
            # Must classify as RECOMMEND_MEAL
            if direct_intent != Intent.RECOMMEND_MEAL:
                passed = False
                reason.append(f"direct_classifier got '{direct_intent}', expected '{Intent.RECOMMEND_MEAL}'")
            # Must NEVER classify as workout
            if direct_intent == Intent.RECOMMEND_WORKOUT or looks_work:
                nutrition_routed_to_workout += 1
                passed = False
                reason.append("CRITICAL: Nutrition recommendation leaked into RECOMMEND_WORKOUT!")
            if not looks_meal:
                passed = False
                reason.append("engine._looks_like_meal_recommendation returned False")

        elif expected_intent == Intent.RECOMMEND_WORKOUT:
            # Must classify as RECOMMEND_WORKOUT
            if direct_intent != Intent.RECOMMEND_WORKOUT:
                passed = False
                reason.append(f"direct_classifier got '{direct_intent}', expected '{Intent.RECOMMEND_WORKOUT}'")
            # Must NEVER classify as meal
            if direct_intent == Intent.RECOMMEND_MEAL or looks_meal:
                workout_routed_to_nutrition += 1
                passed = False
                reason.append("CRITICAL: Workout recommendation leaked into RECOMMEND_MEAL!")
            if not looks_work:
                passed = False
                reason.append("engine._looks_like_workout_recommendation returned False")

        if passed:
            passes += 1
        else:
            fails += 1
            failures.append((idx, cat, message, expected_intent, direct_intent, "; ".join(reason)))

    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION RESULTS")
    print("=" * 80)
    print(f"Total Cases Evaluated: {len(CASES)}")
    print(f"PASSED:                {passes} / {len(CASES)} ({passes / len(CASES) * 100:.2f}%)")
    print(f"FAILED:                {fails} / {len(CASES)}")
    print(f"Nutrition → Workout:   {nutrition_routed_to_workout} (MUST BE 0)")
    print(f"Workout → Nutrition:   {workout_routed_to_nutrition} (MUST BE 0)")
    print("=" * 80)

    if failures:
        print("\nALL FAILURES BREAKDOWN BY CATEGORY:")
        fail_cats = {}
        for idx, cat, msg, exp, got, err in failures:
            fail_cats[cat] = fail_cats.get(cat, []) + [(idx, msg, exp, got, err)]
        for cat, items in fail_cats.items():
            print(f"\n--- {cat} ({len(items)} fails) ---")
            for idx, msg, exp, got, err in items[:10]:
                print(f"  #{idx} '{msg}' | exp: {exp} | got: {got} | {err}")
        print("-" * 80)
        raise SystemExit(1)
    else:
        print("\n[SUCCESS] PERFECT SCORE: 1,000 / 1,000 PASSED (100.0%)!")
        print("GUARANTEE VERIFIED: Nutrition/meal requests NEVER route to workout recommendations.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
