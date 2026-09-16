"""
A-to-Z Comprehensive Regression Benchmark Suite
-----------------------------------------------
Covers:
1. Daily Log Read vs Food Logging Isolation & Interruption
2. Compound Food Resolution (e.g. "2 dudhi thepla" -> Dudhi Thepla, never Kheer)
3. Generic Nutrition & Plausibility Validation (no hardcoded food hacks)
4. Piece Scaling (1 Methi Thepla ~149 kcal, 2 ~298 kcal; 1 Dudhi Thepla ~150 kcal, 2 ~300 kcal)
5. All Supported Units & Densities (g, ml, piece, serving, bowl, plate, cup, glass, slice, tsp, tbsp)
6. Preparation Variants & Add-ons (ghee, butter, oil, sugar, without sugar, fried, steamed)
7. Exercise MET Calculations across user body weights (50kg, 70kg, 90kg)
8. Multi-food Handling & Specificity Clarification (e.g. "rotlo and shak")
"""
import asyncio
import io
import sys
from datetime import date

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from database import sync_db, async_db
from food_repository import FoodRepositorySync
from search_engine import FoodSearchEngine
from quantity_parser import parse_quantity, ParsedQuantity
from nutrition_calculator import NutritionCalculator
from nutrition_validator import GenericNutritionValidator
from exercise_calculator import ExerciseCalculator, parse_exercise_input
from chatbot_engine import ChatbotEngine
from llm_service import LLMService, Intent
from conversation import PendingFlow, FlowState

# Test results tracker
passes = 0
fails = 0
failures = []

def test_assert(condition: bool, test_name: str, detail: str = ""):
    global passes, fails
    if condition:
        passes += 1
        print(f"  [PASS] {test_name}")
    else:
        fails += 1
        print(f"  [FAIL] {test_name} -- {detail}")
        failures.append((test_name, detail))

async def run_benchmarks():
    print("=" * 70)
    print("A-TO-Z COMPREHENSIVE REGRESSION BENCHMARK")
    print("=" * 70)

    repo = FoodRepositorySync(sync_db)
    search = FoodSearchEngine(repo)
    calc = NutritionCalculator()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. COMPOUND FOOD RESOLUTION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 1. Compound Food Resolution ---")
    dudhi_res = search.search("dudhi thepla")
    test_assert(dudhi_res.food is not None, "Resolve 'dudhi thepla'")
    test_assert(
        dudhi_res.food and "thepla" in dudhi_res.food.get("food_name", "").lower(),
        "Resolved 'dudhi thepla' is a thepla (not kheer)",
        f"got: {dudhi_res.food.get('food_name') if dudhi_res.food else None}"
    )

    methi_res = search.search("methi thepla")
    test_assert(
        methi_res.food and "thepla" in methi_res.food.get("food_name", "").lower(),
        "Resolve 'methi thepla' to Methi Thepla",
        f"got: {methi_res.food.get('food_name') if methi_res.food else None}"
    )

    rotlo_res = search.search("bajra no rotlo")
    test_assert(
        rotlo_res.food and any(w in rotlo_res.food.get("food_name", "").lower() for w in ("rotlo", "rotla")),
        "Resolve 'bajra no rotlo' to Rotlo",
        f"got: {rotlo_res.food.get('food_name') if rotlo_res.food else None}"
    )

    paratha_res = search.search("aloo paratha")
    test_assert(
        paratha_res.food and "paratha" in paratha_res.food.get("food_name", "").lower(),
        "Resolve 'aloo paratha' to Paratha",
        f"got: {paratha_res.food.get('food_name') if paratha_res.food else None}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2. GENERIC NUTRITION & PIECE SCALING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 2. Generic Nutrition & Piece Scaling ---")

    # Methi Thepla: 1 piece ~145-149 kcal, 2 pieces ~290-298 kcal
    food_methi = repo.get_by_exact_name("methi thepla") or methi_res.food
    if food_methi:
        q_1 = ParsedQuantity(amount=1.0, unit="piece", raw_input="1 piece", confidence=1.0)
        n_1 = calc.calculate(food_methi, q_1)
        test_assert(
            140.0 <= n_1.calories <= 155.0,
            "1 Methi Thepla is approximately 145-149 kcal",
            f"got: {n_1.calories} kcal (grams: {n_1.quantity_grams}g)"
        )

        q_2 = ParsedQuantity(amount=2.0, unit="piece", raw_input="2 pieces", confidence=1.0)
        n_2 = calc.calculate(food_methi, q_2)
        test_assert(
            280.0 <= n_2.calories <= 310.0,
            "2 Methi Thepla is approximately 290-298 kcal",
            f"got: {n_2.calories} kcal (grams: {n_2.quantity_grams}g)"
        )
        test_assert(
            abs(n_2.calories - 2 * n_1.calories) < 1.0,
            "Methi Thepla scales 2x proportionally",
            f"1x: {n_1.calories}, 2x: {n_2.calories}"
        )

    # Dudhi Thepla: 1 piece ~150 kcal, 2 pieces ~300 kcal
    food_dudhi = repo.get_by_exact_name("dudhi thepla") or dudhi_res.food
    if food_dudhi:
        q_1d = ParsedQuantity(amount=1.0, unit="piece", raw_input="1 piece", confidence=1.0)
        n_1d = calc.calculate(food_dudhi, q_1d)
        test_assert(
            145.0 <= n_1d.calories <= 155.0,
            "1 Dudhi Thepla is approximately 150 kcal",
            f"got: {n_1d.calories} kcal (grams: {n_1d.quantity_grams}g)"
        )

        q_2d = ParsedQuantity(amount=2.0, unit="piece", raw_input="2 pieces", confidence=1.0)
        n_2d = calc.calculate(food_dudhi, q_2d)
        test_assert(
            290.0 <= n_2d.calories <= 310.0,
            "2 Dudhi Thepla is approximately 300 kcal",
            f"got: {n_2d.calories} kcal (grams: {n_2d.quantity_grams}g)"
        )
        test_assert(
            abs(n_2d.calories - 2 * n_1d.calories) < 1.0,
            "Dudhi Thepla scales 2x proportionally",
            f"1x: {n_1d.calories}, 2x: {n_2d.calories}"
        )

    # Plausibility checks via GenericNutritionValidator
    test_assert(
        GenericNutritionValidator.is_nutritionally_plausible("methi thepla", 372.0, "Bread"),
        "Generic validator accepts Methi Thepla energy density (~372 kcal/100g)"
    )
    test_assert(
        GenericNutritionValidator.is_nutritionally_plausible("dudhi thepla", 375.0, "Bread"),
        "Generic validator accepts Dudhi Thepla energy density (~375 kcal/100g)"
    )
    test_assert(
        not GenericNutritionValidator.is_nutritionally_plausible("cucumber", 450.0, "Vegetables"),
        "Generic validator rejects impossible calorie density (cucumber 450 kcal/100g)"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. ALL SUPPORTED UNITS & CONTAINER DENSITIES
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 3. Supported Units & Containers ---")
    # Grams & ml direct weight
    test_food = {
        "food_id": "test_generic",
        "food_name": "rice",
        "food_name_display": "Cooked Rice",
        "category": "Cereal Grains and Products",
        "serving_size_g": 100.0,
        "calories_per_100g": 130.0,
        "calories_kcal": 130.0,
        "protein_g": 2.7,
        "carbs_g": 28.0,
        "fat_g": 0.3,
        "fiber_g": 0.4,
    }
    # g
    n_g = calc.calculate(test_food, ParsedQuantity(200.0, "g", "200g", 1.0))
    test_assert(n_g.quantity_grams == 200.0 and n_g.calories == 260.0, "200g of rice = 260 kcal")

    # ml
    n_ml = calc.calculate(test_food, ParsedQuantity(150.0, "ml", "150ml", 1.0))
    test_assert(n_ml.quantity_grams == 150.0 and n_ml.calories == 195.0, "150ml = 195 kcal")

    # bowl (configured for rice as 180g)
    n_bowl = calc.calculate(test_food, ParsedQuantity(1.0, "bowl", "1 bowl", 1.0))
    test_assert(n_bowl.quantity_grams == 180.0, "1 bowl of cooked rice = 180g", f"got: {n_bowl.quantity_grams}g")

    # plate (configured for rice as 300g)
    n_plate = calc.calculate(test_food, ParsedQuantity(1.0, "plate", "1 plate", 1.0))
    test_assert(n_plate.quantity_grams == 300.0, "1 plate of cooked rice = 300g", f"got: {n_plate.quantity_grams}g")

    # tsp (5g) and tbsp (15g)
    test_oil = {
        "food_id": "test_oil",
        "food_name": "ghee",
        "food_name_display": "Ghee",
        "category": "Fats and Oils",
        "serving_size_g": 10.0,
        "calories_per_100g": 900.0,
        "calories_kcal": 90.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 10.0,
        "fiber_g": 0.0,
    }
    n_tsp = calc.calculate(test_oil, ParsedQuantity(1.0, "teaspoon", "1 tsp", 1.0))
    test_assert(n_tsp.quantity_grams == 5.0 and n_tsp.calories == 45.0, "1 tsp ghee = 5g = 45 kcal")

    n_tbsp = calc.calculate(test_oil, ParsedQuantity(1.0, "tablespoon", "1 tbsp", 1.0))
    test_assert(n_tbsp.quantity_grams == 15.0 and n_tbsp.calories == 135.0, "1 tbsp ghee = 15g = 135 kcal")

    # slice
    test_bread = {
        "food_id": "test_bread",
        "food_name": "bread",
        "food_name_display": "White Bread",
        "category": "Bread",
        "serving_size_g": 30.0,
        "calories_per_100g": 265.0,
        "calories_kcal": 79.5,
        "protein_g": 2.7,
        "carbs_g": 15.0,
        "fat_g": 1.0,
        "fiber_g": 0.8,
    }
    n_slice = calc.calculate(test_bread, ParsedQuantity(2.0, "slice", "2 slices", 1.0))
    test_assert(n_slice.quantity_grams == 60.0 and abs(n_slice.calories - 159.0) < 1.0, "2 slices bread = 60g = ~159 kcal")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. PREPARATION VARIANTS & ADD-ONS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 4. Variants & Add-ons ---")
    base_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="normal")
    ghee_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="ghee")
    butter_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="butter")
    oil_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="oil")
    sugar_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="sugar")
    nosugar_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="without sugar")
    fried_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="fried")
    steamed_calc = calc.calculate(test_food, ParsedQuantity(100.0, "g", "100g", 1.0), variant="steamed")

    test_assert(ghee_calc.calories == round(base_calc.calories * 1.25, 1), "Ghee variant adds +25% calories")
    test_assert(butter_calc.calories == round(base_calc.calories * 1.20, 1), "Butter variant adds +20% calories")
    test_assert(oil_calc.calories == round(base_calc.calories * 1.20, 1), "Oil variant adds +20% calories")
    test_assert(sugar_calc.calories == round(base_calc.calories * 1.20, 1), "Sugar variant adds +20% calories")
    test_assert(nosugar_calc.calories == round(base_calc.calories * 0.70, 1), "Without sugar reduces cals by 30%")
    test_assert(fried_calc.calories == round(base_calc.calories * 1.35, 1), "Fried variant adds +35% calories")
    test_assert(steamed_calc.calories == round(base_calc.calories * 0.85, 1), "Steamed variant reduces calories (-15%)")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. EXERCISE CALCULATIONS ACROSS BODY WEIGHTS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 5. Exercise Across Body Weights (50kg, 70kg, 90kg) ---")
    from exercise_calculator import ExerciseSearcher
    ex_calc = ExerciseCalculator()
    ex_inp = parse_exercise_input("running for 30 minutes")
    test_assert(ex_inp is not None, "Parse 'running for 30 minutes'")

    if ex_inp:
        ex_searcher = ExerciseSearcher(sync_db)
        ex_doc, match_type = ex_searcher.search(ex_inp.exercise_query)
        test_assert(ex_doc is not None, "Search exercise 'running'")
        if ex_doc:
            res_50 = ex_calc.calculate(ex_doc, amount=ex_inp.amount, unit=ex_inp.unit, weight_kg=50.0)
            res_70 = ex_calc.calculate(ex_doc, amount=ex_inp.amount, unit=ex_inp.unit, weight_kg=70.0)
            res_90 = ex_calc.calculate(ex_doc, amount=ex_inp.amount, unit=ex_inp.unit, weight_kg=90.0)

            test_assert(res_50 is not None and res_70 is not None and res_90 is not None, "Calculate running across weights")
            if res_50 and res_70 and res_90:
                test_assert(res_50.calories_avg < res_70.calories_avg < res_90.calories_avg,
                            "Calories burned scales strictly with body weight (50kg < 70kg < 90kg)",
                            f"50kg: {res_50.calories_avg}, 70kg: {res_70.calories_avg}, 90kg: {res_90.calories_avg}")

                ratio_cals = res_90.calories_avg / res_50.calories_avg
                test_assert(abs(ratio_cals - 1.8) < 0.1, "Exercise kcal scales linearly with user weight", f"ratio: {ratio_cals:.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. DAILY LOG READ ROUTING & INTERRUPTION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 6. Daily Log Read vs Food Logging Isolation ---")
    user = {"user_id": "test_atoz_bench", "name": "BenchUser", "age": 25, "gender": "male", "height_cm": 170, "weight_kg": 70, "activity_level": "moderate", "fitness_goal": "maintain"}
    bot = ChatbotEngine(async_db, user, LLMService())

    # Test daily log reads without pending flow
    await bot.flow_mgr.save_flow_state(PendingFlow())
    for query, expected_intent in [
        ("aaj nu total log", Intent.GET_SUMMARY),
        ("today's total", Intent.GET_SUMMARY),
        ("aaje shu khadhu?", Intent.QUERY_MEAL),
        ("how many calories did I eat today", Intent.GET_SUMMARY),
        ("aaj kya khaya", Intent.QUERY_MEAL),
        ("what did I eat for breakfast today", Intent.QUERY_MEAL),
    ]:
        await bot.flow_mgr.save_flow_state(PendingFlow())
        res = await bot.process_message(query)
        test_assert(
            res.intent == expected_intent,
            f"Route '{query}' -> {expected_intent}",
            f"got: {res.intent}, msg: {res.message[:60]}"
        )
        flow_after = await bot.flow_mgr.get_pending_flow()
        test_assert(
            flow_after.state == FlowState.IDLE,
            f"Daily log query '{query}' does NOT trigger food flow (remains IDLE)"
        )

    # Test mid-flow interruption
    await bot.flow_mgr.save_flow_state(PendingFlow())
    r_food = await bot.process_message("methi thepla")
    test_assert(r_food.intent.startswith("flow:"), "Food input enters clarification flow")

    r_interrupt = await bot.process_message("aaj nu total log")
    test_assert(
        r_interrupt.intent == Intent.GET_SUMMARY,
        "Mid-flow 'aaj nu total log' successfully interrupts food flow and routes to GET_SUMMARY",
        f"got: {r_interrupt.intent}"
    )
    flow_interrupted = await bot.flow_mgr.get_pending_flow()
    test_assert(
        flow_interrupted.state == FlowState.IDLE,
        "Pending flow was reset to IDLE after interruption"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 7. MULTI-FOOD & GENERIC SPECIFICITY PRESERVATION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 7. Multi-food & Specificity Preservation ---")
    await bot.flow_mgr.save_flow_state(PendingFlow())
    r_multi = await bot.process_message("aaje me bajara no rotlo and shak khadhu")
    test_assert(
        "rotlo" in r_multi.message.lower() or "rotla" in r_multi.message.lower(),
        "Detected Bajra no Rotlo in multi-food phrase"
    )
    # Check that generic shak is preserved as an unresolved item
    flow_multi = await bot.flow_mgr.get_pending_flow()
    has_shak_unresolved = bool(
        getattr(flow_multi, "unresolved_items", None)
        and any("shak" in u.lower() or "shaak" in u.lower() for u in flow_multi.unresolved_items)
    )
    test_assert(
        has_shak_unresolved or "shaak" in r_multi.message.lower() or "shak" in r_multi.message.lower(),
        "Preserved generic 'shak' in flow without dropping it"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 8. FOOD FLOW QUANTITY CONSUMPTION VS EXERCISE MISROUTING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- 8. Food Flow Quantity Consumption vs Exercise Misrouting ---")
    
    # 8.1 Active flow in AWAITING_QUANTITY must consume valid food quantities
    food_thepla = repo.get_by_exact_name("methi thepla")
    fid = str(food_thepla.get("id") or food_thepla.get("_id"))
    
    quantity_test_replies = ["1 plate", "2 pieces", "100g", "1 bowl", "1 cup", "2 rotla", "150 ml"]
    for q_reply in quantity_test_replies:
        # Reset flow to awaiting quantity
        test_flow = PendingFlow(
            state=FlowState.AWAITING_QUANTITY,
            food_id=fid,
            food_name=food_thepla["food_name"],
            food_name_display="Methi Thepla",
            meal_type="breakfast",
            variant="normal",
            items=[{
                "food_id": fid,
                "food_name": food_thepla["food_name"],
                "food_name_display": "Methi Thepla",
                "food_doc": food_thepla,
                "variant": "normal",
                "quantity": None,
                "meal_type": "breakfast",
                "needs_variant": False,
                "needs_quantity": True,
            }],
            current_index=0,
        )
        await bot.flow_mgr.save_flow_state(test_flow)
        
        reply_res = await bot.process_message(q_reply)
        test_assert(
            "exercise" not in reply_res.intent.lower(),
            f"Reply '{q_reply}' in AWAITING_QUANTITY is NOT routed to exercise (intent: {reply_res.intent})"
        )
        test_assert(
            "lever" not in reply_res.message.lower() and "neck" not in reply_res.message.lower(),
            f"Reply '{q_reply}' in AWAITING_QUANTITY never mentions Lever Neck Extension"
        )
        test_assert(
            "thepla" in reply_res.message.lower() or reply_res.action_taken == "food_logged" or reply_res.intent.startswith("flow:"),
            f"Reply '{q_reply}' in AWAITING_QUANTITY stays in thepla food flow"
        )

    # 8.2 Standalone food portion / quantity expressions must never route to exercise
    standalone_quantities = ["1 plate", "100g", "2 pieces", "1 bowl", "2 cups", "1 katori"]
    for sq in standalone_quantities:
        await bot.flow_mgr.save_flow_state(PendingFlow())
        sq_res = await bot.process_message(sq)
        test_assert(
            sq_res.intent != Intent.LOG_EXERCISE,
            f"Standalone quantity '{sq}' does not route to LOG_EXERCISE (got: {sq_res.intent})"
        )
        test_assert(
            "lever" not in sq_res.message.lower() and "neck" not in sq_res.message.lower(),
            f"Standalone quantity '{sq}' does not resolve to Lever Neck Extension"
        )

    # 8.3 Genuine exercise queries MUST route to exercise
    genuine_exercises = [
        "15 pushups",
        "did 30 min yoga",
        "ran 5 km",
        "cycling 45 minutes",
        "30 min jogging",
        "did 20 squats"
    ]
    for ge in genuine_exercises:
        test_assert(
            bot._looks_like_exercise(ge.lower()),
            f"Exercise query '{ge}' detected as exercise intent by engine"
        )

    # 8.4 Multi-food flow consuming quantity and transitioning to next food
    await bot.flow_mgr.save_flow_state(PendingFlow())
    r_multi_init = await bot.process_message("aaje me bajara no rotlo and shak khadhu")
    flow_m = await bot.flow_mgr.get_pending_flow()
    if flow_m.state != FlowState.IDLE:
        r_qty_reply = await bot.process_message("1 plate")
        test_assert(
            "exercise" not in r_qty_reply.intent.lower(),
            "Multi-food quantity reply '1 plate' is not routed to exercise"
        )
        test_assert(
            "lever" not in r_qty_reply.message.lower(),
            "Multi-food quantity reply '1 plate' does not mention Lever Neck Extension"
        )

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE: {passes} PASSED, {fails} FAILED")
    print("=" * 70)
    if fails > 0:
        print("\nFailures:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_benchmarks())
