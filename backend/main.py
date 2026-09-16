"""
Fitness Chatbot Backend API — MongoDB version (No Auth)
"""
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime, date as date_type, timezone, timedelta
import os

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

def _today_ist():
    """Get today's date in IST."""
    return datetime.now(IST).date()

import schemas
from database import get_db, get_settings, validate_config, create_indexes, sync_db, async_db
from food_repository import FoodRepository, FoodRepositorySync
from search_engine import FoodSearchEngine
from quantity_parser import parse_quantity
from nutrition_calculator import NutritionCalculator
from exercise_calculator import ExerciseSearcher, ExerciseCalculator, parse_exercise_input
from user_repository import UserRepository
from bmr_tdee_calculator import compute_full_target
from daily_log_repository import DailyLogRepository
from conversation import ConversationStore

settings = get_settings()
_config_warnings = validate_config()

# Default user ID (no login required)
DEFAULT_USER_ID = "default_user"

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend API for Personalized Fitness AI Chatbot (MongoDB)",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def startup_event():
    await create_indexes()
    # Auto-create default user if not exists
    db = async_db
    existing = await db["user_profiles"].find_one({"user_id": DEFAULT_USER_ID})
    if not existing:
        try:
            await db["user_profiles"].insert_one({
                "user_id": DEFAULT_USER_ID,
                "name": "User",
                "email": None,
                "age": 25,
                "gender": "male",
                "height_cm": 170,
                "weight_kg": 70,
                "activity_level": "moderate",
                "fitness_goal": "maintain",
                "diet_type": "non_veg",
                "target_weight_kg": None,
                "medical_conditions": None,
                "custom_calorie_goal": None,
            })
        except Exception:
            pass  # Already exists or index conflict — safe to ignore


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# FRONTEND STATIC FILES
# ============================================================================

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend-static")

    @app.get("/app", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ============================================================================
# HELPERS
# ============================================================================

def _sync_food_repo():
    return FoodRepositorySync(sync_db)


def _search_engine():
    return FoodSearchEngine(_sync_food_repo())


def _build_smart_result(result) -> schemas.SmartSearchResult:
    food_schema = schemas.FoodDetail(**result.food) if result.food else None
    return schemas.SmartSearchResult(
        query_original=result.query_original,
        query_normalized=result.query_normalized,
        match_type=result.match_type.value,
        confidence=result.confidence,
        food=food_schema,
        variant_detected=result.variant_detected,
        stripped_query=result.stripped_query,
        candidates=[schemas.SmartSearchCandidate(**c) for c in result.candidates],
        message=result.message,
    )


def _food_log_response(log: dict) -> schemas.FoodLogResponse:
    return schemas.FoodLogResponse(
        id=str(log.get("id", "")),
        user_id=log["user_id"],
        log_date=log["log_date"],
        meal_type=log["meal_type"],
        food_id=log["food_id"],
        food_name=log["food_name"],
        food_name_display=log["food_name_display"],
        quantity_input=log["quantity_input"],
        quantity_amount=log["quantity_amount"],
        quantity_unit=log["quantity_unit"],
        quantity_grams=log["quantity_grams"],
        variant=log.get("variant"),
        calories=log["calories"],
        protein_g=log["protein_g"],
        carbs_g=log["carbs_g"],
        fat_g=log["fat_g"],
        fiber_g=log["fiber_g"],
        search_confidence=log.get("search_confidence"),
        created_at=log.get("created_at"),
    )


def _exercise_log_response(log: dict) -> schemas.ExerciseLogResponse:
    return schemas.ExerciseLogResponse(
        id=str(log.get("id", "")),
        user_id=log["user_id"],
        log_date=log["log_date"],
        exercise_id=log["exercise_id"],
        exercise_name=log["exercise_name"],
        exercise_name_display=log["exercise_name_display"],
        category=log["category"],
        exercise_input=log["exercise_input"],
        amount=log["amount"],
        unit=log["unit"],
        calories_min=log["calories_min"],
        calories_avg=log["calories_avg"],
        calories_max=log["calories_max"],
        search_confidence=log.get("search_confidence"),
        created_at=log.get("created_at"),
    )


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/", response_model=schemas.HealthCheckResponse)
@app.get("/health", response_model=schemas.HealthCheckResponse)
async def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(),
    }


@app.get("/stats", response_model=schemas.StatsResponse)
async def get_stats(db=Depends(get_db)):
    total_foods = await db["foods"].count_documents({})
    total_exercises = await db["exercises"].count_documents({})
    total_aliases = await db["food_aliases"].count_documents({})
    veg = await db["foods"].count_documents({"vegetarian_status": "Veg"})
    non_veg = await db["foods"].count_documents({"vegetarian_status": "Non-Veg"})
    egg = await db["foods"].count_documents({"vegetarian_status": "Eggetarian"})
    cats = await db["exercises"].distinct("category")
    return {
        "total_foods": total_foods,
        "total_exercises": total_exercises,
        "total_aliases": total_aliases,
        "vegetarian_foods": veg,
        "non_veg_foods": non_veg,
        "eggetarian_foods": egg,
        "exercise_categories": sorted([c for c in cats if c]),
    }


# ============================================================================
# FOOD ENDPOINTS
# ============================================================================

@app.get("/api/foods/search", response_model=schemas.FoodSearchResponse)
async def search_foods(
    query: Optional[str] = Query(None),
    vegetarian_status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    repo = FoodRepository(db)
    foods = await repo.search(query=query, vegetarian_status=vegetarian_status,
                              category=category, limit=limit, offset=offset)
    total = await repo.count(query=query, vegetarian_status=vegetarian_status, category=category)
    return {"total": total, "foods": foods}


@app.get("/api/foods/smart-search", response_model=schemas.SmartSearchResult, tags=["Food Search Engine"])
async def smart_search_food(query: str = Query(..., min_length=1)):
    """Full matching pipeline: exact → alias → variant strip → fuzzy → low-confidence."""
    result = _search_engine().search(query)
    return _build_smart_result(result)


@app.get("/api/foods/multi-search", response_model=schemas.MultiSearchResponse, tags=["Food Search Engine"])
async def multi_search_food(query: str = Query(..., min_length=1), limit: int = Query(5, ge=1, le=20)):
    results = _search_engine().search_multi(query, limit=limit)
    return schemas.MultiSearchResponse(query=query, results=[_build_smart_result(r) for r in results])


@app.post("/api/foods/calculate-nutrition", response_model=schemas.NutritionCalcResponse, tags=["Nutrition Calculator"])
async def calculate_nutrition(request: schemas.NutritionCalcRequest):
    search_result = _search_engine().search(request.food_query)

    if search_result.match_type.value in ("no_match", "low_confidence"):
        raise HTTPException(status_code=422, detail=f"Could not identify food '{request.food_query}'")

    food = search_result.food
    parsed_qty = parse_quantity(request.quantity)
    if parsed_qty.amount is None or parsed_qty.amount <= 0 or not parsed_qty.unit:
        parsed_qty = ParsedQuantity(amount=1.0, unit="serving", raw_input=request.quantity, confidence=1.0)
    variant = request.variant
    if not variant and search_result.variant_detected:
        variant = search_result.variant_detected.split(",")[0].strip()

    result = NutritionCalculator().calculate(food, parsed_qty, variant=variant)
    overall_confidence = min(search_result.confidence, parsed_qty.confidence)

    return schemas.NutritionCalcResponse(
        food_id=result.food_id, food_name=result.food_name,
        food_name_display=result.food_name_display,
        search_match_type=search_result.match_type.value,
        search_confidence=search_result.confidence,
        quantity_input=request.quantity, quantity_amount=result.quantity_amount,
        quantity_unit=result.quantity_unit, quantity_grams=result.quantity_grams,
        variant=result.variant, variant_description=result.variant_description,
        calories=result.calories, protein_g=result.protein_g, carbs_g=result.carbs_g,
        fat_g=result.fat_g, fiber_g=result.fiber_g, sugar_g=result.sugar_g,
        sodium_mg=result.sodium_mg, serving_size_g=result.serving_size_g,
        calculation_note=result.calculation_note, overall_confidence=overall_confidence,
    )


@app.get("/api/foods/by-name/{food_name}", response_model=schemas.FoodDetail)
async def get_food_by_name(food_name: str, db=Depends(get_db)):
    food = await FoodRepository(db).get_by_exact_name(food_name)
    if not food:
        raise HTTPException(status_code=404, detail=f"Food '{food_name}' not found")
    return food


@app.get("/api/foods/alias/{alias}")
async def search_by_alias(alias: str, db=Depends(get_db)):
    repo = FoodRepository(db)
    alias_row = await repo.get_exact_alias(alias)
    if alias_row:
        food = await repo.get_by_id(alias_row["food_id"])
        if food:
            return {"food": food, "matched_alias": alias_row["alias"]}
    foods = await repo.search_by_alias_substring(alias, limit=10)
    if not foods:
        raise HTTPException(status_code=404, detail=f"No foods found for alias '{alias}'")
    return {"foods": foods, "query": alias}


@app.post("/api/foods/calculate-portion", response_model=schemas.FoodPortionResponse)
async def calculate_food_portion(request: schemas.FoodPortionRequest, db=Depends(get_db)):
    food = await FoodRepository(db).get_by_id(request.food_id)
    if not food:
        raise HTTPException(status_code=404, detail=f"Food '{request.food_id}' not found")
    ratio = request.portion_g / food["serving_size_g"]
    return {
        "food_id": food["food_id"], "food_name": food["food_name"],
        "food_name_display": food["food_name_display"], "portion_g": request.portion_g,
        "calories": round(food["calories_kcal"] * ratio, 1),
        "protein_g": round(food["protein_g"] * ratio, 1),
        "carbs_g": round(food["carbs_g"] * ratio, 1),
        "fat_g": round(food["fat_g"] * ratio, 1),
        "fiber_g": round(food["fiber_g"] * ratio, 1),
        "sugar_g": round(food["sugar_g"] * ratio, 1) if food.get("sugar_g") else None,
        "sodium_mg": round(food["sodium_mg"] * ratio, 1) if food.get("sodium_mg") else None,
    }


@app.get("/api/foods/{food_id}", response_model=schemas.FoodDetail)
async def get_food(food_id: str, db=Depends(get_db)):
    food = await FoodRepository(db).get_by_id(food_id)
    if not food:
        raise HTTPException(status_code=404, detail=f"Food '{food_id}' not found")
    return food


# ============================================================================
# EXERCISE ENDPOINTS
# ============================================================================

@app.get("/api/exercises/search", response_model=schemas.ExerciseSearchResponse)
async def search_exercises(
    query: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    filter_q = {}
    if query:
        filter_q["exercise_name"] = {"$regex": query.lower(), "$options": "i"}
    if category:
        filter_q["category"] = category
    cursor = db["exercises"].find(filter_q, {"_id": 0}).skip(offset).limit(limit)
    exercises = await cursor.to_list(length=limit)
    total = await db["exercises"].count_documents(filter_q)
    return {"total": total, "exercises": exercises}


@app.get("/api/exercises/categories")
async def get_exercise_categories(db=Depends(get_db)):
    cats = await db["exercises"].distinct("category")
    return {"categories": sorted([c for c in cats if c])}


@app.get("/api/exercises/by-name/{exercise_name}", response_model=schemas.ExerciseDetail)
async def get_exercise_by_name(exercise_name: str, db=Depends(get_db)):
    ex = await db["exercises"].find_one({"exercise_name": exercise_name.lower()}, {"_id": 0})
    if not ex:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_name}' not found")
    return ex


@app.post("/api/exercises/calculate-calories", response_model=schemas.ExerciseCalorieResponse)
async def calculate_exercise_calories(request: schemas.ExerciseCalorieRequest, db=Depends(get_db)):
    ex = await db["exercises"].find_one({"exercise_id": request.exercise_id}, {"_id": 0})
    if not ex:
        raise HTTPException(status_code=404, detail=f"Exercise '{request.exercise_id}' not found")
    return {
        "exercise_id": ex["exercise_id"], "exercise_name": ex["exercise_name"],
        "exercise_name_display": ex["exercise_name_display"], "category": ex["category"],
        "amount": request.amount, "unit": ex["measurement_unit"],
        "calories_burned": round(request.amount * ex["calories_per_unit_avg"], 1),
        "calories_min": round(request.amount * ex["calories_per_unit_min"], 1),
        "calories_max": round(request.amount * ex["calories_per_unit_max"], 1),
    }


@app.post("/api/exercises/smart-calculate", response_model=schemas.SmartExerciseCalcResponse, tags=["Exercise Calculator"])
async def smart_calculate_exercise(request: schemas.SmartExerciseCalcRequest):
    parsed = parse_exercise_input(request.exercise_input)
    if not parsed.exercise_query:
        raise HTTPException(status_code=422, detail="Could not parse exercise name from input.")

    searcher = ExerciseSearcher(sync_db)
    exercise, confidence = searcher.search(parsed.exercise_query)

    if not exercise:
        raise HTTPException(status_code=422, detail=f"No exercise found matching '{parsed.exercise_query}'.")

    result = ExerciseCalculator().calculate(exercise, amount=parsed.amount, unit=parsed.unit)

    return schemas.SmartExerciseCalcResponse(
        exercise_id=result.exercise_id, exercise_name=result.exercise_name,
        exercise_name_display=result.exercise_name_display, category=result.category,
        input_raw=request.exercise_input, input_amount=result.input_amount,
        input_unit=result.input_unit, effective_amount=result.amount,
        effective_unit=result.unit, calories_min=result.calories_min,
        calories_avg=result.calories_avg, calories_max=result.calories_max,
        search_confidence=confidence, typical_duration_min=result.typical_duration_min,
        calculation_note=result.calculation_note,
    )


@app.get("/api/exercises/{exercise_id}", response_model=schemas.ExerciseDetail)
async def get_exercise(exercise_id: str, db=Depends(get_db)):
    ex = await db["exercises"].find_one({"exercise_id": exercise_id}, {"_id": 0})
    if not ex:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return ex


# ============================================================================
# USER PROFILE ENDPOINTS
# ============================================================================

@app.post("/api/users", response_model=schemas.UserProfileResponse, status_code=201, tags=["User Profile"])
async def create_user_profile(data: schemas.UserProfileCreate, db=Depends(get_db)):
    repo = UserRepository(db)
    payload = data.model_dump()
    # Convert enums to values
    for key in ("gender", "activity_level", "fitness_goal", "diet_type"):
        if hasattr(payload.get(key), "value"):
            payload[key] = payload[key].value
    try:
        user = await repo.create(payload)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return user


@app.get("/api/users", response_model=list[schemas.UserProfileResponse], tags=["User Profile"])
async def list_users(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), db=Depends(get_db)):
    return await UserRepository(db).get_all(limit=limit, offset=offset)


@app.get("/api/users/{user_id}", response_model=schemas.UserProfileResponse, tags=["User Profile"])
async def get_user_profile(user_id: str, db=Depends(get_db)):
    user = await UserRepository(db).get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return user


@app.put("/api/users/{user_id}", response_model=schemas.UserProfileResponse, tags=["User Profile"])
async def update_user_profile(user_id: str, data: schemas.UserProfileUpdate, db=Depends(get_db)):
    payload = data.model_dump(exclude_unset=True)
    user = await UserRepository(db).update(user_id, payload)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return user


@app.delete("/api/users/{user_id}", tags=["User Profile"])
async def delete_user_profile(user_id: str, db=Depends(get_db)):
    ok = await UserRepository(db).delete(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return {"message": f"User '{user_id}' deleted"}


@app.get("/api/users/{user_id}/calorie-target", response_model=schemas.CalorieTargetResponse, tags=["User Profile"])
async def get_calorie_target(user_id: str, db=Depends(get_db)):
    user = await UserRepository(db).get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    target = compute_full_target(user)
    return schemas.CalorieTargetResponse(
        user_id=user["user_id"], name=user["name"], age=user["age"],
        gender=user["gender"], height_cm=user["height_cm"], weight_kg=user["weight_kg"],
        activity_level=user["activity_level"], fitness_goal=user["fitness_goal"],
        bmr=target.bmr, tdee=target.tdee, calorie_target=target.calorie_target,
        protein_target_g=target.protein_target_g, carbs_target_g=target.carbs_target_g,
        fat_target_g=target.fat_target_g, bmr_formula=target.bmr_formula,
        tdee_multiplier=target.tdee_multiplier, calorie_adjustment=target.calorie_adjustment,
        custom_calorie_goal=user.get("custom_calorie_goal"),
    )


# ============================================================================
# DAILY LOGGING ENDPOINTS
# ============================================================================

@app.post("/api/logs/food", response_model=schemas.FoodLogResponse, status_code=201, tags=["Daily Logging"])
async def log_food(request: schemas.LogFoodRequest, db=Depends(get_db)):
    log_date = date_type.fromisoformat(request.log_date) if request.log_date else _today_ist()

    search_result = _search_engine().search(request.food_query)
    if search_result.match_type.value in ("no_match", "low_confidence"):
        raise HTTPException(status_code=422, detail=f"Could not identify food '{request.food_query}'.")

    food = search_result.food
    parsed_qty = parse_quantity(request.quantity)
    if parsed_qty.amount is None or parsed_qty.amount <= 0 or not parsed_qty.unit:
        parsed_qty = ParsedQuantity(amount=1.0, unit="serving", raw_input=request.quantity, confidence=1.0)
    variant = request.variant
    if not variant and search_result.variant_detected:
        variant = search_result.variant_detected.split(",")[0].strip()

    result = NutritionCalculator().calculate(food, parsed_qty, variant=variant)

    log = await DailyLogRepository(db).create_food_log(
        user_id=request.user_id, log_date=log_date, meal_type=request.meal_type.value,
        food_id=food["food_id"], food_name=food["food_name"],
        food_name_display=food["food_name_display"], quantity_input=request.quantity,
        quantity_amount=result.quantity_amount, quantity_unit=result.quantity_unit,
        quantity_grams=result.quantity_grams,
        variant=result.variant if result.variant != "normal" else None,
        calories=result.calories, protein_g=result.protein_g, carbs_g=result.carbs_g,
        fat_g=result.fat_g, fiber_g=result.fiber_g, search_confidence=search_result.confidence,
    )
    return _food_log_response(log)


@app.post("/api/logs/exercise", response_model=schemas.ExerciseLogResponse, status_code=201, tags=["Daily Logging"])
async def log_exercise(request: schemas.LogExerciseRequest, db=Depends(get_db)):
    log_date = date_type.fromisoformat(request.log_date) if request.log_date else _today_ist()

    parsed = parse_exercise_input(request.exercise_input)
    if not parsed.exercise_query:
        raise HTTPException(status_code=422, detail="Could not parse exercise from input.")

    exercise, confidence = ExerciseSearcher(sync_db).search(parsed.exercise_query)
    if not exercise:
        raise HTTPException(status_code=422, detail=f"No exercise found matching '{parsed.exercise_query}'.")

    # Burn scales with the logging user's actual body weight.
    _log_user = await UserRepository(db).get_by_id(request.user_id)
    _weight_kg = None
    if _log_user:
        try:
            _w = float(_log_user.get("weight_kg") or _log_user.get("weight") or 0)
            _weight_kg = _w if _w > 0 else None
        except (TypeError, ValueError):
            _weight_kg = None

    calc = ExerciseCalculator().calculate(exercise, amount=parsed.amount, unit=parsed.unit, weight_kg=_weight_kg)

    log = await DailyLogRepository(db).create_exercise_log(
        user_id=request.user_id, log_date=log_date,
        exercise_id=exercise["exercise_id"], exercise_name=exercise["exercise_name"],
        exercise_name_display=exercise["exercise_name_display"], category=exercise["category"],
        exercise_input=request.exercise_input, amount=calc.amount, unit=calc.unit,
        calories_min=calc.calories_min, calories_avg=calc.calories_avg,
        calories_max=calc.calories_max, search_confidence=confidence,
    )
    return _exercise_log_response(log)


@app.get("/api/logs/{user_id}/today", response_model=schemas.DayLogsResponse, tags=["Daily Logging"])
async def get_today_logs(user_id: str, db=Depends(get_db)):
    today = _today_ist()
    repo = DailyLogRepository(db)
    food_logs = await repo.get_food_logs_by_date(user_id, today)
    exercise_logs = await repo.get_exercise_logs_by_date(user_id, today)
    return schemas.DayLogsResponse(
        user_id=user_id, date=today.isoformat(),
        food_logs=[_food_log_response(l) for l in food_logs],
        exercise_logs=[_exercise_log_response(l) for l in exercise_logs],
    )


@app.get("/api/logs/{user_id}/date/{log_date}", response_model=schemas.DayLogsResponse, tags=["Daily Logging"])
async def get_logs_by_date(user_id: str, log_date: str, db=Depends(get_db)):
    d = date_type.fromisoformat(log_date)
    repo = DailyLogRepository(db)
    food_logs = await repo.get_food_logs_by_date(user_id, d)
    exercise_logs = await repo.get_exercise_logs_by_date(user_id, d)
    return schemas.DayLogsResponse(
        user_id=user_id, date=log_date,
        food_logs=[_food_log_response(l) for l in food_logs],
        exercise_logs=[_exercise_log_response(l) for l in exercise_logs],
    )


@app.get("/api/logs/{user_id}/range", tags=["Daily Logging"])
async def get_logs_by_range(
    user_id: str,
    start_date: str = Query(...),
    end_date: str = Query(...),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db=Depends(get_db),
):
    s = date_type.fromisoformat(start_date)
    e = date_type.fromisoformat(end_date)
    repo = DailyLogRepository(db)
    food_logs = await repo.get_food_logs_by_range(user_id, s, e)
    exercise_logs = await repo.get_exercise_logs_by_range(user_id, s, e)
    return {
        "user_id": user_id, "start_date": start_date, "end_date": end_date,
        "total_food_logs": len(food_logs), "total_exercise_logs": len(exercise_logs),
        "limit": limit, "offset": offset,
        "food_logs": [_food_log_response(l) for l in food_logs[offset:offset + limit]],
        "exercise_logs": [_exercise_log_response(l) for l in exercise_logs[offset:offset + limit]],
    }


@app.get("/api/logs/{user_id}/summary/{log_date}", response_model=schemas.DailySummaryResponse, tags=["Daily Logging"])
async def get_daily_summary(user_id: str, log_date: str, db=Depends(get_db)):
    d = date_type.fromisoformat(log_date)
    user = await UserRepository(db).get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    target = compute_full_target(user)
    repo = DailyLogRepository(db)
    food_totals = await repo.get_daily_food_totals(user_id, d)
    exercise_totals = await repo.get_daily_exercise_totals(user_id, d)

    net = food_totals["total_calories_consumed"] - exercise_totals["total_calories_burned"]

    return schemas.DailySummaryResponse(
        user_id=user_id, date=log_date, calorie_target=target.calorie_target,
        total_calories_consumed=food_totals["total_calories_consumed"],
        total_protein_g=food_totals["total_protein_g"],
        total_carbs_g=food_totals["total_carbs_g"],
        total_fat_g=food_totals["total_fat_g"],
        total_fiber_g=food_totals["total_fiber_g"],
        total_calories_burned=exercise_totals["total_calories_burned"],
        net_calories=round(net, 1),
        remaining_calories=round(target.calorie_target - net, 1),
        food_entries=food_totals["food_entries"],
        exercise_entries=exercise_totals["exercise_entries"],
        meal_breakdown=food_totals["meal_breakdown"],
    )


@app.delete("/api/logs/food/{log_id}", tags=["Daily Logging"])
async def delete_food_log(log_id: str, user_id: str = Query(...), db=Depends(get_db)):
    ok = await DailyLogRepository(db).delete_food_log(log_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Food log not found or not owned by user")
    return {"message": f"Food log {log_id} deleted"}


@app.delete("/api/logs/exercise/{log_id}", tags=["Daily Logging"])
async def delete_exercise_log(log_id: str, user_id: str = Query(...), db=Depends(get_db)):
    ok = await DailyLogRepository(db).delete_exercise_log(log_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Exercise log not found or not owned by user")
    return {"message": f"Exercise log {log_id} deleted"}


# ============================================================================
# CHATBOT ENDPOINTS (No auth — uses default user)
# ============================================================================

async def _get_default_user(db):
    """Get the default user profile for chatbot."""
    user = await db["user_profiles"].find_one({"user_id": DEFAULT_USER_ID}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=500, detail="Default user not found. Restart server.")
    return user


@app.post("/api/chat/message", response_model=schemas.ChatResponse, tags=["Chatbot"])
async def chat_message(
    request: schemas.ChatMessageRequest,
    db=Depends(get_db),
):
    """Text chatbot. Supports English/Hindi/Gujarati with multi-turn clarification."""
    from chatbot_engine import ChatbotEngine
    from llm_service import LLMService

    user = await _get_default_user(db)
    engine = ChatbotEngine(db, user, LLMService())
    result = await engine.process_message(request.message, context=request.context, auto_log=request.auto_log)

    return schemas.ChatResponse(
        message=result.message, intent=result.intent, action_taken=result.action_taken,
        data=result.data, needs_confirmation=result.needs_confirmation,
        pending_action=result.pending_action, options=result.options, success=result.success,
    )


@app.post("/api/chat/confirm", response_model=schemas.ChatResponse, tags=["Chatbot"])
async def chat_confirm(
    request: schemas.ChatConfirmRequest,
    db=Depends(get_db),
):
    from chatbot_engine import ChatbotEngine
    from llm_service import LLMService

    if not request.confirm:
        return schemas.ChatResponse(message="Cancelled. What else can I help with?", intent="cancelled", success=True)

    user = await _get_default_user(db)
    engine = ChatbotEngine(db, user, LLMService())
    result = await engine.execute_confirmed_action(request.pending_action)

    return schemas.ChatResponse(
        message=result.message, intent=result.intent, action_taken=result.action_taken,
        data=result.data, needs_confirmation=False, options=result.options, success=result.success,
    )


@app.post("/api/chat/voice", response_model=schemas.ChatResponse, tags=["Chatbot"])
async def chat_voice(
    audio: UploadFile = File(...),
    language: Optional[str] = Query(None, description="2-letter language hint (en/hi/gu/…)"),
    auto_log: bool = Query(False),
    stt_provider: Optional[str] = Query(
        None,
        description="STT provider: 'auto' (default), 'sarvam', or 'whisper'. "
                    "'auto' uses Sarvam when SARVAM_API_KEY is set, else Whisper.",
    ),
    db=Depends(get_db),
):
    """Voice → STT (Sarvam Saaras v3 or Groq Whisper) → chatbot pipeline.

    - stt_provider='auto' (default): Sarvam when key present, else Whisper.
    - stt_provider='sarvam': Force Sarvam Saaras v3.
    - stt_provider='whisper': Force Groq Whisper Large V3.
    """
    from stt_service import STTService
    from chatbot_engine import ChatbotEngine
    from llm_service import LLMService

    audio_data = await audio.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="No audio data in request")

    provider = (stt_provider or "auto").lower().strip()

    try:
        stt = STTService(provider=provider)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=f"STT service not configured: {e}")

    stt_result = await stt.transcribe(
        audio_data,
        filename=audio.filename or "audio.webm",
        language=language,
    )

    if not stt_result.success:
        return schemas.ChatResponse(
            message=f"Could not transcribe audio: {stt_result.error}",
            intent="stt_error", success=False,
            data={"stt_error": stt_result.error, "stt_provider": stt_result.provider},
        )

    display_text = stt_result.raw_text or stt_result.text
    process_text = stt_result.text   # normalised — fed to chatbot

    user = await _get_default_user(db)
    engine = ChatbotEngine(db, user, LLMService())
    result = await engine.process_message(process_text, auto_log=auto_log)

    return schemas.ChatResponse(
        message=result.message, intent=result.intent, action_taken=result.action_taken,
        data={
            **result.data,
            "transcribed_text":  display_text,
            "normalised_text":   process_text,
            "translit_text":     getattr(stt_result, "translit_text", ""),
            "stt_language":      stt_result.language,
            "stt_confidence":    stt_result.confidence,
            "stt_provider":      stt_result.provider,
        },
        needs_confirmation=result.needs_confirmation, pending_action=result.pending_action,
        options=result.options, success=result.success,
    )


# ── STT comparison endpoint ──────────────────────────────────────────────────

@app.post("/api/stt/compare", tags=["STT"])
async def stt_compare(
    audio: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language hint (en/hi/gu/…)"),
):
    """
    Run the same audio through BOTH Sarvam Saaras v3 AND Groq Whisper and
    return side-by-side results for comparison.

    Response fields per provider:
      transcript   — native-script / verbatim text
      translit     — Roman transliteration (Sarvam only)
      normalised   — food-search-ready text
      language     — detected language code
      confidence   — detection confidence 0–1
      latency_ms   — round-trip time in milliseconds
      success      — bool
      error        — error string if success=False
    """
    import asyncio, time
    from stt_service import STTService

    audio_data = await audio.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="No audio data")

    filename = audio.filename or "audio.webm"

    async def run_provider(provider: str) -> dict:
        t0 = time.monotonic()
        try:
            stt = STTService(provider=provider)
        except ValueError as e:
            return {
                "provider": provider, "success": False,
                "error": str(e), "latency_ms": 0,
                "transcript": "", "translit": "",
                "normalised": "", "language": "", "confidence": 0.0,
            }
        result = await stt.transcribe(audio_data, filename=filename, language=language)
        latency = int((time.monotonic() - t0) * 1000)
        return {
            "provider":   result.provider,
            "success":    result.success,
            "error":      result.error,
            "latency_ms": latency,
            "transcript": result.raw_text,
            "translit":   getattr(result, "translit_text", ""),
            "normalised": result.text,
            "language":   result.language,
            "confidence": round(result.confidence, 3),
        }

    # Run both providers in parallel
    sarvam_task  = asyncio.create_task(run_provider("sarvam"))
    whisper_task = asyncio.create_task(run_provider("whisper"))
    sarvam_out, whisper_out = await asyncio.gather(sarvam_task, whisper_task)

    # Simple agreement check
    def _tokens(t: str) -> set:
        return {w.lower().strip(".,!?") for w in (t or "").split() if len(w) > 2}

    s_tok = _tokens(sarvam_out["normalised"])
    w_tok = _tokens(whisper_out["normalised"])
    overlap = s_tok & w_tok
    agreement = round(len(overlap) / max(len(s_tok | w_tok), 1), 2)

    return {
        "sarvam":    sarvam_out,
        "whisper":   whisper_out,
        "agreement": agreement,          # 0–1, token-level overlap ratio
        "filename":  filename,
        "language_hint": language,
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", settings.API_PORT))

    print(f"\n  Fitness AI Chatbot is running:")
    print(f"  Local:     http://localhost:{port}/app")
    print(f"  API docs:  http://localhost:{port}/docs")
    print(f"  Press CTRL+C to stop.\n")

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,  # 0.0.0.0 — required for Render and other cloud hosts
        port=port,
        reload=settings.API_RELOAD,
    )
