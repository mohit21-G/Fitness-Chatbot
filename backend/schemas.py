"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Union, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class Gender(str, Enum):
    male = "male"
    female = "female"


class ActivityLevel(str, Enum):
    sedentary = "sedentary"          # desk job, no exercise
    light = "light"                  # 1-3 days/week light exercise
    moderate = "moderate"            # 3-5 days/week moderate exercise
    active = "active"                # 6-7 days/week hard exercise
    very_active = "very_active"      # athlete / 2x per day


class FitnessGoal(str, Enum):
    lose_weight = "lose_weight"
    maintain = "maintain"
    gain_muscle = "gain_muscle"
    gain_weight = "gain_weight"


class DietType(str, Enum):
    veg = "veg"
    non_veg = "non_veg"
    eggetarian = "eggetarian"
    vegan = "vegan"


# ============================================================================
# USER PROFILE SCHEMAS
# ============================================================================

class UserProfileCreate(BaseModel):
    """Request to create/register a user profile (from website signup)."""
    user_id: str = Field(..., min_length=1, description="Unique user ID from website auth")
    name: str = Field(..., min_length=1)
    email: Optional[str] = None
    password: Optional[str] = Field(None, min_length=6, description="Password (hashed before storage)")

    age: int = Field(..., ge=10, le=120)
    gender: Gender
    height_cm: float = Field(..., ge=50, le=300)
    weight_kg: float = Field(..., ge=20, le=500)

    activity_level: ActivityLevel
    fitness_goal: FitnessGoal
    diet_type: DietType

    target_weight_kg: Optional[float] = Field(None, ge=20, le=500)
    medical_conditions: Optional[str] = None


class UserProfileUpdate(BaseModel):
    """Request to update user profile fields (partial update)."""
    name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = Field(None, ge=10, le=120)
    gender: Optional[Gender] = None
    height_cm: Optional[float] = Field(None, ge=50, le=300)
    weight_kg: Optional[float] = Field(None, ge=20, le=500)
    activity_level: Optional[ActivityLevel] = None
    fitness_goal: Optional[FitnessGoal] = None
    diet_type: Optional[DietType] = None
    target_weight_kg: Optional[float] = Field(None, ge=20, le=500)
    medical_conditions: Optional[str] = None


class UserProfileResponse(BaseModel):
    """Full user profile response."""
    user_id: str
    name: str
    email: Optional[str] = None
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    activity_level: str
    fitness_goal: str
    diet_type: str
    target_weight_kg: Optional[float] = None
    medical_conditions: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class CalorieTargetResponse(BaseModel):
    """BMR + TDEE + personalized calorie target."""
    user_id: str
    name: str

    # Body stats
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    activity_level: str
    fitness_goal: str

    # Calculations
    bmr: float                   # Basal Metabolic Rate (kcal/day)
    tdee: float                  # Total Daily Energy Expenditure
    calorie_target: float        # Goal-adjusted daily calories
    protein_target_g: float      # Recommended daily protein
    carbs_target_g: float        # Recommended daily carbs
    fat_target_g: float          # Recommended daily fat

    # Explanation
    bmr_formula: str             # "Mifflin-St Jeor"
    tdee_multiplier: float       # Activity multiplier used
    calorie_adjustment: str      # e.g. "-500 kcal (lose weight)"


# ============================================================================
# FOOD SCHEMAS
# ============================================================================

class FoodBase(BaseModel):
    """Base food schema"""
    food_id: str
    food_name: str
    food_name_display: str
    category: Optional[str] = None
    vegetarian_status: str
    serving_size_g: float
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class FoodDetail(FoodBase):
    """Detailed food schema with all fields"""
    aliases: Optional[Union[list[str], str]] = None
    subcategory: Optional[str] = None
    region: Optional[str] = None
    state: Optional[str] = None
    calories_per_100g: float
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    preparation_variant: Optional[str] = None
    is_verified: bool
    data_source: Optional[str] = None
    
    class Config:
        from_attributes = True


class FoodSearchResponse(BaseModel):
    """Food search response"""
    total: int
    foods: list[FoodBase]


class FoodPortionRequest(BaseModel):
    """Request for calculating food calories with custom portion"""
    food_id: str
    portion_g: float = Field(gt=0, description="Portion size in grams")


class FoodPortionResponse(BaseModel):
    """Response for food calorie calculation"""
    food_id: str
    food_name: str
    food_name_display: str
    portion_g: float
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None


# ============================================================================
# EXERCISE SCHEMAS
# ============================================================================

class ExerciseBase(BaseModel):
    """Base exercise schema"""
    exercise_id: str
    exercise_name: str
    exercise_name_display: str
    category: str
    measurement_unit: str
    calories_per_unit_avg: float


class ExerciseDetail(ExerciseBase):
    """Detailed exercise schema"""
    calories_per_unit_min: float
    calories_per_unit_max: float
    typical_duration_min: Optional[float] = None
    session_calories_min: Optional[float] = None
    session_calories_max: Optional[float] = None
    session_calories_avg: Optional[float] = None
    is_verified: bool
    
    class Config:
        from_attributes = True


class ExerciseSearchResponse(BaseModel):
    """Exercise search response"""
    total: int
    exercises: list[ExerciseBase]


class ExerciseCalorieRequest(BaseModel):
    """Request for calculating exercise calories"""
    exercise_id: str
    amount: float = Field(gt=0, description="Amount (duration in minutes or reps)")


class ExerciseCalorieResponse(BaseModel):
    """Response for exercise calorie calculation"""
    exercise_id: str
    exercise_name: str
    exercise_name_display: str
    category: str
    amount: float
    unit: str
    calories_burned: float
    calories_min: float
    calories_max: float


class SmartExerciseCalcRequest(BaseModel):
    """Request for smart exercise calorie calculation (natural language)."""
    exercise_input: str = Field(
        ..., min_length=1,
        description="Natural language input, e.g. 'jogging 30 minutes', '50 push-ups', '3 sets of 12 squats'"
    )


class SmartExerciseCalcResponse(BaseModel):
    """Response from POST /api/exercises/smart-calculate."""
    exercise_id: str
    exercise_name: str
    exercise_name_display: str
    category: str

    # Input echo
    input_raw: str
    input_amount: float
    input_unit: str

    # Effective (after conversion)
    effective_amount: float
    effective_unit: str

    # Calories
    calories_min: float
    calories_avg: float
    calories_max: float

    # Metadata
    search_confidence: float
    typical_duration_min: Optional[float] = None
    calculation_note: str


# ============================================================================
# ALIAS SCHEMAS
# ============================================================================

class FoodAliasBase(BaseModel):
    """Food alias schema"""
    food_id: str
    alias: str
    alias_type: str
    language: str
    
    class Config:
        from_attributes = True


# ============================================================================
# SMART SEARCH SCHEMAS
# ============================================================================

class SmartSearchCandidate(BaseModel):
    """A low-confidence candidate returned for clarification"""
    food_id: str
    food_name: str
    food_name_display: str
    confidence: float
    calories_kcal: float
    serving_size_g: float


class SmartSearchResult(BaseModel):
    """Response from /api/foods/smart-search"""
    query_original: str
    query_normalized: str
    match_type: str          # exact_name | exact_alias | variant_strip |
                             # fuzzy_name | fuzzy_alias | low_confidence | no_match
    confidence: float        # 0.0 – 1.0

    # Present on confident matches
    food: Optional[FoodDetail] = None

    # Present on variant_strip matches
    variant_detected: Optional[str] = None
    stripped_query: Optional[str] = None

    # Present on low_confidence / no_match
    candidates: list[SmartSearchCandidate] = []
    message: str = ""


class MultiSearchResponse(BaseModel):
    """Response from /api/foods/multi-search"""
    query: str
    results: list[SmartSearchResult]


# ============================================================================
# NUTRITION CALCULATION SCHEMAS (Day 4)
# ============================================================================

class NutritionCalcRequest(BaseModel):
    """Request for the full food search → quantity → nutrition pipeline."""
    food_query: str = Field(..., min_length=1, description="Food name/phrase (will be smart-searched)")
    quantity: str = Field(
        default="1 serving",
        description="Quantity string, e.g. '150g', '2 bowls', 'half plate', '3 pieces'"
    )
    variant: Optional[str] = Field(
        default=None,
        description="Cooking variant: ghee, fried, boiled, grilled, steamed, baked, roasted, normal"
    )


class NutritionCalcResponse(BaseModel):
    """Full nutrition calculation result."""
    # Food identification
    food_id: str
    food_name: str
    food_name_display: str

    # Search metadata
    search_match_type: str
    search_confidence: float

    # Quantity parsed
    quantity_input: str
    quantity_amount: float
    quantity_unit: str
    quantity_grams: float

    # Variant
    variant: str
    variant_description: str

    # Calculated nutrition
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None

    # Reference
    serving_size_g: float
    calculation_note: str
    overall_confidence: float


class NutritionCalcErrorResponse(BaseModel):
    """Returned when food cannot be confidently identified."""
    food_query: str
    search_match_type: str
    search_confidence: float
    message: str
    candidates: list[SmartSearchCandidate] = []


# ============================================================================
# DAILY LOGGING SCHEMAS (Day 7)
# ============================================================================

class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snack = "snack"


class LogFoodRequest(BaseModel):
    """Request to log a food item consumed."""
    user_id: str = Field(..., min_length=1)
    meal_type: MealType
    food_query: str = Field(..., min_length=1, description="Food name (smart-searched)")
    quantity: str = Field(default="1 serving", description="e.g. '2 bowls', '150g', '3 pieces'")
    variant: Optional[str] = Field(None, description="ghee, fried, boiled, etc.")
    log_date: Optional[str] = Field(None, description="YYYY-MM-DD (defaults to today)")


class LogExerciseRequest(BaseModel):
    """Request to log an exercise performed."""
    user_id: str = Field(..., min_length=1)
    exercise_input: str = Field(..., min_length=1, description="e.g. 'jogging 30 minutes', '50 push-ups'")
    log_date: Optional[str] = Field(None, description="YYYY-MM-DD (defaults to today)")


class FoodLogResponse(BaseModel):
    """Response for a single food log entry."""
    id: str
    user_id: str
    log_date: str
    meal_type: str
    food_id: str
    food_name: str
    food_name_display: str
    quantity_input: str
    quantity_amount: float
    quantity_unit: str
    quantity_grams: float
    variant: Optional[str] = None
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    search_confidence: Optional[float] = None
    created_at: Optional[datetime] = None


class ExerciseLogResponse(BaseModel):
    """Response for a single exercise log entry."""
    id: str
    user_id: str
    log_date: str
    exercise_id: str
    exercise_name: str
    exercise_name_display: str
    category: str
    exercise_input: str
    amount: float
    unit: str
    calories_min: float
    calories_avg: float
    calories_max: float
    search_confidence: Optional[float] = None
    created_at: Optional[datetime] = None


class DailySummaryResponse(BaseModel):
    """Daily nutrition + exercise summary for a user."""
    user_id: str
    date: str
    # Calorie budget
    calorie_target: float
    # Food totals
    total_calories_consumed: float
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    total_fiber_g: float
    # Exercise totals
    total_calories_burned: float
    # Net
    net_calories: float              # consumed - burned
    remaining_calories: float        # target - net_calories
    # Counts
    food_entries: int
    exercise_entries: int
    # Breakdown by meal
    meal_breakdown: dict


class DayLogsResponse(BaseModel):
    """All logs for a specific day."""
    user_id: str
    date: str
    food_logs: list[FoodLogResponse]
    exercise_logs: list[ExerciseLogResponse]


class UpdateFoodLogRequest(BaseModel):
    """Request to update an existing food log entry."""
    meal_type: Optional[MealType] = None
    food_query: Optional[str] = Field(None, min_length=1, description="New food (re-calculated)")
    quantity: Optional[str] = Field(None, description="New quantity")
    variant: Optional[str] = None


class UpdateExerciseLogRequest(BaseModel):
    """Request to update an existing exercise log entry."""
    exercise_input: Optional[str] = Field(None, min_length=1, description="New exercise input (re-calculated)")


class BulkFoodLogRequest(BaseModel):
    """Request to log multiple food items at once."""
    user_id: str = Field(..., min_length=1)
    log_date: Optional[str] = Field(None, description="YYYY-MM-DD (defaults to today)")
    items: list[dict] = Field(..., description="List of {meal_type, food_query, quantity, variant}")


class BulkExerciseLogRequest(BaseModel):
    """Request to log multiple exercises at once."""
    user_id: str = Field(..., min_length=1)
    log_date: Optional[str] = Field(None, description="YYYY-MM-DD (defaults to today)")
    items: list[dict] = Field(..., description="List of {exercise_input}")


class TokenRequest(BaseModel):
    """Login request to get JWT token."""
    user_id: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    expires_in: int = 3600


# ============================================================================
# CHATBOT SCHEMAS (Day 8)
# ============================================================================

class ChatMessageRequest(BaseModel):
    """Request to send a text message to the chatbot."""
    message: str = Field(..., min_length=1, description="User message (English/Hindi/Gujarati)")
    context: Optional[str] = Field(None, description="Previous conversation context")
    auto_log: bool = Field(False, description="If True, skip confirmation and log directly")


class ChatConfirmRequest(BaseModel):
    """Request to confirm a pending chatbot action."""
    confirm: bool = Field(..., description="True to confirm, False to cancel")
    pending_action: dict = Field(..., description="The pending_action object from previous response")


class ChatResponse(BaseModel):
    """Response from the chatbot."""
    message: str
    intent: str
    action_taken: Optional[str] = None
    data: dict = {}
    needs_confirmation: bool = False
    pending_action: Optional[dict] = None
    options: list[str] = []              # Clickable option buttons for the user
    success: bool = True


# ============================================================================
# API RESPONSE SCHEMAS
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    app_name: str
    version: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


class StatsResponse(BaseModel):
    """Database statistics response"""
    total_foods: int
    total_exercises: int
    total_aliases: int
    vegetarian_foods: int
    non_veg_foods: int
    eggetarian_foods: int
    exercise_categories: list[str]
