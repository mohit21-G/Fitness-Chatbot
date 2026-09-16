"""
MongoDB Database Connection Manager
Replaces SQLAlchemy with motor (async) + pymongo (sync) for MongoDB.
"""
import os
import logging
from functools import lru_cache

from pydantic_settings import BaseSettings
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Application settings (loaded from .env if present)."""

    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "fitness_chatbot"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001          # overridden by $PORT env var (e.g. on Render)
    API_RELOAD: bool = True

    def __init__(self, **data):
        # Render (and similar platforms) inject $PORT instead of $API_PORT.
        # Apply it here so the rest of the code only references API_PORT.
        import os
        if "PORT" in os.environ and "API_PORT" not in os.environ:
            data.setdefault("API_PORT", int(os.environ["PORT"]))
        super().__init__(**data)

    # Application
    APP_NAME: str = "Fitness Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # JWT
    JWT_SECRET_KEY: str = "fitness-chatbot-secret-key-change-in-production"

    # External APIs
    GROQ_API_KEY: str = ""
    SARVAM_API_KEY: str = ""       # Sarvam Saaras v3 — get at console.sarvam.ai
    CF_ACCOUNT_ID: str = ""
    CF_API_TOKEN: str = ""
    CF_MODEL: str = "@cf/qwen/qwen3-30b-a3b-fp8"
    USDA_API_KEY: str = "DEMO_KEY"
    OPEN_FOOD_FACTS_USER_AGENT: str = "FitnessAIChatbot/1.0 (contact@fitnessai.local)"

    # External Exercise data sources (used only when local DB can't resolve)
    EXERCISEDB_API_KEY: str = ""                       # RapidAPI key for ExerciseDB (optional)
    EXERCISEDB_HOST: str = "exercisedb.p.rapidapi.com"
    WGER_BASE_URL: str = "https://wger.de"             # wger reads are keyless

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# ============================================================================
# MONGODB CONNECTION
# ============================================================================

settings = get_settings()

# Async client (for FastAPI async endpoints)
async_client = AsyncIOMotorClient(settings.MONGODB_URL)
async_db = async_client[settings.MONGODB_DB_NAME]

# Sync client (for import scripts and sync operations)
sync_client = MongoClient(settings.MONGODB_URL)
sync_db = sync_client[settings.MONGODB_DB_NAME]


# ============================================================================
# COLLECTIONS (equivalent to tables)
# ============================================================================

# Async collections (use in API endpoints)
def get_foods_collection():
    return async_db["foods"]

def get_exercises_collection():
    return async_db["exercises"]

def get_food_aliases_collection():
    return async_db["food_aliases"]

def get_users_collection():
    return async_db["user_profiles"]

def get_food_logs_collection():
    return async_db["daily_food_logs"]

def get_exercise_logs_collection():
    return async_db["daily_exercise_logs"]

def get_conversations_collection():
    return async_db["conversation_messages"]


# Sync collections (use in import scripts)
def get_sync_foods():
    return sync_db["foods"]

def get_sync_exercises():
    return sync_db["exercises"]

def get_sync_aliases():
    return sync_db["food_aliases"]

def get_sync_users():
    return sync_db["user_profiles"]


# ============================================================================
# FastAPI DEPENDENCY
# ============================================================================

async def get_db():
    """FastAPI dependency — returns the async MongoDB database instance."""
    return async_db


# ============================================================================
# INDEX CREATION
# ============================================================================

async def create_indexes():
    """Create MongoDB indexes for performance (called on startup)."""
    # Foods
    await async_db["foods"].create_index("food_id", unique=True)
    await async_db["foods"].create_index("food_name")
    await async_db["foods"].create_index("vegetarian_status")
    await async_db["foods"].create_index([("food_name", "text")])

    # Exercises
    await async_db["exercises"].create_index("exercise_id", unique=True)
    await async_db["exercises"].create_index("exercise_name")

    # Aliases
    await async_db["food_aliases"].create_index("alias")
    await async_db["food_aliases"].create_index("food_id")

    # Users
    await async_db["user_profiles"].create_index("user_id", unique=True)

    # Food logs
    await async_db["daily_food_logs"].create_index([("user_id", 1), ("log_date", 1)])

    # Exercise logs
    await async_db["daily_exercise_logs"].create_index([("user_id", 1), ("log_date", 1)])

    # Conversations
    await async_db["conversation_messages"].create_index([("user_id", 1), ("created_at", -1)])

    logger.info("MongoDB indexes created")


# ============================================================================
# STARTUP VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration on startup."""
    warnings = []

    if settings.JWT_SECRET_KEY == "fitness-chatbot-secret-key-change-in-production":
        warnings.append("JWT_SECRET_KEY is using default dev value! Change for production.")

    if not settings.GROQ_API_KEY:
        warnings.append("GROQ_API_KEY not set. Voice endpoint will return 503.")

    if not settings.CF_ACCOUNT_ID or not settings.CF_API_TOKEN:
        warnings.append("CF_ACCOUNT_ID/CF_API_TOKEN not set. Using fallback parser.")

    # Test MongoDB connection
    try:
        sync_client.admin.command("ping")
        logger.info(f"MongoDB connected: {settings.MONGODB_URL}/{settings.MONGODB_DB_NAME}")
    except Exception as e:
        warnings.append(f"MongoDB connection FAILED: {e}")

    for w in warnings:
        logger.warning(f"[CONFIG] {w}")

    return warnings
