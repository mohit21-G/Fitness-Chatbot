# Fitness AI Chatbot

Personalized fitness tracking chatbot with voice/text support in English, Hindi, and Gujarati.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start MongoDB (must be running on localhost:27017)
# 3. Import data (first time only)
cd backend
python import_to_mongodb.py

# 4. Run the server
python main.py
```

**Open:** http://localhost:8001/app

## Project Structure

```
fitness chatbot/
├── backend/
│   ├── main.py                  # FastAPI app (all endpoints)
│   ├── database.py              # MongoDB connection + config
│   ├── schemas.py               # Pydantic request/response models
│   ├── food_repository.py       # Food + alias DB queries
│   ├── search_engine.py         # Fuzzy food search (rapidfuzz)
│   ├── quantity_parser.py       # "2 bowls" → (2.0, "bowl")
│   ├── nutrition_calculator.py  # Unit→grams + variant multipliers
│   ├── exercise_calculator.py   # Exercise search + calorie calc
│   ├── user_repository.py       # User profile CRUD
│   ├── bmr_tdee_calculator.py   # Mifflin-St Jeor BMR/TDEE
│   ├── daily_log_repository.py  # Food/exercise log CRUD + summary
│   ├── conversation.py          # Chat memory + multi-turn flow
│   ├── chatbot_engine.py        # AI orchestrator (LLM → search → calc → log)
│   ├── llm_service.py           # Cloudflare Workers AI (Qwen3)
│   ├── stt_service.py           # Groq Whisper Large V3
│   ├── auth.py                  # JWT + bcrypt authentication
│   ├── import_to_mongodb.py     # CSV → MongoDB import script
│   ├── test_mongodb.py          # Integration test suite
│   ├── .env                     # Environment config (secrets)
│   └── .env.example             # Config template
├── frontend/
│   ├── index.html               # Login + Chat UI
│   ├── styles.css               # Responsive CSS
│   ├── api.js                   # API client (JWT auth)
│   └── app.js                   # Chat UI logic + voice
├── Clean dataset/               # Processed CSVs (imported to MongoDB)
├── Raw dataset/                 # Original source CSVs (untouched)
└── requirements.txt             # Python dependencies
```

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URL` | Yes | MongoDB connection string |
| `JWT_SECRET_KEY` | Yes | Random secret for JWT tokens |
| `GROQ_API_KEY` | No | Enables voice input (STT) |
| `CF_ACCOUNT_ID` | No | Enables LLM intent parsing |
| `CF_API_TOKEN` | No | Cloudflare Workers AI token |

Without API keys, the chatbot uses a rule-based fallback parser for text.

## API Documentation

Swagger UI: http://localhost:8001/docs

## Tech Stack

- **Backend:** FastAPI + MongoDB (motor/pymongo)
- **AI:** Groq Whisper (STT) + Cloudflare Qwen3 (LLM)
- **Frontend:** Vanilla HTML/CSS/JS (served by FastAPI)
- **Auth:** JWT + bcrypt
- **Search:** rapidfuzz (fuzzy matching)
