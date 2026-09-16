# Fitness AI Chatbot

Personalized fitness tracking chatbot with voice/text support in English, Hindi, and Gujarati.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MongoDB
Make sure MongoDB is running on `localhost:27017` (or your Atlas URL is set in `backend/.env`).

### 3. Import data (first time only)

```bash
cd backend
python import_to_mongodb.py
```

---

## Running the Services

The backend and frontend start independently. Open **two terminals**.

### Terminal 1 — Backend (FastAPI)

```bash
cd backend
python main.py
```

Or with uvicorn directly:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Backend API: http://localhost:8001  
Swagger docs: http://localhost:8001/docs

> The backend also serves the frontend at **http://localhost:8001/app** — use this if you don't want to run a separate frontend server.

---

### Terminal 2 — Frontend (standalone dev server)

```bash
cd frontend
npm run dev
```

On first run, `serve` will be downloaded automatically via `npx` (no global install needed).

Frontend: http://localhost:3000

> The frontend auto-detects it is running on port 3000 and points all API calls to `http://localhost:8001`. No manual configuration needed.

---

## Modes at a Glance

| Mode | How to open | API target |
|------|-------------|------------|
| Backend-served (single command) | `python main.py` → http://localhost:8001/app | Same origin (8001) |
| Standalone frontend | `npm run dev` → http://localhost:3000 | http://localhost:8001 |

---

## Custom Backend URL

If your backend runs on a different host or port, set an override in the browser console before loading the page:

```js
localStorage.setItem('FITNESS_API_BASE', 'http://your-host:8001')
```

Then refresh the page. The frontend will use that URL for all API calls.

---

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
│   ├── import_to_mongodb.py     # CSV → MongoDB import script
│   ├── .env                     # Environment config (secrets)
│   └── .env.example             # Config template
├── frontend/
│   ├── index.html               # Chat UI
│   ├── styles.css               # Responsive CSS
│   ├── api.js                   # API client (auto-detects backend URL)
│   ├── app.js                   # Chat UI logic + voice
│   └── package.json             # Dev server (npx serve)
├── Clean dataset/               # Processed CSVs (imported to MongoDB)
├── Raw dataset/                 # Original source CSVs (untouched)
└── requirements.txt             # Python dependencies
```

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URL` | Yes | MongoDB connection string |
| `API_HOST` | No | Bind address (default: `0.0.0.0`) |
| `API_PORT` | No | Backend port (default: `8001`) |
| `GROQ_API_KEY` | No | Enables voice input (STT) |
| `CF_ACCOUNT_ID` | No | Enables LLM intent parsing |
| `CF_API_TOKEN` | No | Cloudflare Workers AI token |

Without API keys, the chatbot uses a rule-based fallback parser for text.

## Tech Stack

- **Backend:** FastAPI + MongoDB (motor/pymongo)
- **AI:** Groq Whisper (STT) + Cloudflare Qwen3 (LLM)
- **Frontend:** Vanilla HTML/CSS/JS
- **Search:** rapidfuzz (fuzzy matching)
