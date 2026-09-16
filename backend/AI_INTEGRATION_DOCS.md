# Fitness AI Chatbot — Integration Documentation

## For Web Team: API Integration Guide

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     FRONTEND (Web/App)                        │
│  Voice Input → Audio blob                                    │
│  Text Input  → String                                        │
└───────────────┬──────────────────────────────────────────────┘
                │ HTTP (JWT Bearer Token)
                ▼
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                          │
│                                                              │
│  POST /api/chat/message  ← Text chatbot                     │
│  POST /api/chat/voice    ← Voice chatbot                    │
│  POST /api/chat/confirm  ← Confirm pending action           │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  Groq STT   │  │ CF Qwen3 LLM │  │ Fallback Parser │    │
│  │  (Whisper)  │  │ (Intent)     │  │ (Rule-based)    │    │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘    │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Chatbot Engine (Orchestrator)               │   │
│  │  Multi-turn Flow → Search → Calculate → Log          │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Existing APIs: Food Search, Nutrition Calc,          │   │
│  │  Exercise Calc, User Profile, Daily Logs, BMR/TDEE   │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │   SQLite DB  │  (2100+ foods, 55 exercises, user data)   │
│  └─────────────┘                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Authentication

### Login (get JWT token)
```http
POST /api/auth/login
Content-Type: application/json

{
  "user_id": "user_001",
  "password": "fitness123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user_id": "user_001",
  "expires_in": 86400
}
```

### Use token in all chatbot requests:
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

---

## 3. Chatbot Endpoints

### 3.1 Text Message

```http
POST /api/chat/message
Authorization: Bearer {token}
Content-Type: application/json

{
  "message": "maine breakfast me 3 idli khaayi with ghee",
  "auto_log": false
}
```

**Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | User text (English/Hindi/Gujarati) |
| context | string | No | Previous context (auto-managed via memory) |
| auto_log | bool | No | `true` = skip confirmation, log directly |

**Response:**
```json
{
  "message": "Tame 3 pieces Idli (ghee) khadhi. Approx 160 calories...\nShu aa tamara breakfast daily log ma save karu?",
  "intent": "log_food",
  "action_taken": null,
  "data": {"nutrition": {...}},
  "needs_confirmation": true,
  "pending_action": {"type": "log_food", "food_id": "...", "calories": 160.4, ...},
  "success": true
}
```

### 3.2 Confirm Action

When `needs_confirmation: true`, frontend shows confirm/cancel buttons:

```http
POST /api/chat/confirm
Authorization: Bearer {token}
Content-Type: application/json

{
  "confirm": true,
  "pending_action": { /* from previous response's pending_action */ }
}
```

**Response:**
```json
{
  "message": "Done! Idli (160 kcal) saved to your breakfast.",
  "intent": "log_food",
  "action_taken": "food_logged",
  "success": true
}
```

### 3.3 Voice Message

```http
POST /api/chat/voice
Authorization: Bearer {token}
Content-Type: multipart/form-data

audio: [binary audio file]
language: "gu" (optional: en, hi, gu)
auto_log: false (optional)
```

**Response:** Same as text message, plus:
```json
{
  "data": {
    "transcribed_text": "bhakri ghee vali 2",
    "stt_language": "gu",
    ...
  }
}
```

---

## 4. Conversation Flow (Multi-turn)

The chatbot asks clarification questions when info is missing:

```
Frontend sends: "bhakri"
Bot responds:   "Bhakri — ghee vali ke normal?" (needs_confirmation: false)

Frontend sends: "ghee vali"
Bot responds:   "Bhakri ketli khidhi?" (needs_confirmation: false)

Frontend sends: "2 pieces"
Bot responds:   "Breakfast, lunch, dinner ke snack?" (needs_confirmation: false)

Frontend sends: "breakfast"
Bot responds:   "2 pieces Bhakri (ghee) = 312 cal. Save karu?" (needs_confirmation: true)

Frontend sends: confirm via /api/chat/confirm
Bot responds:   "Done! Bhakri (312 kcal) saved to your breakfast."
```

**Frontend Implementation Notes:**
- When `needs_confirmation: true` → show Yes/No buttons
- When `needs_confirmation: false` → show text input for user's answer
- The bot automatically detects language (Gujarati/Hindi/English) and responds accordingly
- Each message builds on conversation memory (no need to send context manually)

---

## 5. Supported Intents

| Intent | Triggered By | Example |
|--------|-------------|---------|
| `greeting` | hi/hello/namaste/kem cho | "kem cho" |
| `log_food` | food mentions | "3 idli breakfast me" |
| `log_exercise` | exercise mentions + duration | "jogging 30 minutes" |
| `get_summary` | summary keywords | "aaj ka summary" |
| `get_calories` | calorie/nutrition questions | "paneer me kitni calories" |
| `get_profile` | profile keywords | "show my profile" |
| `clarification_needed` | unclear input | "kuch khaya" |

---

## 6. Supported Languages

| Language | Input | Response | Voice (STT) |
|----------|-------|----------|-------------|
| English | ✅ | ✅ | ✅ |
| Hindi (Romanized) | ✅ | ✅ | ✅ |
| Gujarati (Romanized) | ✅ | ✅ | ✅ |
| Hindi (Devanagari) | ✅ | ✅ | ✅ |
| Gujarati (Script) | ✅ | ✅ | ✅ |

Auto-detection based on script characters and keywords.

---

## 7. Environment Setup

Copy `.env.example` to `.env` and configure:

```env
# Required for basic operation
DATABASE_URL=sqlite:///./fitness_chatbot.db
JWT_SECRET_KEY=your-production-secret-here

# Required for voice (optional — text chat works without this)
GROQ_API_KEY=gsk_your_groq_key

# Required for LLM (optional — fallback parser works without this)
CF_ACCOUNT_ID=your_cloudflare_account_id
CF_API_TOKEN=your_cloudflare_api_token
CF_MODEL=@cf/qwen/qwen2.5-coder-32b-instruct
```

**Without API keys:** Text chat works with rule-based fallback parser.
**Without GROQ_API_KEY:** Voice endpoint returns 503.

---

## 8. Error Handling (Frontend)

| HTTP Code | Meaning | Frontend Action |
|-----------|---------|----------------|
| 200 | Success | Display bot message |
| 201 | Created | Log entry created |
| 401 | Unauthorized | Redirect to login |
| 422 | Validation/Not Found | Show error message from response |
| 503 | STT not configured | Show "Voice unavailable" |

**Response fields to check:**
- `success: false` → show error state
- `needs_confirmation: true` → show confirm/cancel UI
- `action_taken: "food_logged"` → show success animation
- `intent: "clarification_needed"` → show text input prompt

---

## 9. Complete API List (32+ endpoints)

### Health
- `GET /` — Health check
- `GET /health` — Health status
- `GET /stats` — Database statistics

### Food Search
- `GET /api/foods/search` — Text search with filters
- `GET /api/foods/smart-search?query=` — Fuzzy + alias + variant search
- `GET /api/foods/multi-search?query=` — Top-N candidates
- `GET /api/foods/{food_id}` — Get by ID
- `GET /api/foods/by-name/{name}` — Get by name
- `GET /api/foods/alias/{alias}` — Search by alias
- `POST /api/foods/calculate-portion` — Portion calculation
- `POST /api/foods/calculate-nutrition` — Full nutrition pipeline

### Exercise
- `GET /api/exercises/search` — Search exercises
- `GET /api/exercises/categories` — List categories
- `GET /api/exercises/{exercise_id}` — Get by ID
- `GET /api/exercises/by-name/{name}` — Get by name
- `POST /api/exercises/calculate-calories` — Basic calculation
- `POST /api/exercises/smart-calculate` — Natural language calculation

### User Profile
- `POST /api/users` — Create profile
- `GET /api/users` — List all
- `GET /api/users/{user_id}` — Get profile
- `PUT /api/users/{user_id}` — Update profile
- `DELETE /api/users/{user_id}` — Delete profile
- `GET /api/users/{user_id}/calorie-target` — BMR/TDEE/targets

### Daily Logging
- `POST /api/logs/food` — Log food
- `POST /api/logs/exercise` — Log exercise
- `POST /api/logs/food/bulk` — Bulk log food
- `POST /api/logs/exercise/bulk` — Bulk log exercise
- `PUT /api/logs/food/{id}` — Update food log
- `PUT /api/logs/exercise/{id}` — Update exercise log
- `DELETE /api/logs/food/{id}` — Delete food log
- `DELETE /api/logs/exercise/{id}` — Delete exercise log
- `GET /api/logs/{user_id}/today` — Today's logs
- `GET /api/logs/{user_id}/date/{date}` — Logs by date
- `GET /api/logs/{user_id}/range` — Logs by range (paginated)
- `GET /api/logs/{user_id}/summary/{date}` — Daily summary

### Authentication
- `POST /api/auth/login` — Get JWT token
- `GET /api/auth/me` — Current user (protected)

### Chatbot
- `POST /api/chat/message` — Text chat (protected)
- `POST /api/chat/confirm` — Confirm action (protected)
- `POST /api/chat/voice` — Voice chat (protected)

---

## 10. Testing Results

| Test Suite | Cases | Pass | Rate |
|-----------|-------|------|------|
| Food Search (Day 3) | 50 | 50 | 100% |
| Nutrition Calc (Day 4) | 50 | 50 | 100% |
| Day 5 Regression | 50 | 50 | 100% |
| User Profile (Day 6) | 50 | 50 | 100% |
| Daily Logging (Day 7) | 50 | 49 | 98% |
| Chatbot (Day 8) | 25 | 25 | 100% |
| E2E Chatbot | 35 | 35 | 100% |
| **Total** | **310** | **309** | **99.7%** |

---

## 11. Safety Notes

- All calorie/nutrition values are **estimated** from database records
- Bot appends disclaimer: "These are estimated values, not medical advice"
- The chatbot is a **fitness tracking assistant**, not a medical system
- LLM never invents calorie values — all data comes from the database
- API keys are stored only in backend `.env` — never exposed to frontend

---

## 12. Remaining Known Limitations

1. **Fallback parser** handles ~80% of common patterns; with Qwen3 API key it reaches ~95%+
2. **Multi-food** in one message splits on "aur/and/," — complex sentences may need LLM
3. **Voice accuracy** depends on audio quality and Whisper model performance for regional accents
4. **Session memory** keeps last 10 messages — long conversations may lose early context
5. **No real-time updates** — frontend should poll/refresh for latest data

---

## 13. Quick Start for Frontend Developer

```javascript
// 1. Login
const loginRes = await fetch('/api/auth/login', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({user_id: 'user_001', password: 'fitness123'})
});
const {access_token} = await loginRes.json();

// 2. Send chat message
const chatRes = await fetch('/api/chat/message', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({message: 'bhakri', auto_log: false})
});
const botResponse = await chatRes.json();

// 3. Handle response
if (botResponse.needs_confirmation) {
  // Show confirm/cancel UI
  // On confirm: POST /api/chat/confirm with pending_action
} else if (botResponse.action_taken) {
  // Show success message
} else {
  // Show bot message, await next user input
}

// 4. Voice (optional)
const formData = new FormData();
formData.append('audio', audioBlob, 'recording.webm');
const voiceRes = await fetch('/api/chat/voice?auto_log=false', {
  method: 'POST',
  headers: {'Authorization': `Bearer ${access_token}`},
  body: formData
});
```

---

*Generated: 2026-08-21 | Swagger Docs: http://localhost:8001/docs*
