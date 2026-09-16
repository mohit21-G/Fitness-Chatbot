"""MongoDB migration verification test."""
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8001"
passed = 0
total = 0


def check(name, ok, detail=""):
    global passed, total
    total += 1
    mark = "+" if ok else "x"
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"  {mark} {name:45} {status}  {str(detail)[:40]}")


def get(path, token=None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(f"{BASE}{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            return {"status": r.status, "data": json.loads(r.read())}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode()[:150]}
    except Exception as e:
        return {"status": 0, "error": str(e)[:150]}


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=20) as r:
            return {"status": r.status, "data": json.loads(r.read())}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode()[:150]}
    except Exception as e:
        return {"status": 0, "error": str(e)[:150]}


print("=" * 78)
print("  MONGODB MIGRATION VERIFICATION")
print("=" * 78)

# --- Core ---
print("\n  A. Core Endpoints")
print("-" * 74)
r = get("/health")
check("Health check", r["status"] == 200 and r["data"]["status"] == "healthy", "healthy")

r = get("/stats")
ok = r["status"] == 200 and r["data"]["total_foods"] > 2000
check("Stats (MongoDB counts)", ok, f"{r['data'].get('total_foods',0)} foods, {r['data'].get('total_exercises',0)} ex")

# --- Food Search ---
print("\n  B. Food Search (MongoDB)")
print("-" * 74)
for q, expected in [("butter chicken", "butter chicken"), ("paneer", "paneer"),
                    ("BHAKHARI", "bhakri"), ("doodh", "milk")]:
    r = get(f"/api/foods/smart-search?query={urllib.request.quote(q)}")
    food = r["data"].get("food") or {}
    check(f"Search: {q}", r["status"] == 200 and expected in food.get("food_name", ""),
          f"{food.get('food_name','?')} ({r['data'].get('match_type','?')})")

# --- Nutrition ---
print("\n  C. Nutrition Calculator")
print("-" * 74)
for food, qty, cmin, cmax in [("rice", "200g", 230, 300), ("idli", "3 pieces", 80, 200),
                               ("bhakri", "2 pieces", 200, 300)]:
    r = post("/api/foods/calculate-nutrition", {"food_query": food, "quantity": qty})
    cal = r["data"].get("calories", 0) if r["status"] == 200 else 0
    check(f"Nutrition: {food} {qty}", cmin <= cal <= cmax, f"{cal:.0f} kcal")

# --- Exercise ---
print("\n  D. Exercise Calculator")
print("-" * 74)
for inp, cmin, cmax in [("jogging 30 minutes", 200, 400), ("50 push-ups", 10, 40),
                        ("cycling 45 min", 250, 450)]:
    r = post("/api/exercises/smart-calculate", {"exercise_input": inp})
    cal = r["data"].get("calories_avg", 0) if r["status"] == 200 else 0
    check(f"Exercise: {inp}", cmin <= cal <= cmax, f"{cal:.0f} kcal")

# --- User + Auth ---
print("\n  E. User Profile + Auth")
print("-" * 74)
user = {
    "user_id": "mongo_test", "name": "Mongo Tester", "age": 28, "gender": "male",
    "height_cm": 175, "weight_kg": 75, "activity_level": "moderate",
    "fitness_goal": "lose_weight", "diet_type": "non_veg", "password": "Test@123"
}
r = post("/api/users", user)
check("Create user", r["status"] in (201, 409), f"HTTP {r['status']}")

r = post("/api/auth/login", {"user_id": "mongo_test", "password": "Test@123"})
check("Login (bcrypt)", r["status"] == 200, f"HTTP {r['status']}")
token = r["data"].get("access_token", "") if r["status"] == 200 else ""

r = get("/api/users/mongo_test/calorie-target")
check("BMR/TDEE calculation", r["status"] == 200 and r["data"].get("bmr", 0) > 1500,
      f"BMR={r['data'].get('bmr',0):.0f} TDEE={r['data'].get('tdee',0):.0f}")

r = get("/api/auth/me", token=token)
check("JWT /me endpoint", r["status"] == 200, f"user={r['data'].get('name','?')}")

# --- Logging ---
print("\n  F. Daily Logging")
print("-" * 74)
r = post("/api/logs/food", {"user_id": "mongo_test", "meal_type": "breakfast",
                             "food_query": "idli", "quantity": "3 pieces"})
check("Log food", r["status"] == 201, f"{r['data'].get('calories',0):.0f} kcal")
food_log_id = r["data"].get("id") if r["status"] == 201 else None

r = post("/api/logs/exercise", {"user_id": "mongo_test", "exercise_input": "jogging 30 minutes"})
check("Log exercise", r["status"] == 201, f"{r['data'].get('calories_avg',0):.0f} kcal")

r = get("/api/logs/mongo_test/today")
check("Get today's logs", r["status"] == 200 and len(r["data"]["food_logs"]) >= 1,
      f"{len(r['data'].get('food_logs',[]))} food, {len(r['data'].get('exercise_logs',[]))} ex")

from datetime import date
today = date.today().isoformat()
r = get(f"/api/logs/mongo_test/summary/{today}")
ok = r["status"] == 200 and r["data"]["calorie_target"] > 0
check("Daily summary", ok, f"target={r['data'].get('calorie_target',0):.0f} net={r['data'].get('net_calories',0):.0f}")

if food_log_id:
    r = get(f"/api/logs/food/{food_log_id}?user_id=mongo_test")  # will 405, use delete
    import urllib.request as ur
    req = ur.Request(f"{BASE}/api/logs/food/{food_log_id}?user_id=mongo_test", method="DELETE")
    try:
        with ur.urlopen(req) as resp:
            check("Delete food log", resp.status == 200, "deleted")
    except Exception as e:
        check("Delete food log", False, str(e)[:30])

# --- Chatbot ---
print("\n  G. Chatbot (multi-turn + multilingual)")
print("-" * 74)
r = post("/api/chat/message", {"message": "kem cho"}, token=token)
check("Gujarati greeting", r["status"] == 200 and r["data"]["intent"] == "greeting",
      r["data"].get("message", "")[:35] if r["status"] == 200 else "")

r = post("/api/chat/message", {"message": "2 idli breakfast", "auto_log": True}, token=token)
check("Chat food logging", r["status"] == 200 and r["data"].get("action_taken") == "food_logged",
      r["data"].get("message", "")[:35] if r["status"] == 200 else "")

r = post("/api/chat/message", {"message": "aaj ka summary dikhao"}, token=token)
check("Chat summary", r["status"] == 200 and r["data"]["intent"] == "get_summary",
      f"consumed={r['data'].get('data',{}).get('consumed',0):.0f}")

r = post("/api/chat/message", {"message": "rice calories"}, token=token)
check("Chat calorie query", r["status"] == 200 and r["data"]["intent"] == "get_calories",
      f"cal={r['data'].get('data',{}).get('calories',0):.0f}")

r = post("/api/chat/message", {"message": "hello"})
check("No auth -> 401", r["status"] == 401, f"HTTP {r['status']}")

# --- Frontend ---
print("\n  H. Frontend")
print("-" * 74)
try:
    resp = urllib.request.urlopen(f"{BASE}/app")
    html = resp.read().decode()
    check("Frontend /app loads", "Fitness AI Chatbot" in html, f"{len(html)} bytes")
except Exception as e:
    check("Frontend /app loads", False, str(e)[:30])

try:
    resp = urllib.request.urlopen(f"{BASE}/static/app.js")
    check("Static files served", resp.status == 200, "app.js OK")
except Exception as e:
    check("Static files served", False, str(e)[:30])

# --- Summary ---
print()
print("=" * 78)
print(f"  RESULT: {passed}/{total} PASS ({passed/total*100:.0f}%)")
print("=" * 78)
