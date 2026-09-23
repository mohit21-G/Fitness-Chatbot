import sys
from test_comprehensive_2500_benchmark import build_benchmark_cases
from llm_service import LLMService

cases = build_benchmark_cases()
llm = LLMService()
failures = []
for c in cases:
    parsed = llm.parse_intent_sync(c.message)
    reasons = []
    if parsed.intent != c.expected_intent:
        reasons.append(f"intent expected={c.expected_intent} got={parsed.intent}")
    if c.expect_duration_none and parsed.duration_min is not None:
        reasons.append(f"duration expected=None got={parsed.duration_min}")
    elif c.expected_duration_min is not None and parsed.duration_min != c.expected_duration_min:
        reasons.append(f"duration expected={c.expected_duration_min} got={parsed.duration_min}")
    if c.expected_min_reps is not None and (parsed.reps or 0) < c.expected_min_reps:
        reasons.append(f"reps expected>={c.expected_min_reps} got={parsed.reps}")
    
    if reasons:
        failures.append((c.case_id, c.category, c.message, reasons))

with open("scratch_failures.txt", "w", encoding="utf-8") as f:
    f.write(f"Total failures: {len(failures)}\n")
    for fid, cat, msg, r in failures:
        f.write(f"[{cat}] #{fid} '{msg}' -> {', '.join(r)}\n")

print(f"Wrote {len(failures)} failures to scratch_failures.txt")
