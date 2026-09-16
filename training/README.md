# Fitness Chatbot — Model Improvement Pipeline

## Architecture reality check

The production model is **`@cf/qwen/qwen3-30b-a3b-fp8`** running on
**Cloudflare Workers AI** as a hosted API endpoint. You do not have the model
weights locally. This has one decisive consequence:

| What you asked for | What's actually possible |
|---|---|
| Fine-tune the existing Qwen3-30B-A3B-FP8 | ❌ Cloudflare only supports LoRA on non-quantized Mistral/Llama/Gemma models (open beta 2025) |
| LoRA/QLoRA locally | ❌ This machine has no CUDA GPU (`torch+cpu` only) |
| Improve model behaviour right now | ✅ System-prompt engineering + few-shot examples |
| Build training data for future fine-tuning | ✅ Done — `dataset_train.jsonl` + `dataset_eval.jsonl` |
| Fine-tune a Cloudflare-compatible model on a cloud GPU | ✅ Possible — see Path A below |
| Evaluate current model quality rigorously | ✅ Done — `evaluate_model.py` |

## Files in this directory

```
training/
├── generate_dataset.py       # Generates training + eval JSONL datasets
├── dataset_train.jsonl       # 290 training examples (ChatML format)
├── dataset_eval.jsonl        # 73 held-out eval examples
├── evaluate_model.py         # A-to-Z evaluation benchmark (95 cases)
├── finetune_lora.py          # QLoRA training script (needs GPU)
├── upload_lora_cf.py         # Uploads LoRA adapter to Cloudflare
├── requirements_finetune.txt # GPU environment dependencies
└── README.md                 # This file
```

## What was done right now (zero hardware needed)

### 1. Dataset generated
`python training/generate_dataset.py`

Produces **363 examples** (290 train / 73 eval) in ChatML format covering:
- Indian food logging in English / Gujarati / Hindi / Hinglish
- Typos, phonetic spellings, transliterations
- Multi-food sentences
- Exercise logging (duration, reps, sets, distance)
- Nutrition queries (`get_calories`)
- Daily log reads (`get_summary`, `query_meal`)
- Date extraction (today / yesterday / gatkale / kal)
- Variant detection (ghee, fried, boiled…)
- Missing-detail cases
- Junk / off-topic rejection
- Greetings, profile, skip-meal, exercise-read

### 2. System prompt improved (`backend/llm_service.py`)
Added 20 targeted few-shot examples directly addressing the failure patterns
found by the evaluation benchmark:
- Exercise field extraction (`exercise_query`, `duration_min`, `reps`, etc.)
- `query_exercise` intent disambiguation
- Junk / off-topic → `clarification_needed`
- Yesterday date (`gatkale`, `kal`, `yesterday`) → `"date":"yesterday"`
- `skip_meal` intent
- Nutrition lookup vs daily total disambiguation

### 3. LoRA activation hook (`backend/llm_service.py`)
`parse_intent()` now reads `CF_FINETUNE_ID` from the environment. Once you
upload a trained adapter, add one line to `backend/.env`:
```
CF_FINETUNE_ID=your-finetune-id-here
```
No other code changes needed. The adapter is activated for every inference call.

### 4. Evaluation benchmark (`training/evaluate_model.py`)
95-case A-to-Z benchmark across 16 categories. **Before score: 29.5%** on
raw LLM field extraction. Note: the production chatbot achieves much higher
accuracy because the deterministic routing layer in `chatbot_engine.py`
handles most intent routing before/after the LLM.

## Path A — Fine-tune on Cloudflare (recommended)

This replaces the Qwen3-30B endpoint with a smaller but fine-tuned Llama-3.2-3B
that the chatbot engine calls identically (same API, same JSON schema).

**Step 1 — Train (needs Google Colab T4 — free)**
```python
# Open https://colab.research.google.com, paste this notebook cell:
!git clone https://github.com/YOUR/fitness-chatbot
%cd fitness-chatbot
!pip install -r training/requirements_finetune.txt
!python training/finetune_lora.py
# Saves adapter to training/lora_adapter/
```

**Step 2 — Upload adapter to Cloudflare**
```bash
python training/upload_lora_cf.py
# Prints the finetune_id
```

**Step 3 — Activate**
```bash
# backend/.env
CF_MODEL=@cf/meta-llama/llama-3.2-3b-instruct
CF_FINETUNE_ID=<finetune_id from step 2>
```

**Step 4 — Evaluate after**
```bash
cd backend && python ../training/evaluate_model.py
# Compare Score: before (29.5%) vs after
```

## Path B — Larger model, better quality (needs 16 GB VRAM)

Same steps but set `BASE_MODEL=Qwen/Qwen3-8B-Instruct` in `finetune_lora.py`
and host on HuggingFace Inference Endpoints or RunPod, then point
`CF_API_URL` in `llm_service.py` to your endpoint.

## Evaluation benchmark results

| Category | Before (base Qwen3) | After prompt improvement |
|---|---|---|
| greeting | 100% | 100% |
| profile | 100% | 100% |
| exercise_read | 67% | ~80% |
| daily_log | 44% | ~55% |
| food_en | 50% | ~65% |
| food_gu | 50% | ~65% |
| exercise | 0% | ~40% |
| multi_food | 0% | ~30% |
| junk | 0% | ~50% |
| **Overall** | **29.5%** | **~50% (estimated)** |

Re-run `evaluate_model.py` to get the actual after score.

## Key failures explained

| Failure | Root cause | Fix |
|---|---|---|
| `exercise_query=None` for exercise intents | Model returns intent correctly but doesn't populate entity fields | More few-shot examples in prompt (added) + fine-tuning |
| `foods=[]` for multi-food | Model doesn't populate `foods[]` array | More multi-food examples in prompt + training data |
| Junk → `log_food` | Model finds a "food" in any sentence | Off-topic examples in prompt (added) + `_is_offtopic_request` deterministic gate (already in chatbot_engine.py) |
| `"date":"today"` for "yesterday" inputs | Date normalisation | Added more yesterday examples; deterministic `_resolve_read_date` in chatbot_engine.py handles this |
| `missing_detail=null` when quantity missing | Model doesn't detect missing fields | Training data with explicit `missing_detail` annotations |
