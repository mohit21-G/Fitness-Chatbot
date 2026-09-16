"""
finetune_lora.py
=================
QLoRA fine-tuning script for the Fitness Chatbot intent parser.

!! READ BEFORE RUNNING !!
---------------------------------------------------------------------------
The CURRENT production model (@cf/qwen/qwen3-30b-a3b-fp8) is a Cloudflare
hosted quantized endpoint. Cloudflare Workers AI only supports custom LoRA
adapters on NON-quantized Mistral / Llama / Gemma family models (open beta
as of 2025).

Therefore, this script targets TWO upgrade paths:

PATH A — Train a LoRA adapter for a Cloudflare-compatible model and use it
          as an exact drop-in on Workers AI:
          Base model: meta-llama/Llama-3.2-3B-Instruct  (fits in 16 GB VRAM)
          CF endpoint: @cf/meta-llama/llama-3.2-3b-instruct (LoRA supported)

PATH B — Train a larger, higher-quality adapter locally (if more GPU RAM
          is available) and host on HuggingFace / runpod / modal:
          Base model: Qwen/Qwen3-8B-Instruct  (needs 16 GB VRAM with QLoRA)

Hardware requirements:
  Path A: 8 GB VRAM (consumer GPU)    or Google Colab T4 (free)
  Path B: 16 GB VRAM                  or Colab A100 (paid)
  CPU-only: NOT supported — QLoRA requires a CUDA-capable GPU.

This machine currently has NO NVIDIA GPU. To run the training:
  1. Push this repo to GitHub
  2. Open in Google Colab or RunPod
  3. Run:  pip install -r training/requirements_finetune.txt
  4. Run:  python training/finetune_lora.py

The trained adapter files will be saved to:
  training/lora_adapter/adapter_model.safetensors
  training/lora_adapter/adapter_config.json

Then upload them to Cloudflare via training/upload_lora_cf.py.
---------------------------------------------------------------------------

Dependencies (install in the GPU environment):
    pip install unsloth datasets trl transformers peft bitsandbytes accelerate
"""
from __future__ import annotations

import json
import os

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
# PATH A — Cloudflare-compatible base model (8 GB VRAM, free Colab T4)
BASE_MODEL_PATH_A = "meta-llama/Llama-3.2-3B-Instruct"
CF_TARGET_MODEL_A = "@cf/meta-llama/llama-3.2-3b-instruct"

# PATH B — Higher quality (16 GB VRAM)
BASE_MODEL_PATH_B = "Qwen/Qwen3-8B-Instruct"

# Choose your path:
BASE_MODEL = os.getenv("BASE_MODEL", BASE_MODEL_PATH_A)

TRAIN_FILE   = os.path.join(os.path.dirname(__file__), "dataset_train.jsonl")
EVAL_FILE    = os.path.join(os.path.dirname(__file__), "dataset_eval.jsonl")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "lora_adapter")

LORA_RANK    = 16     # r — higher = more capacity, more memory
LORA_ALPHA   = 32     # scaling factor — usually 2×rank
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 1024   # prompt + JSON response fits easily
BATCH_SIZE   = 4
GRAD_ACCUM   = 4      # effective batch = 4×4 = 16
LR           = 2e-4
EPOCHS       = 3
WARMUP_RATIO = 0.05

TRAIN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPU detected. Fine-tuning requires a CUDA-capable GPU.")
            print()
            print("Options:")
            print("  1. Run on Google Colab (free T4 GPU): https://colab.research.google.com")
            print("  2. Use RunPod / Modal / Lambda Labs with a 16 GB+ GPU")
            print("  3. Train on a cloud VM with NVIDIA A100 (recommended)")
            print()
            print("This machine runs CPU-only PyTorch (torch+cpu).  Exiting.")
            raise SystemExit(1)
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        raise SystemExit(1)


def load_dataset(path: str):
    """Load ChatML JSONL into HuggingFace Dataset."""
    from datasets import Dataset  # type: ignore
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def format_chatml(example: dict) -> dict:
    """Convert our ChatML messages to a single text string for training."""
    messages = example["messages"]
    text_parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return {"text": "\n".join(text_parts)}


def main():
    check_gpu()

    print("=" * 60)
    print("  Fitness Chatbot LoRA Fine-tuning")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Train data : {TRAIN_FILE}")
    print(f"  Output     : {OUTPUT_DIR}")
    print("=" * 60)

    # ── Imports (only available in GPU environment) ─────────────────────────
    try:
        from unsloth import FastLanguageModel  # type: ignore
        from trl import SFTTrainer, SFTConfig  # type: ignore
        UNSLOTH = True
    except ImportError:
        UNSLOTH = False
        from transformers import (  # type: ignore
            AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model  # type: ignore
        from trl import SFTTrainer, SFTConfig  # type: ignore

    # ── Load datasets ────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = load_dataset(TRAIN_FILE)
    eval_ds  = load_dataset(EVAL_FILE)
    train_ds = train_ds.map(format_chatml)
    eval_ds  = eval_ds.map(format_chatml)
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading model (4-bit QLoRA)...")
    if UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,  # auto
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=TRAIN_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        import torch  # type: ignore
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        )
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=TRAIN_TARGET_MODULES, bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ── Trainer ──────────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # ── Save adapter ─────────────────────────────────────────────────────────
    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Patch adapter_config.json for Cloudflare (requires model_type field)
    cfg_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Cloudflare requires model_type to be mistral | gemma | llama
        cfg["model_type"] = "llama"   # llama-3.2 is llama family
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("Patched adapter_config.json with model_type=llama (Cloudflare requirement).")

    print()
    print("Training complete. Next steps:")
    print(f"  1. Check adapter files: {OUTPUT_DIR}/adapter_model.safetensors")
    print(f"                          {OUTPUT_DIR}/adapter_config.json")
    print("  2. Upload to Cloudflare: python training/upload_lora_cf.py")
    print(f"  3. Target CF model: {CF_TARGET_MODEL_A}")


if __name__ == "__main__":
    main()
