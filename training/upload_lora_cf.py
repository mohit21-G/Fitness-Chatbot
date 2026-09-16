"""
upload_lora_cf.py
==================
Upload a trained LoRA adapter to Cloudflare Workers AI and integrate it
with the production chatbot.

After successful upload, the chatbot's llm_service.py is patched to pass
the finetune_id in every inference request.

Usage:
    python training/upload_lora_cf.py [--name my-fitness-lora] [--model @cf/...]

Requirements:
  • Cloudflare API Token with Workers AI: Edit permissions
  • Trained adapter files at training/lora_adapter/
    - adapter_model.safetensors
    - adapter_config.json  (must contain "model_type": "llama" or "mistral")
  • Cloudflare-compatible base model — see finetune_lora.py for supported list

Cloudflare LoRA constraints (open beta, 2025):
  • Only Mistral / Llama / Gemma family models (NON-quantized)
  • LoRA rank must be <= 32
  • adapter_model.safetensors must be < 300 MB
  • Max 100 LoRA adapters per account
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def load_env():
    """Load CF credentials from backend/.env."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, "backend", ".env")
    env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fitness-chatbot-lora",
                        help="Name of the fine-tune on Cloudflare")
    parser.add_argument("--model",
                        default="@cf/meta-llama/llama-3.2-3b-instruct",
                        help="Cloudflare model that supports LoRA")
    parser.add_argument("--adapter-dir",
                        default=os.path.join(os.path.dirname(__file__), "lora_adapter"),
                        help="Directory with adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate files and print curl commands without calling the API")
    args = parser.parse_args()

    env = load_env()
    account_id = env.get("CF_ACCOUNT_ID") or os.getenv("CF_ACCOUNT_ID")
    api_token  = env.get("CF_API_TOKEN")  or os.getenv("CF_API_TOKEN")

    if not account_id or not api_token:
        print("ERROR: CF_ACCOUNT_ID and CF_API_TOKEN must be set in backend/.env")
        sys.exit(1)

    # Check adapter files
    safetensors = os.path.join(args.adapter_dir, "adapter_model.safetensors")
    config_json  = os.path.join(args.adapter_dir, "adapter_config.json")
    for p in (safetensors, config_json):
        if not os.path.exists(p):
            print(f"ERROR: Missing file: {p}")
            print("       Run training/finetune_lora.py first to generate the adapter.")
            sys.exit(1)

    # Validate adapter_config.json
    with open(config_json) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    if model_type not in ("mistral", "gemma", "llama"):
        print(f"ERROR: adapter_config.json must have model_type in "
              f"(mistral, gemma, llama).  Found: {model_type!r}")
        print("       Patch the file and re-run.")
        sys.exit(1)
    rank = cfg.get("r", 0)
    if rank > 32:
        print(f"WARNING: LoRA rank={rank} > 32. Cloudflare may reject this adapter.")
    size_mb = os.path.getsize(safetensors) / 1e6
    if size_mb > 300:
        print(f"WARNING: adapter_model.safetensors is {size_mb:.0f} MB > 300 MB limit.")

    print("=" * 60)
    print("  Cloudflare LoRA Upload")
    print(f"  Model:       {args.model}")
    print(f"  Name:        {args.name}")
    print(f"  Adapter dir: {args.adapter_dir}")
    print(f"  Rank:        {rank}")
    print(f"  Size:        {size_mb:.1f} MB")
    print("=" * 60)

    BASE = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai"

    if args.dry_run:
        print("\nDRY RUN — equivalent curl commands:\n")
        print(f"# 1. Create fine-tune")
        print(f'curl "{BASE}/finetunes" \\')
        print(f'  --request POST \\')
        print(f'  --header "Authorization: Bearer $CF_API_TOKEN" \\')
        print(f'  --json \'{{"model":"{args.model}","name":"{args.name}"}}\'\n')
        print(f"# 2. Upload adapter_config.json")
        print(f'curl -X POST "{BASE}/finetunes/<ID>/finetune-assets/" \\')
        print(f'  -H "Authorization: Bearer $CF_API_TOKEN" \\')
        print(f'  -F "file_name=adapter_config.json" \\')
        print(f'  -F "file=@{config_json}"\n')
        print(f"# 3. Upload adapter_model.safetensors")
        print(f'curl -X POST "{BASE}/finetunes/<ID>/finetune-assets/" \\')
        print(f'  -H "Authorization: Bearer $CF_API_TOKEN" \\')
        print(f'  -F "file_name=adapter_model.safetensors" \\')
        print(f'  -F "file=@{safetensors}"\n')
        return

    import httpx  # type: ignore

    headers = {"Authorization": f"Bearer {api_token}"}

    # 1. Create fine-tune
    print("Step 1: Creating fine-tune on Cloudflare...")
    r = httpx.post(
        f"{BASE}/finetunes",
        headers={**headers, "Content-Type": "application/json"},
        json={"model": args.model, "name": args.name},
        timeout=30,
    )
    data = r.json()
    if not data.get("success"):
        print(f"ERROR: {data}")
        sys.exit(1)
    ft_id = data["result"]["id"]
    print(f"  Created fine-tune id: {ft_id}")

    # 2. Upload files
    for fname, fpath in [("adapter_config.json", config_json),
                          ("adapter_model.safetensors", safetensors)]:
        print(f"Step 2: Uploading {fname} ({os.path.getsize(fpath)/1e6:.1f} MB)...")
        with open(fpath, "rb") as fh:
            r2 = httpx.post(
                f"{BASE}/finetunes/{ft_id}/finetune-assets/",
                headers=headers,
                files={"file": (fname, fh), "file_name": (None, fname)},
                timeout=600,
            )
        data2 = r2.json()
        if not data2.get("success"):
            print(f"  ERROR uploading {fname}: {data2}")
            sys.exit(1)
        print(f"  Uploaded {fname} successfully.")

    print()
    print("Upload complete!")
    print(f"  Fine-tune name : {args.name}")
    print(f"  Fine-tune ID   : {ft_id}")
    print()
    print("Now patch backend/.env to use the LoRA adapter:")
    print(f'  CF_MODEL={args.model}')
    print(f'  CF_FINETUNE_ID={ft_id}')
    print()
    print("And update backend/llm_service.py parse_intent() to pass lora= in the payload.")
    print("  payload = {..., \"lora\": finetune_id}")

    # Save the IDs to a file for the integration step
    meta = {"model": args.model, "finetune_name": args.name, "finetune_id": ft_id}
    out = os.path.join(args.adapter_dir, "cloudflare_upload.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {out}")


if __name__ == "__main__":
    main()
