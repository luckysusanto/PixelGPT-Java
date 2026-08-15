import os
from huggingface_hub import login
from datasets import load_dataset

# === Setup ===
CACHE_DIR = "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# === Authenticate (only needed once per environment) ===
# You can get your token from https://huggingface.co/settings/tokens
# Recommended: store in an environment variable instead of hardcoding
HF_TOKEN = "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"
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️  No HF_TOKEN found in environment. Set it via:")
    print("    export HF_TOKEN='your_token_here'")

print(f"--- Starting dataset download to cache directory: {CACHE_DIR} ---")

# === Download dataset ===
dataset_name = "izzako/balinese-pixelgpt"

ds = load_dataset(
    dataset_name,
    cache_dir=CACHE_DIR,
    token=HF_TOKEN,  # Ensures private or gated datasets can be accessed
    trust_remote_code=True
)

print(f"--- ✅ Dataset '{dataset_name}' successfully downloaded to cache. ---")
print(ds)
