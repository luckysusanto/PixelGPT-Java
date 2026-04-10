import os
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found! Please check your .env file.")

print("Authenticating with Hugging Face...")
login(token=hf_token)

# 1. Define your custom cache directory
CACHE_DIR = "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets"

# Ensure the directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# 2. Force Hugging Face to use this directory for EVERYTHING in this script
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# 3. List the datasets you pushed to the hub in your first script
datasets_to_cache = [
    "Exqrch/Rebuttal-javanese-pixelgpt",
    "Exqrch/Rebuttal-balinese-pixelgpt",
    "Exqrch/Rebuttal-pureBali-graphemeExperimentOnly",
    "Exqrch/Rebuttal-sundanese-pixelgpt",
    "Exqrch/Rebuttal-lampung-pixelgpt"
]

print("="*60)
print(f"Starting dataset downloads to custom cache:\n{CACHE_DIR}")
print("="*60)

for repo_id in datasets_to_cache:
    print(f"\n📥 Downloading and caching: {repo_id}...")
    try:
        # load_dataset automatically handles the downloading and arrow-caching
        ds = load_dataset(repo_id, cache_dir=CACHE_DIR)
        
        print(f"✅ Successfully cached {repo_id}!")
        print(f"   ↳ Train samples: {len(ds['train'])}")
        print(f"   ↳ Test samples:  {len(ds['test'])}")
    except Exception as e:
        print(f"❌ Failed to cache {repo_id}.")
        print(f"   Error: {e}")

print("\n" + "="*60)
print("All downloads complete!")
print("="*60)