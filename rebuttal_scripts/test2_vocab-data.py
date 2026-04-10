import os
from datasets import load_dataset
import tqdm

# 1. Setup paths to your specific dataset versions
datasets_to_check = {
    "Javanese": {
        "path": "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7",
        "model_limit": 20066
    },
    "Balinese": {
        "path": "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210",
        "model_limit": 20066
    },
    "Pure Balinese": {
        "path": "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-pure_bali-grapheme_experiment_only/default/0.0.0/97c9a64ef822783fe211834f590bc8e2c3a02b08",
        "model_limit": 15368
    },
    "Sundanese": {
        "path": "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55",
        "model_limit": 4178
    },
    "Lampung": {
        "path": "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73",
        "model_limit": 4178
    }
}

column_name = "tok_grapheme"

print(f"{'LANGUAGE':<12} | {'MAX ID':<8} | {'LIMIT':<8} | {'STATUS'}")
print("-" * 50)

for lang, info in datasets_to_check.items():
    try:
        # Load the dataset
        ds = load_dataset(info["path"], split="train")
        
        # Scan for max ID (we check first 20,000 samples for efficiency, 
        # usually errors are consistent across the whole set)
        global_max = 0
        num_to_scan = min(len(ds), 20000)
        
        for i in range(num_to_scan):
            sample_ids = ds[i][column_name]
            if sample_ids:
                m = max(sample_ids)
                if m > global_max:
                    global_max = m
        
        limit = info["model_limit"]
        # If max ID is 20066 and limit is 20066, that's an ERROR 
        # (IDs are 0-indexed, so max allowed is limit-1)
        status = "✅ OK" if global_max < limit else "❌ CRASH RISK"
        
        print(f"{lang:<12} | {global_max:<8} | {limit:<8} | {status}")
        
        if global_max >= limit:
            print(f"   ⚠️  Alert: {lang} has ID {global_max} but model only goes up to {limit-1}")

    except Exception as e:
        print(f"{lang:<12} | Error loading dataset: {e}")

print("-" * 50)