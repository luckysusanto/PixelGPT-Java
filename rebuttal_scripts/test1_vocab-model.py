import os
import sys
from transformers import AutoTokenizer

# --- Add your project root to Python path ---
PROJECT_ROOT = "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

# Path to your saved models
MODELS_ROOT = "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/models"

model_folders = [
    "dualGPT-vocabResize-java-grapheme",
    "dualGPT-vocabResize-bali-grapheme",
    "dualGPT-vocabResize-sunda-grapheme",
    "dualGPT-vocabResize-llama2",
    "dualGPT-vocabResize-komodo",
    "dualGPT-vocabResize-mt5"
]

# Header
header = f"{'MODEL FOLDER':<35} | {'EMBED':<6} | {'PAD ID':<7} | {'EOS ID':<7} | {'PAD==EOS':<7} | {'STATUS'}"
print(header)
print("-" * len(header))

for folder in model_folders:
    path = os.path.join(MODELS_ROOT, folder)
    
    if not os.path.exists(path):
        print(f"{folder:<35} | {'N/A':<6} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | MISSING")
        continue

    try:
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        pad_eq_eos = (pad_id == eos_id) if (pad_id is not None and eos_id is not None) else "N/A"

        # 2. Load Model (CPU) to get Embedding size
        config = ErniePixelConfig.from_pretrained(path)
        model = ErniePixelForCausalLM.from_pretrained(path, config=config, device_map="cpu")
        embed_len = model.get_input_embeddings().weight.shape[0]
        
        # 3. Validation Logic
        issues = []
        
        # Check if IDs are out of bounds
        if pad_id is not None and pad_id >= embed_len:
            issues.append(f"PAD OOB ({pad_id})")
        if eos_id is not None and eos_id >= embed_len:
            issues.append(f"EOS OOB ({eos_id})")
        if pad_id is None:
            issues.append("MISSING PAD")
            
        status = " ✅ OK" if not issues else " ❌ " + ", ".join(issues)

        # Print row
        print(f"{folder:<35} | {embed_len:<6} | {str(pad_id):<7} | {str(eos_id):<7} | {str(pad_eq_eos):<7} | {status}")

    except Exception as e:
        print(f"{folder:<35} | ERROR: {str(e)}")

print("-" * len(header))