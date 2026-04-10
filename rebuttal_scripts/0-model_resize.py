import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

# Add pixelgpt base to python
PROJECT_ROOT = "/workspace/pixel/PixelGPT-Java/"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

# 1. Load environment variables and authenticate
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found! Please check your .env file.")

print("Authenticating with Hugging Face...")
login(token=hf_token)

# 2. Setup paths
CACHE_DIR = "/workspace/pixel/PixelGPT-Java//hf_cache_updated"
os.environ["HF_HOME"] = CACHE_DIR # Protect your home directory!

base_model_path = "ernie-research/DualGPT"
output_dir = os.path.join(CACHE_DIR, "models")
os.makedirs(output_dir, exist_ok=True)

tokenizers = {
    "java-grapheme": "izzako/javanese-llama-tokenizer",
    "bali-grapheme": "izzako/balinese-llama-tokenizer",
    "sunda-grapheme": "izzako/sunda-llama-tokenizer", # lampung pakai sunda tokenizer
    "llama2": "ernie-research/DualGPT",
    "komodo": "Yellow-AI-NLP/komodo-7b-base",
    "mt5": "google/mt5-small"
}

print(f"\nLoading custom config from {base_model_path}...")
config = ErniePixelConfig.from_pretrained(base_model_path, cache_dir=CACHE_DIR)

print(f"Loading custom base model from {base_model_path}...")
# Load using your custom class instead of AutoModel
base_model = ErniePixelForCausalLM.from_pretrained(
    base_model_path, 
    config=config,
    device_map="cpu",
    cache_dir=CACHE_DIR
)

for name, tok_path in tokenizers.items():
    print(f"\nProcessing {name} ({tok_path})...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, cache_dir=CACHE_DIR)
    
    # --- CRITICAL FIX: Add PAD if missing BEFORE resizing ---
    if tokenizer.pad_token is None:
        print(f"Adding [PAD] token to {name}...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Now the length will correctly be 35,009 for Komodo
    new_vocab_size = len(tokenizer)
    base_model.resize_token_embeddings(new_vocab_size)
    
    save_path = os.path.join(output_dir, f"dualGPT-vocabResize-{name}")
    base_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Final Vocab: {new_vocab_size} | Saved to {save_path}")

print("\nAll models prepared successfully!")