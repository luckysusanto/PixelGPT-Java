from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glob
import shutil
from pixel import AksaraPyGameTextRenderer
import argparse
from transformers import LlamaTokenizerFast, AutoTokenizer
from datasets import Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import login

import sys

try:
    HF_HOME = os.environ["HF_HOME"]
    # Extra safety check: Ensure the path is actually in your projectnb space
    if "/projectnb/multilm/lsusanto/" not in HF_HOME:
        print(f"CRITICAL ERROR: HF_HOME is set to '{HF_HOME}', which is NOT in projectnb.")
        print("To protect your home quota, the script will exit now.")
        sys.exit(1)
except KeyError:
    print("CRITICAL ERROR: Environment variable 'HF_HOME' is not set.")
    print("Please set it in your shell script (e.g., export HF_HOME='/projectnb/multilm/lsusanto/hf_cache')")
    sys.exit(1)

print(f"HF Cache successfully redirected to: {HF_HOME}")

def process_parquet_to_dataset(filepath, renderer, t1, t2, t3, t4, batch_path, batch_size=5000, debug=False):
    """Process parquet with 4 different tokenizers and trimmed pixel values"""
    df = pd.read_parquet(filepath)
    
    # If debug mode, only process first 2000 rows
    if debug:
        df = df.head(2000)
        print(f"DEBUG MODE: Processing only {len(df)} samples from {filepath}")
    
    os.makedirs(batch_path, exist_ok=True)
    
    records = []
    num_parts = 1 + (len(df)) // batch_size
    print(f'Processing {filepath}: {num_parts} parts')
    
    patch_size = 16 

    for i, row in enumerate(tqdm(df.itertuples(), total=len(df))):
        # 1. Pixel Rendering & Trimming
        # The renderer places a black patch (EOS) immediately after the text patches.
        encoding = renderer(row.chunk_aksara)
        total_patches = encoding.num_text_patches + 1
        total_patches = min(total_patches, renderer.max_seq_length)
        crop_width = total_patches * patch_size
        
        # Slice and ensure uint8 for efficiency
        trimmed_pixels = encoding.pixel_values[:, :crop_width].astype(np.uint8)

        # 2. Tokenization using 4 different strategies
        # T1: Custom Grapheme-based Llama (expects split words/graphemes)
        tok_grapheme = t1.encode(row.tokenized_text, is_split_into_words=True)
        
        # T2: Standard Llama-2 (DualGPT)
        tok_llama2 = t2.encode(row.chunk_text)
        
        # T3: Komodo-7b (Indonesian-Optimized Llama)
        tok_komodo = t3.encode(row.chunk_text)
        
        # T4: mT5 (Google Multilingual Unigram)
        tok_mt5 = t4.encode(row.chunk_text)

        records.append({
            'text_id': row.doc_id,
            'chunk_id': row.chunk_id,
            'pixel_values': trimmed_pixels.tolist(),
            'num_patches': total_patches,
            'text': row.chunk_text,
            'tok_grapheme': tok_grapheme,
            'tok_llama2': tok_llama2,
            'tok_komodo': tok_komodo,
            'tok_mt5': tok_mt5
        })
        
        # periodically save to disk
        if len(records) >= batch_size:
            ds = Dataset.from_list(records)
            ds.to_parquet(os.path.join(batch_path, f"tmp_dataset_part_{i//batch_size}.parquet"))
            records = []  # free memory
    
    if records:
        ds = Dataset.from_list(records)
        ds.to_parquet(os.path.join(batch_path, f"tmp_dataset_part_final.parquet"))
    
    print('Loading and concatenating...')
    paths = sorted(glob.glob(f"{batch_path}/tmp_dataset_part_*.parquet"))
    datasets_list = [Dataset.from_parquet(p, cache_dir=HF_HOME) for p in paths]
    full_ds = concatenate_datasets(datasets_list)
    
    return full_ds

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process language data for PixelGPT Tokenizer Comparison')
    parser.add_argument('--renderer_config_path', type=str, default='../../renderers/m4_renderer')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path for tok_grapheme')
    parser.add_argument('--lang', type=str, default='javanese',
                        choices=['javanese', 'balinese', 'sundanese', 'lampung'],
                        help='Language to process')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: process only 2000 max samples from each split')
    
    args = parser.parse_args()
    
    # Map language names
    lang_map = {
        'javanese': ('jawa', 'jv'),
        'balinese': ('bali', 'ban'),
        'sundanese': ('sunda', 'su'),
        'lampung': ('lampung', 'ljp')
    }
    lang2, lang_code = lang_map[args.lang]
    
    # --- Load Tokenizers ---
    print("Loading all 4 Tokenizers...")
    # tok_grapheme: Language specific BPE (Grapheme-aware)
    t1 = LlamaTokenizerFast.from_pretrained(args.tokenizer_path, add_eos_token=True)
    
    # tok_llama2: Standard Llama-2 BPE
    t2 = LlamaTokenizerFast.from_pretrained('ernie-research/DualGPT', add_eos_token=True)
    
    # tok_komodo: Southeast Asian Optimized BPE
    t3 = AutoTokenizer.from_pretrained('Yellow-AI-NLP/komodo-7b-base', add_eos_token=True)
    
    # tok_mt5: Massive Multilingual Unigram
    t4 = AutoTokenizer.from_pretrained('google/mt5-small', add_eos_token=True)
    
    # --- Load and Configure Renderer ---
    print("Configuring Renderer (Font Size 6 for no-clipping)...")
    custom_text_renderer = AksaraPyGameTextRenderer.from_pretrained(args.renderer_config_path)
    custom_text_renderer.font_size = 6
    custom_text_renderer.load_font()
    
    BASE_PATH = f'../../../pretrain_data/{lang2}/'
    TRAIN_FILEPATH = os.path.join(BASE_PATH, f"pretraining_{lang2}.parquet")
    TEST_FILEPATH = os.path.join(BASE_PATH, f"test_{lang2}.parquet")
    TRAIN_BATCH_PATH = os.path.join(BASE_PATH, 'batches_train')
    TEST_BATCH_PATH = os.path.join(BASE_PATH, 'batches_test')
    
    # Process train data
    print("Processing training data...")
    train_ds = process_parquet_to_dataset(TRAIN_FILEPATH, custom_text_renderer, t1, t2, t3, t4, TRAIN_BATCH_PATH, debug=args.debug)
    
    # Process test data
    print("Processing test data...")
    test_ds = process_parquet_to_dataset(TEST_FILEPATH, custom_text_renderer, t1, t2, t3, t4, TEST_BATCH_PATH, debug=args.debug)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({'train': train_ds, 'test': test_ds})
    
    # Push to hub
    login(os.environ.get('HF_TOKEN'))
    repo_name = f"Exqrch/Rebuttal-{args.lang}-pixelgpt" + ("-debug" if args.debug else "")
    print(f'Pushing to hub: {repo_name}')
    
    dataset_dict.push_to_hub(
        repo_name,
        commit_message=f"{'[DEBUG] ' if args.debug else ''} tokenizer ablation study setup",
        private=False
    )
    
    # Update dataset card description
    from huggingface_hub import HfApi
    api = HfApi()
    description = f"""
# {args.lang.title()} PixelGPT Tokenizer Ablation Dataset
Optimized with **Font Size 6** and **Dynamic Trimming**.

## Tokenizer Schema
1. `tok_grapheme`: Language-specific Grapheme BPE ({args.tokenizer_path})
2. `tok_llama2`: Standard Llama-2 BPE (ernie-research/DualGPT)
3. `tok_komodo`: SEA-Optimized BPE (yellow-ai-central/komodo-7b-v1)
4. `tok_mt5`: Google Multilingual Unigram (google/mt5-small)
"""
    api.upload_file(
        path_or_fileobj=description.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset"
    )
    
    # Clean up
    for path in [TRAIN_BATCH_PATH, TEST_BATCH_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
    
    print("\n" + "="*60)
    print(f"COMPLETE: {repo_name}")
    print("="*60)