from dotenv import load_dotenv
load_dotenv()
from PIL import Image
import numpy as np
import regex
import pandas as pd
import os
from tqdm import tqdm
import glob
import json
import shutil
from pixel import AksaraPyGameTextRenderer
import argparse
from transformers import LlamaTokenizerFast, TrainingArguments, AutoTokenizer
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets, load_from_disk
from huggingface_hub import login

def process_parquet_to_dataset(filepath, custom_text_renderer, tokenizer, original_tokenizer, batch_path, batch_size=5000, debug=False):
    """Process a parquet file and return concatenated dataset"""
    df = pd.read_parquet(filepath)
    
    # If debug mode, only process first 2000 rows
    if debug:
        df = df.head(2000)
        print(f"DEBUG MODE: Processing only {len(df)} samples from {filepath}")
    
    os.makedirs(batch_path, exist_ok=True)
    
    records = []
    num_parts = 1 + (len(df)) // batch_size
    print(f'Processing {filepath}: {num_parts} parts')
    
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df))):
        pixel_encoding = custom_text_renderer(row.chunk_aksara) # "han"
        text_encoding = tokenizer.encode(row.tokenized_text, is_split_into_words=True)
        llama_text_encoding = original_tokenizer.encode(row.chunk_text)
        records.append({
            'text_id': row.doc_id,
            'chunk_id': row.chunk_id,
            'pixel_values': pixel_encoding.pixel_values,
            'grapheme_token_ids': text_encoding,
            'llama_token_ids': llama_text_encoding,
            'text': row.chunk_text
        })
        
        # periodically save to disk
        if len(records) >= batch_size:
            ds = Dataset.from_list(records)
            ds.to_parquet(os.path.join(batch_path, f"tmp_dataset_part_{i//batch_size}.parquet"))
            records = []  # free memory
    
    if records:
        ds = Dataset.from_list(records)
        ds.to_parquet(os.path.join(batch_path, f"tmp_dataset_part_{i//batch_size}.parquet"))
    
    print('Loading and concatenating...')
    paths = sorted(glob.glob(f"{batch_path}/tmp_dataset_part_*.parquet"))
    datasets_list = [Dataset.from_parquet(p) for p in paths]
    full_ds = concatenate_datasets(datasets_list)
    
    return full_ds

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process language data for PixelGPT')
    parser.add_argument('--renderer_config_path', type=str, default='../../renderers/m4_renderer',
                        help='Path to renderer configuration')
    parser.add_argument('--tokenizer_path', type=str, default='izzako/sunda-llama-tokenizer',
                        help='Path or name of the tokenizer')
    parser.add_argument('--lang', type=str, default='sundanese',
                        choices=['javanese', 'balinese', 'sundanese', 'lampung'],
                        help='Language to process')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: process only 2000 max samples from each split')
    
    args = parser.parse_args()
    
    # Map language names
    if args.lang == 'javanese':
        lang2 = 'jawa'
        lang_code = 'jv'
    elif args.lang == 'balinese':
        lang2 = 'bali'
        lang_code = 'ban'
    elif args.lang == 'sundanese':
        lang2 = 'sunda'
        lang_code = 'su'
    elif args.lang == 'lampung':
        lang2 = 'lampung'
        lang_code = 'ljp'
    
    if args.debug:
        print("\n" + "="*60)
        print("DEBUG MODE ENABLED - Processing only 2000 samples (max) per split")
        print("="*60 + "\n")
    
    print("Load Renderer...")
    custom_text_renderer = AksaraPyGameTextRenderer.from_pretrained(args.renderer_config_path)
    
    print("Load Tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer_path, force_download=True, add_eos_token=True)
    original_tokenizer = LlamaTokenizerFast.from_pretrained('ernie-research/DualGPT', add_eos_token=True)
    
    BASE_PATH = f'../../../pretrain_data/{lang2}/'
    TRAIN_FILEPATH = os.path.join(BASE_PATH, f"pretraining_{lang2}.parquet")
    TEST_FILEPATH = os.path.join(BASE_PATH, f"test_{lang2}.parquet")
    TRAIN_BATCH_PATH = os.path.join(BASE_PATH, 'batches_train')
    TEST_BATCH_PATH = os.path.join(BASE_PATH, 'batches_test')
    
    batch_size = 5000
    
    # Process train data
    print("Processing training data...")
    train_ds = process_parquet_to_dataset(
        TRAIN_FILEPATH, 
        custom_text_renderer, 
        tokenizer,
        original_tokenizer,
        TRAIN_BATCH_PATH, 
        batch_size,
        debug=args.debug
    )
    
    # Process test data
    print("Processing test data...")
    test_ds = process_parquet_to_dataset(
        TEST_FILEPATH, 
        custom_text_renderer, 
        tokenizer,
        original_tokenizer,
        TEST_BATCH_PATH, 
        batch_size,
        debug=args.debug
    )
    
    # Create DatasetDict with both splits
    dataset_dict = DatasetDict({
        'train': train_ds,
        'test': test_ds
    })
    
    # Calculate statistics
    train_samples = len(train_ds)
    test_samples = len(test_ds)
    total_samples = train_samples + test_samples
    
    # Create dataset description with proper YAML frontmatter
    debug_note = "\n\n⚠️ **DEBUG VERSION**: This dataset contains only 2000 max samples per split for testing purposes.\n" if args.debug else ""
    
    dataset_description = f"""---
language:
- {lang_code}
license: cc-by-4.0
task_categories:
- text-generation
- image-to-text
tags:
- pixelgpt
- {args.lang}
- aksara
- multimodal
pretty_name: {args.lang.title()} PixelGPT Dataset
size_categories:
- {'n<1K' if args.debug else ('1K<n<10K' if total_samples < 10000 else ('10K<n<100K' if total_samples < 100000 else '100K<n<1M'))}
---

# {args.lang.title()} PixelGPT Dataset
{debug_note}
This dataset contains preprocessed {args.lang.title()} text data for training PixelGPT models.

## Dataset Statistics
- **Language**: {args.lang.title()} ({lang2})
- **Total samples**: {total_samples:,}
- **Train samples**: {train_samples:,}
- **Test samples**: {test_samples:,}

## Tokenizers
- **Grapheme tokenizer**: {args.tokenizer_path}
- **LLaMA tokenizer**: ernie-research/DualGPT

## Features
- `text_id`: Document identifier
- `chunk_id`: Chunk identifier within document
- `pixel_values`: Rendered pixel representation of aksara text
- `grapheme_token_ids`: Token IDs from grapheme-based tokenizer (with EOS token)
- `llama_token_ids`: Token IDs from LLaMA tokenizer (with EOS token)
- `text`: Original text chunk

## Renderer Configuration
- **Renderer path**: {args.renderer_config_path}

## Usage
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("izzako/{args.lang}-pixelgpt{'-debug' if args.debug else ''}")

# Access train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Example: Get first sample
sample = train_data[0]
print(sample['text'])
```

## Citation

If you use this dataset, please cite:
```bibtex
@dataset{{{args.lang}_pixelgpt,
  title = {{{args.lang.title()} PixelGPT Dataset}},
  author = {{Musa Izzanardi Wijanarko}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/izzako/{args.lang}-pixelgpt{'-debug' if args.debug else ''}}}
}}
```
"""
    
    # Push to hub
    login(os.environ.get('HF_TOKEN'))
    print('Pushing to hub...')
    
    repo_name = f"izzako/{args.lang}-pixelgpt" + ("-debug" if args.debug else "")
    commit_msg = f"{'[DEBUG] ' if args.debug else ''}Added both train and test splits - {total_samples:,} total samples"
    
    dataset_dict.push_to_hub(
        repo_name,
        config_name="default",
        commit_message=commit_msg,
        private=False
    )
    
    # Update dataset card with description
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=dataset_description.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
        commit_message="Add dataset description with YAML metadata"
    )
    
    # Clean up batch directories
    print("\nCleaning up temporary batch files...")
    if os.path.exists(TRAIN_BATCH_PATH):
        shutil.rmtree(TRAIN_BATCH_PATH)
        print(f"Removed: {TRAIN_BATCH_PATH}")
    if os.path.exists(TEST_BATCH_PATH):
        shutil.rmtree(TEST_BATCH_PATH)
        print(f"Removed: {TEST_BATCH_PATH}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING COMPLETE" + (" [DEBUG MODE]" if args.debug else ""))
    print("="*60)
    print(f"Language:              {args.lang.title()} ({lang2})")
    print(f"Grapheme Tokenizer:    {args.tokenizer_path}")
    print(f"LLaMA Tokenizer:       ernie-research/DualGPT")
    print(f"Renderer Config:       {args.renderer_config_path}")
    print(f"Mode:                  {'DEBUG (2000 max samples/split)' if args.debug else 'FULL'}")
    print(f"-"*60)
    print(f"Train samples:         {train_samples:,}")
    print(f"Test samples:          {test_samples:,}")
    print(f"Total samples:         {total_samples:,}")
    print(f"-"*60)
    print(f"Dataset uploaded to:   {repo_name}")
    print("="*60)
    
    print("\nDone!")