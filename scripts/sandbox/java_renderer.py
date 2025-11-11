from PIL import Image
import numpy as np
import regex
import pandas as pd
import os
from tqdm import tqdm
import glob
import json
from pixel import AksaraPyGameTextRenderer

renderer_config_path = '../../renderers/m4_renderer'
custom_text_renderer = AksaraPyGameTextRenderer.from_pretrained(renderer_config_path)

from transformers import LlamaTokenizerFast, TrainingArguments, AutoTokenizer

from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets, load_from_disk
from huggingface_hub import login

print("Load Tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained("izzako/javanese-llama-tokenizer")


df = pd.read_parquet('../../../pretrain_data/jawa/pretraining_jawa.parquet')


batch_size = 5000
records = []

num_parts = (len(df)-1)//batch_size
print(f'there is {num_parts} parts')

for i, row in tqdm(df.iterrows(), total=len(df)):
    # if i//batch_size<240: ## continue from before
    #     continue
    pixel_encoding = custom_text_renderer(list(row['tokenized_aksara']))
    text_encoding = tokenizer.encode(row['tokenized_text'], is_split_into_words=True)
    example = {
        'text_id': row['id'],
        'chunk_id': row['chunk_id'],
        'pixel_values': pixel_encoding.pixel_values,
        'token_ids': text_encoding
    }
    records.append(example)

    # periodically save to disk
    if len(records) >= batch_size:
        ds = Dataset.from_list(records)
        ds.to_parquet(f"../../../pretrain_data/jawa/batches/tmp_dataset_part_{i//batch_size}.parquet")
        records = []  # free memory

login(os.environ.get('HF_TOKEN'))

print('load and concatenate..')
paths = sorted(glob.glob("../../../pretrain_data/jawa/batches/tmp_dataset_part_*.parquet"))
datasets_list = [Dataset.from_parquet(p) for p in paths]
full_ds = concatenate_datasets(datasets_list)
full_ds.push_to_hub(
    "izzako/javanese-pixelgpt-poc",
    config_name="default",
    commit_message="Initial dataset upload",
    private=False  # make it private if needed
)