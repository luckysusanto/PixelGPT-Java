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
from pixel import AksaraPyGameTextRenderer

from transformers import LlamaTokenizerFast, TrainingArguments, AutoTokenizer

from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets, load_from_disk
from huggingface_hub import login

from multiprocessing import Pool, cpu_count


print("Load Renderer...")
renderer_config_path = '../../renderers/m4_renderer'
custom_text_renderer = AksaraPyGameTextRenderer.from_pretrained(renderer_config_path)

print("Load Tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained("izzako/aksara-llama-tokenizer",force_download=True)

lang='balinese'
BASE_PATH = '../../../pretrain_data/bali/'
FILEPATH = os.path.join(BASE_PATH,"pretraining_bali.parquet")
BATCH_PATH = os.path.join(BASE_PATH,'batches')

df = pd.read_parquet(FILEPATH)
os.makedirs(BATCH_PATH,exist_ok=True)

batch_size = 5000
records = []

num_parts = (len(df)-1)//batch_size
print(f'there is {num_parts} parts')

for i, row in enumerate(tqdm(df.itertuples(), total=len(df))):
    # if i//batch_size<78: ## continue from before
    #     continue
    pixel_encoding = custom_text_renderer(row.chunk_aksara)
    text_encoding = tokenizer.encode(row.tokenized_text, is_split_into_words=True)
    records.append({
        'text_id': row.doc_id,
        'chunk_id': row.chunk_id,
        'pixel_values': pixel_encoding.pixel_values,
        'token_ids': text_encoding
    })

    # periodically save to disk
    if len(records) >= batch_size:
        ds = Dataset.from_list(records)
        ds.to_parquet(os.path.join(BATCH_PATH,f"tmp_dataset_part_{i//batch_size}.parquet"))
        records = []  # free memory

login(os.environ.get('HF_TOKEN'))

print('load and concatenate..')
paths = sorted(glob.glob(f"{BATCH_PATH}/tmp_dataset_part_*.parquet"))
datasets_list = [Dataset.from_parquet(p) for p in paths]
full_ds = concatenate_datasets(datasets_list)
full_ds.push_to_hub(
    f"izzako/{lang}-pixelgpt-poc",
    config_name="default",
    commit_message="chunked version, verified no more overflow",
    private=False  # make it private if needed
)