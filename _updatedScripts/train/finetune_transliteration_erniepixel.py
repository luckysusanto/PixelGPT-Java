import os
import logging
import time
import sys
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from PIL import Image

import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Features, Sequence, Value
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback
)
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForImageTransliteration

# =============================================================================
# 1. CRITICAL CLUSTER FIXES
# =============================================================================
os.environ["HF_DATASETS_LOCKING_DISABLED"] = "true" 
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = False

def rank_print(msg):
    try:
        rank = int(os.environ.get("RANK", 0))
    except:
        rank = 0
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[RANK {rank} | {ts}] {msg}", flush=True)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the pretrained ErniePixelForCausalLM checkpoint"})
    tokenizer_path: str = field(default="izzako/javanese-llama-tokenizer")
    image_size: List[int] = field(default_factory=lambda: [16, 16384], metadata={"nargs": 2})
    patch_size: int = field(default=16)
    num_channels: int = field(default=3)
    hidden_size: int = field(default=768)
    intermediate_size: int = field(default=3072)
    num_hidden_layers: int = field(default=12)
    num_attention_heads: int = field(default=12)

@dataclass
class DataTrainingArguments:
    dataset_a_name: str = field(metadata={"help": "The first dataset path"})
    dataset_b_name: Optional[str] = field(default=None, metadata={"help": "The second dataset path (Optional)"})
    train_split: str = field(default="train")
    image_column: str = field(default="pixel_values")
    # Default to token_ids, but shell script will override to 'grapheme_token_ids'
    text_column: str = field(default="token_ids") 
    max_seq_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)
    validation_samples_per_dataset: int = field(default=1000)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/dualgpt-finetune-transliteration")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: float = field(default=5.0) 
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = field(default='cosine')
    warmup_ratio: float = field(default=0.03)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=50)
    
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    eval_steps: int = field(default=500) 
    save_steps: int = field(default=500)
    
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    save_total_limit: int = field(default=2) 
    
    dataloader_num_workers: int = field(default=0)
    remove_unused_columns: bool = field(default=False) 
    weight_decay: float = field(default=0.1)
    
    # CRITICAL DDP SETTINGS
    ddp_find_unused_parameters: bool = field(default=True)
    ddp_broadcast_buffers: bool = field(default=False)
    
    max_training_time_hours: float = field(default=24.0)

# -------------------------
# Callbacks
# -------------------------
class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit_hours=24.0):
        self.start_time = None
        self.time_limit = time_limit_hours * 3600

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None and (time.time() - self.start_time > self.time_limit):
            rank_print(f"Time limit reached.")
            control.should_training_stop = True
            control.should_save = True 
            return control

# -------------------------
# Collator
# -------------------------
@dataclass
class TransliterationCollator:
    tokenizer: Any
    patch_size: int
    image_size: List[int]
    num_channels: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_list = []
        pixel_mask_list = []
        input_ids_list = []
        labels_list = []
        
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)

        for item in batch:
            if item.get('pixel_values') is None: continue 
            
            pixel_list.append(item['pixel_values'])
            pixel_mask_list.append(torch.ones(num_patches, dtype=torch.long))

            if item.get('input_ids') is None: continue
            
            txt = torch.tensor(item['input_ids'], dtype=torch.long)
            input_ids_list.append(txt)
            labels_list.append(txt)

        batch_out = {}
        if len(pixel_list) > 0:
            batch_out['pixel_values'] = torch.stack(pixel_list)
            batch_out['pixel_attention_mask'] = torch.stack(pixel_mask_list)
            batch_out['input_ids'] = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            
            # For transliteration, input_ids (target text) are also the labels
            batch_out['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)
            batch_out['attention_mask'] = (batch_out['input_ids'] != pad_id).long()
        
        return batch_out

# -------------------------
# Transform Logic
# -------------------------
def make_finetune_transform(tokenizer, image_col, text_col, max_len, image_size, num_channels):
    image_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), 
    ])
    eos_token_id = tokenizer.eos_token_id

    def process_data(batch):
        batch_len = len(batch[next(iter(batch))])
        processed_pixels = []
        processed_ids = []
        
        for i in range(batch_len):
            p_ids = None
            # DYNAMIC COLUMN NAME
            raw_ids = batch[text_col][i]
            if raw_ids is not None:
                if isinstance(raw_ids, list) and len(raw_ids) > 0:
                    current_ids = list(raw_ids)
                    if current_ids and current_ids[-1] == eos_token_id:
                        current_ids.pop()
                    if len(current_ids) > max_len - 1:
                        current_ids = current_ids[:max_len - 1]
                    current_ids.append(eos_token_id)
                    p_ids = current_ids
                elif isinstance(raw_ids, str):
                    pass 

            p_img = None
            raw_img = batch[image_col][i]
            if raw_img is not None:
                try:
                    np_img = np.array(raw_img, dtype=np.uint8)
                    pil_image = Image.fromarray(np_img)
                    if num_channels == 3 and pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    p_img = image_transform(pil_image)
                except Exception as e:
                    p_img = None 

            processed_ids.append(p_ids)
            processed_pixels.append(p_img)

        # Standardize keys for collator
        return {"input_ids": processed_ids, "pixel_values": processed_pixels}

    return process_data


def main():
    rank_print("Script Starting...")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.dataloader_num_workers > 0:
        training_args.dataloader_num_workers = 0

    training_args.remove_unused_columns = False 
    set_seed(training_args.seed)

    # 1. Tokenizer
    rank_print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)

    # 2. Model
    rank_print(f"Loading weights from {model_args.model_name_or_path}...")
    config = ErniePixelConfig.from_pretrained(model_args.model_name_or_path)
    config.dropout = 0.1 
    
    model = ErniePixelForImageTransliteration.from_pretrained(
        model_args.model_name_or_path, 
        config=config,
        ignore_mismatched_sizes=True 
    )

    rank_print("Freezing lm_pixel_head...")
    if hasattr(model, 'lm_pixel_head'):
        for param in model.lm_pixel_head.parameters():
            param.requires_grad = False
    
    # Disable Gradient Checkpointing to prevent DDP double-hook errors
    model.gradient_checkpointing_disable()
    model.config.use_cache = False

    # 3. Data Loading
    rank_print("Loading Datasets...")
    
    # --- STRICT FEATURES for Concatenation ---
    # We use a dummy 'mode' column to match structure if needed, 
    # but primarily ensure image and text cols match type.
    common_features = Features({
        data_args.image_column: Sequence(Sequence(Value("uint8"))),
        data_args.text_column: Sequence(Value("int64")),
        "mode": Value("int64") # Added for strict casting
    })

    datasets_to_merge_train = []
    datasets_to_merge_val = []

    def prepare_single_dataset(name):
        rank_print(f"Processing {name}...")
        ds = load_dataset(name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        
        # Add dummy mode column to satisfy strict schema
        ds = ds.add_column("mode", [0] * len(ds))
        
        # SELECT ONLY RELEVANT COLUMNS (Drops 'text', 'llama_token_ids', etc)
        ds = ds.select_columns([data_args.image_column, data_args.text_column, "mode"])
        
        # CAST TO SCHEMA
        ds = ds.cast(common_features)
        
        # SAFE SPLIT (Same as Pretraining)
        total_len = len(ds)
        val_size = data_args.validation_samples_per_dataset
        if total_len < val_size * 2:
            val_size = int(total_len * 0.1) 
            
        all_indices = np.arange(total_len)
        rng = np.random.default_rng(42)
        rng.shuffle(all_indices)
        
        val_indices = all_indices[:val_size]
        train_indices = all_indices[val_size:]
        
        val_ds = ds.select(val_indices)
        train_ds = ds.select(train_indices)
        
        return train_ds, val_ds

    train_a, val_a = prepare_single_dataset(data_args.dataset_a_name)
    datasets_to_merge_train.append(train_a)
    datasets_to_merge_val.append(val_a)

    if data_args.dataset_b_name:
        train_b, val_b = prepare_single_dataset(data_args.dataset_b_name)
        datasets_to_merge_train.append(train_b)
        datasets_to_merge_val.append(val_b)

    rank_print("Concatenating and Shuffling...")
    train_dataset = concatenate_datasets(datasets_to_merge_train).shuffle(seed=42)
    eval_dataset = concatenate_datasets(datasets_to_merge_val).shuffle(seed=42)
    
    rank_print(f"Dataset Ready. Train: {len(train_dataset)} | Val: {len(eval_dataset)}")

    # 4. Transform & Collator
    transform_fn = make_finetune_transform(
        tokenizer, 
        data_args.image_column, 
        data_args.text_column, # <-- Pass the correct column name
        data_args.max_seq_length,
        model_args.image_size,
        model_args.num_channels
    )
    train_dataset.set_transform(transform_fn)
    eval_dataset.set_transform(transform_fn)

    data_collator = TransliterationCollator(
        tokenizer=tokenizer,
        patch_size=model_args.patch_size,
        image_size=model_args.image_size,
        num_channels=model_args.num_channels
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5), 
            TimeLimitCallback(time_limit_hours=training_args.max_training_time_hours)
        ],
    )

    rank_print("Starting Fine-Tuning...")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        rank_print("Saving Model...")
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()