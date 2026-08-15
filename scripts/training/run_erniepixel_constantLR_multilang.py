#!/usr/bin/env python
# coding=utf-8
"""
DualGPT Merged Pretraining Script.
Features:
1. Merges two datasets with robust Schema Casting (fixes List vs Sequence issues).
2. Constant Learning Rate with Warmup.
3. Automatic Epoch Splitting (epoch-0, epoch-1...) for safety.
"""

import os
import logging
import re
import sys
import gc
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
)
from transformers.trainer_utils import get_last_checkpoint
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
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
    dataset_a_name: str = field(metadata={"help": "The first dataset"})
    dataset_b_name: str = field(metadata={"help": "The second dataset"})
    train_split: str = field(default="train")
    image_column: str = field(default="pixel_values")
    text_column: str = field(default="token_ids")
    max_seq_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/merged-pretrain")
    per_device_train_batch_size: int = field(default=8)
    
    # TOTAL number of epochs desired. The script splits this into folders.
    num_train_epochs: float = field(default=1.0) 
    
    # --- OPTIMIZER SETTINGS ---
    learning_rate: float = field(default=5e-4)
    weight_decay: float = field(default=0.1)   
    lr_scheduler_type: str = field(default='constant_with_warmup') 
    warmup_steps: int = field(default=1000)
    # --------------------------

    bf16: bool = field(default=True)
    logging_steps: int = field(default=100)
    save_total_limit: int = field(default=20)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False) 


# -------------------------
# Dynamic Smart Collator
# -------------------------
@dataclass
class SmartMultimodalCollator:
    tokenizer: Any
    patch_size: int
    image_size: List[int]
    num_channels: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_pixel = any(b['pixel_values'] is not None for b in batch)
        has_text = any(b['input_ids'] is not None for b in batch)
        batch_out = {}

        if has_pixel:
            pixel_list = []
            pixel_mask_list = []
            placeholder_img = torch.zeros((self.num_channels, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            for item in batch:
                if item.get('pixel_values') is not None:
                    pixel_list.append(item['pixel_values'])
                    h, w = item['pixel_values'].shape[1], item['pixel_values'].shape[2]
                    num_patches = (h // self.patch_size) * (w // self.patch_size)
                    pixel_mask_list.append(torch.ones(num_patches, dtype=torch.long))
                else:
                    pixel_list.append(placeholder_img)
                    num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
                    pixel_mask_list.append(torch.zeros(num_patches, dtype=torch.long))
            batch_out['pixel_values'] = torch.stack(pixel_list)
            batch_out['pixel_attention_mask'] = torch.stack(pixel_mask_list)

        if has_text:
            input_ids_list = []
            labels_list = []
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            for item in batch:
                if item.get('input_ids') is not None:
                    txt = torch.tensor(item['input_ids'], dtype=torch.long)
                    input_ids_list.append(txt)
                    labels_list.append(txt)
                else:
                    input_ids_list.append(torch.tensor([pad_id], dtype=torch.long))
                    labels_list.append(torch.tensor([-100], dtype=torch.long))
            batch_out['input_ids'] = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            batch_out['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)
            text_mask = (batch_out['input_ids'] != pad_id).long()
            for i, item in enumerate(batch):
                if item.get('input_ids') is None:
                    text_mask[i] = 0
            batch_out['attention_mask'] = text_mask

        return batch_out

# -------------------------
# Lazy Transform Logic
# -------------------------
def make_lazy_transform(tokenizer, image_col, text_col, max_len, image_size, num_channels):
    image_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), 
    ])
    eos_token_id = tokenizer.eos_token_id

    def lazy_process(batch):
        batch_len = len(batch[next(iter(batch))])
        processed_pixels = []
        processed_ids = []
        modes = batch.get('mode', [0] * batch_len)

        for i in range(batch_len):
            mode = modes[i]
            
            p_ids = None
            if mode in [0, 1]: 
                raw_ids = batch[text_col][i]
                if raw_ids is not None and len(raw_ids) > 0:
                    current_ids = list(raw_ids)
                    if current_ids and current_ids[-1] == eos_token_id:
                        current_ids.pop()
                    if len(current_ids) > max_len - 1:
                        current_ids = current_ids[:max_len - 1]
                    current_ids.append(eos_token_id)
                    p_ids = current_ids
            
            p_img = None
            if mode in [0, 2]:
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
        return {"input_ids": processed_ids, "pixel_values": processed_pixels}

    return lazy_process

# -------------------------
# Helper: Resolve Epoch Folders
# -------------------------
def get_next_epoch_paths(base_output_dir):
    if not os.path.exists(base_output_dir):
        return None, os.path.join(base_output_dir, "epoch-0")

    subfolders = [f for f in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, f))]
    epoch_folders = []
    
    for f in subfolders:
        match = re.match(r"epoch-(\d+)", f)
        if match:
            epoch_folders.append(int(match.group(1)))
            
    if not epoch_folders:
        return None, os.path.join(base_output_dir, "epoch-0")
    
    epoch_folders.sort()
    max_epoch = epoch_folders[-1]
    
    latest_model_path = os.path.join(base_output_dir, f"epoch-{max_epoch}")
    new_output_dir = os.path.join(base_output_dir, f"epoch-{max_epoch + 1}")
    
    return latest_model_path, new_output_dir


# -------------------------
# MAIN TRAINING LOOP
# -------------------------
def run_single_epoch(model_args, data_args, training_args, dataset, tokenizer, resume_checkpoint=None):
    """
    Runs exactly 1 training epoch.
    """
    logger.info(f"--- Preparing Epoch Run: {training_args.output_dir} ---")
    
    # 1. Load Model
    # Logic: If model_name_or_path exists (previous epoch), load weights.
    # If not (first epoch), init from scratch.
    load_path = model_args.model_name_or_path
    
    if load_path and os.path.exists(load_path):
        logger.info(f"Loading weights from: {load_path}")
        config = ErniePixelConfig.from_pretrained(load_path)
        model = ErniePixelForCausalLM.from_pretrained(load_path, config=config)
    else:
        logger.info("Initializing model from scratch (Random Init).")
        config = ErniePixelConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.intermediate_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            image_size=model_args.image_size,
            patch_size=model_args.patch_size,
            num_channels=model_args.num_channels,
            rms_norm_eps=1e-6,
        )
        model = ErniePixelForCausalLM(config)
    
    model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 2. Prepare Data (Shuffle happened in main, transform happens here)
    transform_fn = make_lazy_transform(
        tokenizer, 
        data_args.image_column, 
        data_args.text_column,
        data_args.max_seq_length,
        model_args.image_size,
        model_args.num_channels
    )
    dataset.set_transform(transform_fn)

    data_collator = SmartMultimodalCollator(
        tokenizer=tokenizer,
        patch_size=model_args.patch_size,
        image_size=model_args.image_size,
        num_channels=model_args.num_channels
    )

    # 3. Callbacks
    checkpoint_steps = [500, 5000, 10000, 20000, 30000, 40000, 50000, 100000] 
    from transformers import TrainerCallback
    class CustomCheckpointCallback(TrainerCallback):
        def __init__(self, save_steps): self.save_steps = set(save_steps)
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.save_steps: control.should_save = True
            return control

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[CustomCheckpointCallback(checkpoint_steps)],
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    logger.info(f"Saving epoch model to {training_args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    
    # Save Tokenizer to this epoch folder too
    tokenizer.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    set_seed(training_args.seed)
    
    # --- Capture Base Logging Config ---
    base_run_name = training_args.run_name if training_args.run_name else "dualgpt_pretrain"
    base_logging_dir = training_args.logging_dir if training_args.logging_dir else os.path.join(training_args.output_dir, "runs")

    # --- Prepare Tokenizer ---
    tokenizer_source = model_args.model_name_or_path if model_args.model_name_or_path else model_args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # -------------------------------------------------------------------------
    # DATASET PREPARATION & MERGING
    # -------------------------------------------------------------------------
    logger.info("Loading and Merging Datasets...")
    
    # STRICT SCHEMA: Forces both datasets to match (List vs Sequence fix)
    common_features = Features({
        "pixel_values": Sequence(Sequence(Value("uint8"))),
        "token_ids": Sequence(Value("int64")),
        "mode": Value("int64")
    })

    def prepare_single_dataset(name):
        logger.info(f"Processing {name}...")
        ds = load_dataset(name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        
        # 1. Add mode (0 = Paired)
        ds = ds.add_column("mode", [0] * len(ds))
        
        # 2. DROP EXTRA COLUMNS (Text, IDs, etc)
        # We only keep what is strictly necessary and defined in common_features
        ds = ds.select_columns(["pixel_values", "token_ids", "mode"])
        
        # 3. CAST TO SCHEMA (Fixes 'uint8' array issues)
        ds = ds.cast(common_features)
        return ds

    # Load both
    ds_a = prepare_single_dataset(data_args.dataset_a_name)
    ds_b = prepare_single_dataset(data_args.dataset_b_name)
    
    logger.info("Concatenating...")
    combined_dataset = concatenate_datasets([ds_a, ds_b])
    
    # SHUFFLE with Set Seed 42
    logger.info("Shuffling with seed 42...")
    combined_dataset = combined_dataset.shuffle(seed=42)
    logger.info(f"Total Combined Size: {len(combined_dataset)}")

    # -------------------------------------------------------------------------
    # MAIN EPOCH LOOP
    # -------------------------------------------------------------------------
    total_requested_epochs = int(training_args.num_train_epochs)
    
    # Force the internal Trainer to only run 1 epoch at a time
    training_args.num_train_epochs = 1.0
    
    base_output_dir = training_args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    logger.info(f"Requested {total_requested_epochs} sequential epochs.")
    
    for i in range(total_requested_epochs):
        logger.info(f"=== Starting Sequence {i+1}/{total_requested_epochs} ===")
        
        # A. Resolve Folders
        latest_model_path, new_output_dir = get_next_epoch_paths(base_output_dir)
        
        # Determine Epoch ID for logging
        current_epoch_id = int(re.match(r".*epoch-(\d+)", new_output_dir).group(1))

        # B. Check for Crash Recovery
        resume_checkpoint = None
        if os.path.isdir(new_output_dir):
            last_ckpt = get_last_checkpoint(new_output_dir)
            if last_ckpt:
                logger.info(f"Found crash checkpoint in target: {last_ckpt}. Resuming...")
                resume_checkpoint = last_ckpt
        
        # C. Update Args
        training_args.output_dir = new_output_dir
        
        # TensorBoard Cleanliness
        training_args.run_name = f"{base_run_name}_epoch_{current_epoch_id}"
        training_args.logging_dir = os.path.join(base_logging_dir, f"epoch_{current_epoch_id}")
        
        # D. Update Model Source
        if latest_model_path and not resume_checkpoint:
            model_args.model_name_or_path = latest_model_path
        elif not latest_model_path and not resume_checkpoint:
            # First run ever: Init from scratch
            model_args.model_name_or_path = None

        # E. Run Epoch
        run_single_epoch(
            model_args, 
            data_args, 
            training_args, 
            combined_dataset, # Pass the shuffled merged dataset
            tokenizer, 
            resume_checkpoint
        )
        
    logger.info("All requested epochs completed.")

if __name__ == "__main__":
    main()