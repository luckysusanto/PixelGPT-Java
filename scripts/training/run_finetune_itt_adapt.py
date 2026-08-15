#!/usr/bin/env python
# coding=utf-8
"""
DualGPT Few-Shot Fine-Tuning Script (Text-to-Image).
Combines 'Dirty Data' Fixes with Task-Specific Fine-Tuning logic.
"""

import os
import logging
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
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
# IMPORANT: Using the Generation model, not CausalLM
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForImageGeneration

# ---------- memory/threads guards ----------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the PRETRAINED model"})
    tokenizer_path: str = field(default="izzako/javanese-llama-tokenizer")
    image_size: List[int] = field(default_factory=lambda: [16, 16384], metadata={"nargs": 2})
    patch_size: int = field(default=16)
    num_channels: int = field(default=3)


@dataclass
class DataArguments:
    original_dataset_name: str = field(metadata={"help": "The original dataset"})
    new_dataset_name: str = field(metadata={"help": "The new dataset"})
    num_few_shot_samples: int = field(default=64, metadata={"help": "Number of samples per dataset"})
    image_column: str = field(default="pixel_values")
    text_column: str = field(default="token_ids")
    max_seq_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/fewshot_finetune_output")
    per_device_train_batch_size: int = field(default=4)
    # Fine-tuning usually likes slightly higher epochs for tiny data than pretraining
    num_train_epochs: float = field(default=20.0) 
    learning_rate: float = field(default=2e-5)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    # DDP Argument default
    ddp_find_unused_parameters: bool = field(default=False)


# -------------------------
# Text-to-Image Collator
# -------------------------
@dataclass
class TextToImageCollator:
    tokenizer: Any
    patch_size: int
    image_size: List[int]
    num_channels: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_list = []
        pixel_mask_list = []
        input_ids_list = []
        
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)

        for item in batch:
            # 1. Target Image
            if item.get('pixel_values') is None:
                continue 
            
            pixel_list.append(item['pixel_values'])
            pixel_mask_list.append(torch.ones(num_patches, dtype=torch.long))

            # 2. Input Text (Prompt)
            if item.get('input_ids') is None:
                 continue
            
            txt = torch.tensor(item['input_ids'], dtype=torch.long)
            input_ids_list.append(txt)

        batch_out = {}
        
        if len(pixel_list) > 0:
            batch_out['pixel_values'] = torch.stack(pixel_list)
            batch_out['pixel_attention_mask'] = torch.stack(pixel_mask_list)
            
            batch_out['input_ids'] = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            # Create text mask
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
        # Batch is a dict of lists
        batch_len = len(batch[next(iter(batch))])
        processed_pixels = []
        processed_ids = []
        
        for i in range(batch_len):
            # --- Text Processing ---
            p_ids = None
            raw_ids = batch[text_col][i]
            if raw_ids is not None and len(raw_ids) > 0:
                current_ids = list(raw_ids)
                # Deduplicate/Add EOS logic
                if current_ids and current_ids[-1] == eos_token_id:
                    current_ids.pop()
                if len(current_ids) > max_len - 1:
                    current_ids = current_ids[:max_len - 1]
                current_ids.append(eos_token_id)
                p_ids = current_ids

            # --- Image Processing ---
            p_img = None
            raw_img = batch[image_col][i]
            if raw_img is not None:
                try:
                    # Robust cast for List[List] -> Numpy
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

    return process_data


# -------------------------
# Experiment Engine
# -------------------------
def run_finetune_experiment(
    run_name: str,
    output_subfolder: str,
    model_args: ModelArguments,
    training_args: CustomTrainingArguments,
    dataset,
    tokenizer
):
    specific_output_dir = os.path.join(training_args.output_dir, output_subfolder)
    logger.info(f"--- STARTING FINE-TUNE: {run_name} ---")

    # 1. Config/Model Loading (With Fallback)
    model_path = model_args.model_name_or_path
    config_path = model_path
    if os.path.isdir(model_path):
        if "config.json" not in os.listdir(model_path):
            parent_dir = os.path.dirname(model_path.rstrip("/"))
            if os.path.exists(os.path.join(parent_dir, "config.json")):
                config_path = parent_dir

    logger.info(f"Loading weights: {model_path}")
    config = ErniePixelConfig.from_pretrained(config_path)
    # Recommended dropout for fine-tuning small data
    config.dropout = 0.1 
    
    model = ErniePixelForImageGeneration.from_pretrained(model_path, config=config)
    
    # 2. FREEZE Token Head (Fixes DDP Error for Text2Image)
    logger.info("Freezing lm_token_head for Text-to-Image fine-tuning...")
    for param in model.lm_token_head.parameters():
        param.requires_grad = False
        
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 3. Transform & Collator
    transform_fn = make_finetune_transform(
        tokenizer, 
        "pixel_values", 
        "token_ids",
        1024,
        model_args.image_size,
        model_args.num_channels
    )
    dataset.set_transform(transform_fn)

    data_collator = TextToImageCollator(
        tokenizer=tokenizer,
        patch_size=model_args.patch_size,
        image_size=model_args.image_size,
        num_channels=model_args.num_channels
    )

    # 4. Trainer
    current_args = training_args
    current_args.output_dir = specific_output_dir
    current_args.run_name = run_name

    trainer = Trainer(
        model=model,
        args=current_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()

    # 5. Cleanup
    del model
    del trainer
    del dataset
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"--- FINISHED: {run_name} ---\n")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    set_seed(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading Datasets...")

    # --- STRICT SCHEMA (Same as before) ---
    common_features = Features({
        "pixel_values": Sequence(Sequence(Value("uint8"))),
        "token_ids": Sequence(Value("int64")),
        "mode": Value("int64") # Included to prevent filtering errors, though unused
    })

    def prepare_dataset(name, split, N):
        ds = load_dataset(name, split=split, cache_dir=data_args.cache_dir)
        actual_N = min(len(ds), N)
        ds = ds.select(range(actual_N))
        ds = ds.add_column("mode", [0] * len(ds))
        # Keep only strict columns
        ds = ds.select_columns(["pixel_values", "token_ids", "mode"])
        # Fix types
        ds = ds.cast(common_features)
        return ds

    # Load 64 from each
    N = data_args.num_few_shot_samples
    
    logger.info("Preparing Original Dataset...")
    ds_orig = prepare_dataset(data_args.original_dataset_name, "train", N)
    
    logger.info("Preparing New Dataset...")
    ds_new = prepare_dataset(data_args.new_dataset_name, "train", N)

    # --- Scenario 1: Bali Only (64 samples) ---
    dataset_scenario_1 = ds_new

    # --- Scenario 2: Mixed (64 Bali + 64 Java = 128 samples) ---
    # We mix equal parts so the model doesn't just "forget" Java or "ignore" Bali.
    logger.info("Concatenating datasets for Mixed Scenario...")
    dataset_scenario_2 = concatenate_datasets([ds_orig, ds_new]).shuffle(seed=training_args.seed)

    # Run Experiments
    run_finetune_experiment(
        run_name="finetune_new_only",
        output_subfolder="finetune_baliOnly",
        model_args=model_args,
        training_args=training_args,
        dataset=dataset_scenario_1,
        tokenizer=tokenizer
    )

    run_finetune_experiment(
        run_name="finetune_mixed",
        output_subfolder="finetune_mixBaliJava",
        model_args=model_args,
        training_args=training_args,
        dataset=dataset_scenario_2,
        tokenizer=tokenizer
    )

if __name__ == "__main__":
    main()