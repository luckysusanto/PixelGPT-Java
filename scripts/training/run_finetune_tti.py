#!/usr/bin/env python
# coding=utf-8
"""
ErniePixel Fine-Tuning Script for Text-to-Image Generation (Rendering).
Task: Input Text -> Output Image.
FIXED: Resolves DDP + Gradient Checkpointing conflict by freezing the unused Token Head.
"""

import os
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from PIL import Image

import torch
import numpy as np
from datasets import load_dataset
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
# CHANGED: Import the Generation class
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForImageGeneration

# ---------- memory/threads guards ----------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the pretrained ErniePixel checkpoint"})
    tokenizer_path: str = field(default="izzako/javanese-llama-tokenizer")
    # Note: Keep [16, 16384] if generating text strips. Change to [256, 256] if generating square images.
    image_size: List[int] = field(default_factory=lambda: [16, 16384], metadata={"nargs": 2})
    patch_size: int = field(default=16)
    num_channels: int = field(default=3)
    hidden_size: int = field(default=768)
    intermediate_size: int = field(default=3072)
    num_hidden_layers: int = field(default=12)
    num_attention_heads: int = field(default=12)


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(metadata={"help": "Path to the fine-tuning dataset"})
    train_split: str = field(default="train")
    image_column: str = field(default="pixel_values") 
    text_column: str = field(default="token_ids")   
    max_seq_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/dualgpt-finetune-text2image")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: float = field(default=5.0) 
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = field(default='cosine')
    warmup_ratio: float = field(default=0.03)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=50)
    save_strategy: str = field(default="epoch")
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    weight_decay: float = field(default=0.1) 
    
    # Defaults ddp_find_unused_parameters=False (Compatible with Grad Checkpointing)


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
        
        # Safe access to pad token
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        # Calculate number of patches for the mask
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
            
            # Note: We do NOT create 'labels' from text here.
            # The model class ErniePixelForImageGeneration handles pixel loss 
            # automatically when pixel_values are provided.

        batch_out = {}
        
        if len(pixel_list) > 0:
            batch_out['pixel_values'] = torch.stack(pixel_list)
            batch_out['pixel_attention_mask'] = torch.stack(pixel_mask_list)
            
            # Pad the input text prompts
            batch_out['input_ids'] = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
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
        batch_len = len(batch[image_col])
        processed_pixels = []
        processed_ids = []
        
        for i in range(batch_len):
            # Text Processing (Prompt)
            p_ids = None
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
                    # Fallback for raw strings if dataset isn't tokenized
                    pass # Assumes pre-tokenized based on previous conversation

            # Image Processing (Target)
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

            if p_ids is not None and p_img is not None:
                processed_ids.append(p_ids)
                processed_pixels.append(p_img)
            else:
                processed_ids.append(None)
                processed_pixels.append(None)

        return {"input_ids": processed_ids, "pixel_values": processed_pixels}

    return process_data


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False 
    set_seed(training_args.seed)

    # --- TOKENIZER SETUP (Safe Mode) ---
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    # Use EOS as PAD if missing. Do NOT resize embeddings to keep data valid.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure ID is synced just in case
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading pretrained weights from {model_args.model_name_or_path} into ErniePixelForImageGeneration...")
    
    config = ErniePixelConfig.from_pretrained(model_args.model_name_or_path)
    config.dropout = 0.1 
    
    # CHANGED: Load the ImageGeneration class
    model = ErniePixelForImageGeneration.from_pretrained(
        model_args.model_name_or_path, 
        config=config,
        ignore_mismatched_sizes=False 
    )

    # --- CRITICAL DDP FIX FOR TEXT-TO-IMAGE ---
    # Task: Text -> Image
    # We train: lm_pixel_head
    # We ignore: lm_token_head
    # Therefore, we FREEZE lm_token_head to prevent DDP "unused parameter" errors.
    logger.info("Freezing lm_token_head (dead computation) to prevent DDP unused parameter errors...")
    for param in model.lm_token_head.parameters():
        param.requires_grad = False
    
    # Enable Gradient Checkpointing
    model.gradient_checkpointing_enable() 
    model.config.use_cache = False

    dataset = load_dataset(data_args.dataset_name, cache_dir=data_args.cache_dir)
    
    if hasattr(dataset, 'keys'):
        if data_args.train_split in dataset:
            train_dataset = dataset[data_args.train_split]
        else:
            train_dataset = dataset[list(dataset.keys())[0]]
            logger.warning(f"Split {data_args.train_split} not found, using {list(dataset.keys())[0]}")
    else:
        train_dataset = dataset

    transform_fn = make_finetune_transform(
        tokenizer, 
        data_args.image_column, 
        data_args.text_column,
        data_args.max_seq_length,
        model_args.image_size,
        model_args.num_channels
    )

    train_dataset.set_transform(transform_fn)

    # Use the TextToImageCollator
    data_collator = TextToImageCollator(
        tokenizer=tokenizer,
        patch_size=model_args.patch_size,
        image_size=model_args.image_size,
        num_channels=model_args.num_channels
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting Fine-Tuning (Text-to-Image)...")
    train_result = trainer.train()
    
    logger.info("Saving Model...")
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()