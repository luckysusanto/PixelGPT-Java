#!/usr/bin/env python
# coding=utf-8
"""
DualGPT (ErniePixelForCausalLM) pretraining script.
Optimized for Lazy Loading with Multi-Modality Mixing.
"""

import os
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from PIL import Image

import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Value
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

# ---------- memory/threads guards ----------
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
    dataset_name: Optional[str] = field(default=None)
    text_only_dataset_name: Optional[str] = field(default=None)
    image_only_dataset_name: Optional[str] = field(default=None)
    train_split: str = field(default="train")
    image_column: str = field(default="pixel_values")
    text_column: str = field(default="token_ids")
    max_seq_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/dualgpt-pretrain-output")
    per_device_train_batch_size: int = field(default=8)
    num_train_epochs: float = field(default=1.0) 
    
    # --- UPDATED STANDARD OPTIMIZER SETTINGS ---
    learning_rate: float = field(default=5e-4) # Fixed Max LR
    weight_decay: float = field(default=0.1)   # Standard weight decay for LLMs/ViTs
    lr_scheduler_type: str = field(default='cosine') # Cosine is standard for pretraining
    warmup_steps: int = field(default=1000)
    # -------------------------------------------

    bf16: bool = field(default=True)
    logging_steps: int = field(default=100)
    save_total_limit: int = field(default=20)
    dataloader_num_workers: int = field(default=4)
    
    # Removed base_learning_rate as it is no longer used
    
    # Important: We must force this to False later in main() to be safe, 
    # but setting default here helps too.
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
        # Filter out Nones (which happens if transformation failed or wasn't applicable)
        # Note: lazy_process returns dicts with 'pixel_values' and 'input_ids'
        # One of them might be None depending on the mode.
        
        # Check if we have *any* valid pixel data in this batch
        has_pixel = any(b['pixel_values'] is not None for b in batch)
        # Check if we have *any* valid text data in this batch
        has_text = any(b['input_ids'] is not None for b in batch)

        batch_out = {}

        # --- Handle Images ---
        if has_pixel:
            pixel_list = []
            pixel_mask_list = []
            # Placeholder for rows that are text-only
            placeholder_img = torch.zeros((self.num_channels, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            
            for item in batch:
                if item.get('pixel_values') is not None:
                    pixel_list.append(item['pixel_values'])
                    # Calculate patches for mask
                    h, w = item['pixel_values'].shape[1], item['pixel_values'].shape[2]
                    num_patches = (h // self.patch_size) * (w // self.patch_size)
                    pixel_mask_list.append(torch.ones(num_patches, dtype=torch.long))
                else:
                    # Text-only row: insert black image, mask it out
                    pixel_list.append(placeholder_img)
                    num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
                    pixel_mask_list.append(torch.zeros(num_patches, dtype=torch.long))

            batch_out['pixel_values'] = torch.stack(pixel_list)
            batch_out['pixel_attention_mask'] = torch.stack(pixel_mask_list)

        # --- Handle Text ---
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
                    # Image-only row: insert 1 PAD token, mask it via labels=-100
                    input_ids_list.append(torch.tensor([pad_id], dtype=torch.long))
                    labels_list.append(torch.tensor([-100], dtype=torch.long))

            batch_out['input_ids'] = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            batch_out['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)
            
            # Create attention mask
            text_mask = (batch_out['input_ids'] != pad_id).long()
            # Explicitly zero-out mask for image-only rows
            for i, item in enumerate(batch):
                if item.get('input_ids') is None:
                    text_mask[i] = 0
            batch_out['attention_mask'] = text_mask

        return batch_out

# -------------------------
# Lazy Transform Logic
# -------------------------
def make_lazy_transform(tokenizer, image_col, text_col, max_len, image_size, num_channels):
    # Setup Transforms
    image_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), # Converts [0,255] to [0,1]
    ])
    eos_token_id = tokenizer.eos_token_id

    def lazy_process(batch):
        # batch is a dict of lists, including our 'mode' column
        # modes: 0=Paired, 1=Text-Only, 2=Image-Only
        
        batch_len = len(batch[next(iter(batch))])
        processed_pixels = []
        processed_ids = []
        
        # Safe retrieval of modes. If something goes wrong, default to 0 (Paired)
        modes = batch.get('mode', [0] * batch_len)

        for i in range(batch_len):
            mode = modes[i]
            
            # --- PROCESS TEXT (Modes 0 and 1) ---
            p_ids = None
            if mode in [0, 1]: 
                raw_ids = batch[text_col][i]
                if raw_ids is not None and len(raw_ids) > 0:
                    current_ids = list(raw_ids)
                    # Deduplicate EOS if present
                    if current_ids and current_ids[-1] == eos_token_id:
                        current_ids.pop()
                    # Truncate
                    if len(current_ids) > max_len - 1:
                        current_ids = current_ids[:max_len - 1]
                    # Add EOS
                    current_ids.append(eos_token_id)
                    p_ids = current_ids
            
            # --- PROCESS IMAGE (Modes 0 and 2) ---
            p_img = None
            if mode in [0, 2]:
                raw_img = batch[image_col][i]
                if raw_img is not None:
                    try:
                        # Handle List of Lists of Integers (16 x 16384)
                        # We convert directly to numpy uint8
                        np_img = np.array(raw_img, dtype=np.uint8)
                        
                        # Convert to PIL for the Torchvision transforms
                        pil_image = Image.fromarray(np_img)

                        # Ensure RGB if model expects 3 channels
                        if num_channels == 3 and pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        p_img = image_transform(pil_image)
                    except Exception as e:
                        # logger.warning(f"Image error index {i}: {e}")
                        p_img = None 

            processed_ids.append(p_ids)
            processed_pixels.append(p_img)

        return {"input_ids": processed_ids, "pixel_values": processed_pixels}

    return lazy_process


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- CRITICAL FIX: Force remove_unused_columns to False ---
    # This prevents Trainer from dropping the 'mode' column we add below.
    training_args.remove_unused_columns = False

    os.makedirs(training_args.output_dir, exist_ok=True)
    set_seed(training_args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Vocab Sync
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            tokenizer.save_pretrained(training_args.output_dir)
        torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

    # Model
    if model_args.model_name_or_path:
        config = ErniePixelConfig.from_pretrained(model_args.model_name_or_path)
        model = ErniePixelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    else:
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

    # --- Data Loading ---
    datasets_to_mix = []

    # 1. Paired Dataset (Mode 0)
    if data_args.dataset_name:
        logger.info(f"Loading PAIRED dataset: {data_args.dataset_name}")
        ds = load_dataset(data_args.dataset_name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        ds = ds.add_column("mode", [0] * len(ds))
        # Ensure mode is cast to integer to survive concatenation issues
        # ds = ds.cast_column("mode", Value("int8"))
        datasets_to_mix.append(ds)

    # 2. Text Only (Mode 1)
    if data_args.text_only_dataset_name:
        logger.info(f"Loading TEXT-ONLY dataset: {data_args.text_only_dataset_name}")
        ds = load_dataset(data_args.text_only_dataset_name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        ds = ds.add_column("mode", [1] * len(ds))
        # ds = ds.cast_column("mode", Value("int8"))
        datasets_to_mix.append(ds)

    # 3. Image Only (Mode 2)
    if data_args.image_only_dataset_name:
        logger.info(f"Loading IMAGE-ONLY dataset: {data_args.image_only_dataset_name}")
        ds = load_dataset(data_args.image_only_dataset_name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        ds = ds.add_column("mode", [2] * len(ds))
        # ds = ds.cast_column("mode", Value("int8"))
        datasets_to_mix.append(ds)

    if not datasets_to_mix:
        raise ValueError("No datasets provided.")

    logger.info("Concatenating datasets...")
    combined_dataset = concatenate_datasets(datasets_to_mix)
    combined_dataset = combined_dataset.shuffle(seed=training_args.seed)
    
    logger.info(f"Total dataset size: {len(combined_dataset)}")
    
    # Setup Lazy Transform
    transform_fn = make_lazy_transform(
        tokenizer, 
        data_args.image_column, 
        data_args.text_column,
        data_args.max_seq_length,
        model_args.image_size,
        model_args.num_channels
    )

    combined_dataset.set_transform(transform_fn)

    # Collator
    data_collator = SmartMultimodalCollator(
        tokenizer=tokenizer,
        patch_size=model_args.patch_size,
        image_size=model_args.image_size,
        num_channels=model_args.num_channels
    )

    # --- REMOVED MANUAL LR SCALING HERE ---
    # The Trainer will now use the learning_rate and weight_decay 
    # defined in CustomTrainingArguments automatically.

    # Callbacks
    # Define steps relative to epoch size roughly, or hardcode frequently
    checkpoint_steps = [500, 5000, 10000, 20000, 30000, 40000, 50000, 100000] 
    from transformers import TrainerCallback, TrainerControl, TrainerState
    class CustomCheckpointCallback(TrainerCallback):
        def __init__(self, save_steps): self.save_steps = set(save_steps)
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.save_steps: control.should_save = True
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[CustomCheckpointCallback(checkpoint_steps)],
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()