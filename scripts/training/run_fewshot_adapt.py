#!/usr/bin/env python
# coding=utf-8
"""
DualGPT Few-Shot Adaptation Script.
Robust against Schema Mismatches (List vs Sequence) and Extra Columns.
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
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

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
    num_few_shot_samples: int = field(default=64, metadata={"help": "Number of samples to use"})
    image_column: str = field(default="pixel_values")
    text_column: str = field(default="token_ids")
    max_seq_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/fewshot_output")
    per_device_train_batch_size: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: float = field(default=10.0)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)


# -------------------------
# Collator
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
# Lazy Transform
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
# Training Routine
# -------------------------
def run_training_experiment(
    run_name: str,
    output_subfolder: str,
    model_args: ModelArguments,
    training_args: CustomTrainingArguments,
    dataset,
    tokenizer
):
    specific_output_dir = os.path.join(training_args.output_dir, output_subfolder)
    logger.info(f"--- STARTING: {run_name} ---")

    config = ErniePixelConfig.from_pretrained(model_args.model_name_or_path)
    model = ErniePixelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    transform_fn = make_lazy_transform(
        tokenizer,
        "pixel_values",
        "token_ids",
        1024,
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

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    logger.info("Loading Datasets...")

    # --- DEFINE STRICT SCHEMA ---
    # This forces both datasets to be identical in memory
    common_features = Features({
        "pixel_values": Sequence(Sequence(Value("uint8"))),
        "token_ids": Sequence(Value("int64")),
        "mode": Value("int64")
    })

    def prepare_dataset(name, split, N):
        ds = load_dataset(name, split=split, cache_dir=data_args.cache_dir)
        
        # 1. Take subset first (efficiency)
        actual_N = min(len(ds), N)
        ds = ds.select(range(actual_N))

        # 2. Add 'mode'
        ds = ds.add_column("mode", [0] * len(ds))

        # 3. DROP EXTRA COLUMNS
        # Only keep the 3 we need. This drops 'text', 'chunk_id', etc.
        ds = ds.select_columns(["pixel_values", "token_ids", "mode"])

        # 4. CAST TO STRICT SCHEMA
        # This converts List[List] -> Sequence[Sequence]
        ds = ds.cast(common_features)
        
        return ds

    # Load & Prep
    N = data_args.num_few_shot_samples
    
    logger.info("Preparing Original Dataset...")
    ds_orig = prepare_dataset(data_args.original_dataset_name, "train", N)
    
    logger.info("Preparing New Dataset...")
    ds_new = prepare_dataset(data_args.new_dataset_name, "train", N)

    # Strategy 1: New Only
    dataset_scenario_1 = ds_new

    # Strategy 2: Mixed
    logger.info("Concatenating datasets...")
    dataset_scenario_2 = concatenate_datasets([ds_orig, ds_new]).shuffle(seed=training_args.seed)

    # Execution
    run_training_experiment(
        run_name="fewshot_new_only",
        output_subfolder="furtherPretrain_baliOnly",
        model_args=model_args,
        training_args=training_args,
        dataset=dataset_scenario_1,
        tokenizer=tokenizer
    )

    run_training_experiment(
        run_name="fewshot_mixed",
        output_subfolder="furtherPretrain_mixBaliJava",
        model_args=model_args,
        training_args=training_args,
        dataset=dataset_scenario_2,
        tokenizer=tokenizer
    )


if __name__ == "__main__":
    main()