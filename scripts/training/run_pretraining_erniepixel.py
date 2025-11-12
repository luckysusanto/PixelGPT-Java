#!/usr/bin/env python
# coding=utf-8
"""
DualGPT (ErniePixelForCausalLM) pretraining script.
Modeled after the official PIXEL pretrain.py, but adapted for a causal,
decoder-only multimodal model.
"""

# ---------- memory/threads/multiprocessing guards ----------
import os
import torch.multiprocessing as mp

# allocator & fragmentation settings (helps avoid "contiguous allocation" OOM)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")

# prevent BLAS/OMP oversubscription in worker processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# reduce cudnn surprises (optional but safer when debugging memory)
import torch
torch.backends.cudnn.benchmark = False


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
from transformers.trainer_utils import get_last_checkpoint
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

# Import your model/config
from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------
# Dataclasses for arguments
# -------------------------
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Optional pretrained model path to initialize from."}
    )
    tokenizer_path: str = field(
        default="izzako/javanese-llama-tokenizer",
        metadata={"help": "Path to the pretrained tokenizer, either local or on Hugging Face Hub."}
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={"help": "Comma-separated overrides for config when instantiating from scratch."}
    )
    image_size: List[int] = field(
        default_factory=lambda: [16, 16384],
        metadata={
            "help": "Image height and width for preprocessing. Pass two integers.",
            "nargs": 2
        }
    )
    patch_size: int = field(default=16, metadata={"help": "Patch size used by the model."})
    num_channels: int = field(default=3, metadata={"help": "Number of image channels."})
    hidden_size: int = field(default=768, metadata={"help": "Model hidden size."})
    intermediate_size: int = field(default=3072, metadata={"help": "Model intermediate size."})
    num_hidden_layers: int = field(default=12, metadata={"help": "Number of hidden layers."})
    num_attention_heads: int = field(default=12, metadata={"help": "Number of attention heads."})


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(metadata={"help": "Hugging Face dataset id for training (e.g. username/dataset_name)"})
    train_split: str = field(default="train", metadata={"help": "Split name for training."})
    image_column: str = field(default="pixel_values", metadata={"help": "Column that holds raw image data (PIL/ndarray)."})
    text_column: str = field(default="token_ids", metadata={"help": "Column that holds text strings or token ids."})
    max_seq_length: int = field(default=1024, metadata={"help": "Maximum text sequence length (tokens)."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HF dataset cache dir."})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="../experiment_output/dualgpt-pretrain-output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=5e-4, metadata={"help": "The initial learning rate from the paper."}
    )
    lr_scheduler_type: str = field(
        default='linear', metadata={"help": "The scheduler type to use. Matches paper."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer. Matches paper."}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer. Matches paper."}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay for AdamW optimizer."}
    )
    warmup_steps: int = field(
        default=1000, metadata={"help": "Linear warmup over warmup_steps."}
    )
    bf16: bool = field(
        default=True, metadata={"help": "Whether to use bf16 (mixed) precision. Matches paper."}
    )
    fp16: bool = field(
        default=False, metadata={"help": "Whether to use fp16 (mixed) precision. Disabled to prefer bf16."}
    )
    logging_steps: int = field(
        default=100, metadata={"help": "Log every X updates steps."}
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "The checkpoint save strategy to use."}
    )
    save_steps: int = field(
        default=1_000_000_000,
        metadata={"help": "Save checkpoint every X updates steps. Disabled in favor of custom callback."},
    )
    save_total_limit: int = field(
        default=20,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not used by the model forward pass. Should be False for this script."},
    )
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["tensorboard"],
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."},
    )
    base_learning_rate: Optional[float] = field(
        default=None, metadata={"help": "Base LR scaling (optional). absolute_lr = base_lr * total_batch / 256"}
    )
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={"help": "Perform layer scan to find unused layers. Set to False as it defaults to True in accelerate."}
    )


# -------------------------
# Custom callback for checks
# -------------------------
from transformers import TrainerCallback, TrainerControl, TrainerState

class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps: List[int]):
        self.save_steps = set(save_steps)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in self.save_steps:
            control.should_save = True
        return control


# -------------------------
# Data Collator
# -------------------------
def collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, patch_size: int) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids_list = [torch.tensor(item['input_ids']) for item in batch]
    padded_input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    text_attention_mask = (padded_input_ids != tokenizer.pad_token_id).long()
    num_patches_h = pixel_values.shape[2] // patch_size
    assert num_patches_h == 1, f"WARNING: Height patch should be 1, got {num_patches_h} instead."
    num_patches_w = pixel_values.shape[3] // patch_size
    num_patches = num_patches_h * num_patches_w
    pixel_attention_mask = torch.ones(pixel_values.shape[0], num_patches, dtype=torch.long)
    labels = pad_sequence(
        input_ids_list, batch_first=True, padding_value=-100
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": padded_input_ids,
        "labels": labels,
        "attention_mask": text_attention_mask,
        "pixel_attention_mask": pixel_attention_mask,
    }

# -------------------------
# Utility: preprocessing
# -------------------------
def make_preprocess_fn(tokenizer, image_column, text_column, max_seq_length, image_size, num_channels):
    """
    Creates a preprocessing function that handles image transformations and, crucially,
    formats PRE-TOKENIZED text input to ensure correct truncation and EOS termination.
    """
    # Define the image transformation pipeline (this part is correct and unchanged)
    image_transform = transforms.Compose([
        transforms.ToPILImage(), # Ensure input is a PIL Image
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), # Converts HWC [0,255] PIL Image to CHW [0,1] Tensor
    ])
    
    # Define the placeholder tensor for corrupted images (this part is correct and unchanged)
    placeholder_tensor = torch.zeros((num_channels, image_size[0], image_size[1]), dtype=torch.float32)

    def preprocess(examples):
        # --- Image Handling (Correct and Unchanged) ---
        imgs = []
        for i, pil_image in enumerate(examples[image_column]):
            try:
                # Ensure image is in RGB format
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Check for corrupted images (e.g., zero size)
                if pil_image.width == 0 or pil_image.height == 0:
                    raise ValueError(f"Corrupted image with zero dimension: ({pil_image.width}, {pil_image.height})")
                
                imgs.append(image_transform(pil_image))
            except Exception as e:
                item_index = examples['index'][i] if 'index' in examples else f"#{i} in batch"
                logger.warning(f"Error processing image with index {item_index}: {e}. Using a placeholder.")
                imgs.append(placeholder_tensor)

        examples["pixel_values"] = imgs

        # --- NEW: Text Handling for PRE-TOKENIZED Data ---
        # This logic correctly handles lists of token IDs.
        
        # We need the tokenizer just for its special token IDs.
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token defined for Causal LM pretraining.")

        processed_input_ids = []
        for token_ids in examples[text_column]:
            # Create a mutable copy to work with.
            current_ids = list(token_ids)

            # 1. Sanitize: Remove any pre-existing EOS token to prevent duplication.
            #    This makes the function robust even if some data already has an EOS.
            if current_ids and current_ids[-1] == eos_token_id:
                current_ids.pop()
            
            # 2. Truncate: Trim the sequence to make space for the new EOS token.
            #    We truncate to max_seq_length - 1.
            if len(current_ids) > max_seq_length - 1:
                current_ids = current_ids[:max_seq_length - 1]
            
            # 3. Terminate: Append the definitive EOS token.
            #    Now the sequence is guaranteed to be <= max_seq_length and end with EOS.
            current_ids.append(eos_token_id)
            
            processed_input_ids.append(current_ids)
        
        examples["input_ids"] = processed_input_ids
        return examples
    return preprocess

# -------------------------
# Main flow
# -------------------------
def main():
    torch.autograd.set_detect_anomaly(True) # FOR DEBUGGING ONLY
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")

    set_seed(training_args.seed) # default to 42

    logger.info(f"Loading tokenizer from: {model_args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)

    # # Ensure PAD token exists (same across ranks)
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # --- synchronize tokenizer across all processes ---
    # I think this is the cause of the vocab size missmatch.
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            tokenizer.save_pretrained(training_args.output_dir)
        torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
    # ---------------------------------------------------


    if model_args.model_name_or_path:
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        config = ErniePixelConfig.from_pretrained(model_args.model_name_or_path)
        model = ErniePixelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        logger.info("Initializing a new model from scratch.")
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
        # model.resize_token_embeddings(len(tokenizer)) -- Redundant.

    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    logger.info(f"Loading dataset '{data_args.dataset_name}'...")
    ds = load_dataset(data_args.dataset_name, split=data_args.train_split, cache_dir=data_args.cache_dir)
    
    ds = ds.add_column("index", range(len(ds)))
    logger.info(f"Dataset loaded, number of examples: {len(ds)}")

    logger.info("Setting dataset transform...")
    preprocess_fn = make_preprocess_fn(
        tokenizer,
        data_args.image_column,
        data_args.text_column,
        data_args.max_seq_length,
        model_args.image_size,
        model_args.num_channels,
    )

    ds.set_transform(preprocess_fn)

    from functools import partial
    data_collator = partial(collate_fn, tokenizer=tokenizer, patch_size=model_args.patch_size)

    if training_args.base_learning_rate:
        total_train_batch_size = (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        )
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    max_steps = training_args.max_steps if training_args.max_steps > 0 else 1_000_000
    checkpoint_steps = [500, 5000] + list(range(10000, int(max_steps) + 1, 10000))
    checkpoint_cb = CustomCheckpointCallback(save_steps=checkpoint_steps)

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[checkpoint_cb],
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training finished.")

if __name__ == "__main__":
    main()