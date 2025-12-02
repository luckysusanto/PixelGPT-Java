#!/usr/bin/env python
# coding=utf-8
"""
DualGPT (ErniePixelForCausalLM) continued pretraining script.
This script loads a pretrained model, resizes token embeddings for a new tokenizer
(cold start), and continues pretraining on a new dataset.
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
        default="ernie-research/DualGPT", 
        metadata={"help": "The base pretrained model to start from (e.g., 'ernie-research/DualGPT')."}
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
            "help": "Image height and width for preprocessing. Only used if not in model config.",
            "nargs": 2
        }
    )
    patch_size: int = field(default=16, metadata={"help": "Patch size. Only used if not in model config."})
    num_channels: int = field(default=3, metadata={"help": "Image channels. Only used if not in model config."})
    hidden_size: int = field(default=768, metadata={"help": "Hidden size. Only used if not in model config."})
    intermediate_size: int = field(default=3072, metadata={"help": "Intermediate size. Only used if not in model config."})
    num_hidden_layers: int = field(default=12, metadata={"help": "Num layers. Only used if not in model config."})
    num_attention_heads: int = field(default=12, metadata={"help": "Num heads. Only used if not in model config."})


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(metadata={"help": "Hugging Face dataset id for training (e.g. username/dataset_name)"})
    train_split: str = field(default="train", metadata={"help": "Split name for training."})
    image_column: str = field(default="pixel_values", metadata={"help": "Column that holds raw image data (PIL/ndarray)."})
    text_column: str = field(default="token_ids", metadata={"help": "Column that holds pre-tokenized text ids."})
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
        default=5e-4, metadata={"help": "The initial learning rate."}
    )
    lr_scheduler_type: str = field(
        default='linear', metadata={"help": "The scheduler type to use."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer."}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer."}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay for AdamW optimizer."}
    )
    warmup_steps: int = field(
        default=1000, metadata={"help": "Linear warmup over warmup_steps."}
    )
    bf16: bool = field(
        default=True, metadata={"help": "Whether to use bf16 (mixed) precision."}
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
        metadata={"help": "Limit the total amount of checkpoints."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not used by the model forward pass."},
    )
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["tensorboard"],
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of subprocesses to use for data loading."},
    )
    base_learning_rate: Optional[float] = field(
        default=None, metadata={"help": "Base LR scaling (optional)."}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Disable find_unused_parameters for a potential speedup."}
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
    assert num_patches_h == 1, f"Image height must be exactly one patch high. Got {num_patches_h} patches."
    num_patches_w = pixel_values.shape[3] // patch_size
    num_patches = num_patches_h * num_patches_w
    pixel_attention_mask = torch.ones(pixel_values.shape[0], num_patches, dtype=torch.long)
    
    labels = pad_sequence(
        input_ids_list, batch_first=True, padding_value=-100 # -100 is the standard ignore_index for CrossEntropyLoss
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
    image_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    
    placeholder_tensor = torch.zeros((num_channels, image_size[0], image_size[1]), dtype=torch.float32)

    def preprocess(examples):
        # Image Handling
        imgs = []
        for i, pil_image in enumerate(examples[image_column]):
            try:
                if isinstance(pil_image, (list, np.ndarray)):
                    pil_image = Image.fromarray(np.array(pil_image, dtype=np.uint8))

                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                if pil_image.width == 0 or pil_image.height == 0:
                    raise ValueError("Corrupted image with zero dimension.")
                
                imgs.append(image_transform(pil_image))
            except Exception as e:
                item_index = examples.get('index', [f"#{i} in batch"])[i]
                logger.warning(f"Error processing image with index {item_index}: {e}. Using a placeholder.")
                imgs.append(placeholder_tensor)

        examples["pixel_values"] = imgs

        # Text Handling for Pre-Tokenized Data
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token defined for Causal LM pretraining.")

        processed_input_ids = []
        for token_ids in examples[text_column]:
            current_ids = list(token_ids)

            if current_ids and current_ids[-1] == eos_token_id:
                current_ids.pop()
            
            if len(current_ids) > max_seq_length - 1:
                current_ids = current_ids[:max_seq_length - 1]
            
            current_ids.append(eos_token_id)
            processed_input_ids.append(current_ids)
        
        examples["input_ids"] = processed_input_ids
        return examples
    return preprocess

# -------------------------
# Main flow
# -------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")

    set_seed(training_args.seed)

    logger.info(f"Loading tokenizer from: {model_args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            tokenizer.save_pretrained(training_args.output_dir)
        torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

    logger.info(f"Loading base model weights from '{model_args.model_name_or_path}'")
    model = ErniePixelForCausalLM.from_pretrained(model_args.model_name_or_path)

    original_vocab_size = model.config.vocab_size
    new_vocab_size = len(tokenizer)

    if original_vocab_size != new_vocab_size:
        logger.warning(
            f"Vocabulary size mismatch detected. Base model vocab: {original_vocab_size}, New tokenizer vocab: {new_vocab_size}."
        )
        logger.info("Performing a cold start: resizing the model's token embeddings...")
        
        # --- THE CRITICAL FIX ---
        # Re-set the seed right before the random initialization. This guarantees
        # that every process (on every GPU) generates the exact same random weights.
        set_seed(training_args.seed)
        
        model.resize_token_embeddings(new_vocab_size)
        
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
            
        logger.info(f"Model token embeddings resized to {new_vocab_size}.")
    else:
        logger.info("Vocabulary sizes match. No resizing is necessary.")

    model.gradient_checkpointing_disable()
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
        model.config.image_size,
        model.config.num_channels,
    )
    ds.set_transform(preprocess_fn)

    from functools import partial
    data_collator = partial(collate_fn, tokenizer=tokenizer, patch_size=model.config.patch_size)

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

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
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