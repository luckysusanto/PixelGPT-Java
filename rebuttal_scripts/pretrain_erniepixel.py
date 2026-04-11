import os
import logging
import time
import sys
import gc
import datetime
import math
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
from torch.utils.data import Sampler
from torchvision import transforms

from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

# =============================================================================
# 0. HYPERPARAMETER CONFIGURATION SECTION
# =============================================================================
HYPERPARAMETERS = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": [16, 16384],  # [height, max_width]
    "patch_size": 16,
    "num_channels": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 10.0,
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 1000,
    "save_strategy": "steps",
    "evaluation_strategy": "steps",
    "save_steps": 1000,
    "eval_steps": 1000,
    "load_best_model_at_end": True,
    "save_total_limit": 2,
    "early_stopping_patience": 5,
    "logging_steps": 100,
    "report_to": ["tensorboard"],
    "disable_tqdm": False,
    "max_seq_length": 1024,
    "validation_samples_per_dataset": 1000,
    "dataset_a_weight": 1.0,
    "dataset_b_weight": 1.0,
    "max_training_time_hours": 24.0,
    "bf16": True,
    "dataloader_num_workers": 0,
    "remove_unused_columns": False,
    "ddp_find_unused_parameters": True,
    "ddp_broadcast_buffers": False,
}

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
    print(f"[NO-BARRIER | RANK {rank} | {ts}] {msg}", flush=True)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_path: str = field(default="izzako/javanese-llama-tokenizer")
    image_size: List[int] = field(default_factory=lambda: HYPERPARAMETERS["image_size"])
    patch_size: int = field(default=HYPERPARAMETERS["patch_size"])
    num_channels: int = field(default=HYPERPARAMETERS["num_channels"])
    hidden_size: int = field(default=HYPERPARAMETERS["hidden_size"])
    intermediate_size: int = field(default=HYPERPARAMETERS["intermediate_size"])
    num_hidden_layers: int = field(default=HYPERPARAMETERS["num_hidden_layers"])
    num_attention_heads: int = field(default=HYPERPARAMETERS["num_attention_heads"])

@dataclass
class DataTrainingArguments:
    dataset_a_name: str = field(metadata={"help": "The first dataset path"})
    dataset_b_name: Optional[str] = field(default=None, metadata={"help": "The second dataset path (Optional)"})
    train_split: str = field(default="train")
    image_column: str = field(default="pixel_values")
    text_column: str = field(default="token_ids")
    max_seq_length: int = field(default=HYPERPARAMETERS["max_seq_length"])
    cache_dir: Optional[str] = field(default=None)
    validation_samples_per_dataset: int = field(default=HYPERPARAMETERS["validation_samples_per_dataset"])
    dataset_a_weight: float = field(default=HYPERPARAMETERS["dataset_a_weight"])
    dataset_b_weight: float = field(default=HYPERPARAMETERS["dataset_b_weight"])

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/merged-pretrain")
    per_device_train_batch_size: int = field(default=HYPERPARAMETERS["per_device_train_batch_size"])
    per_device_eval_batch_size: int = field(default=HYPERPARAMETERS["per_device_eval_batch_size"])
    gradient_accumulation_steps: int = field(default=HYPERPARAMETERS["gradient_accumulation_steps"])
    num_train_epochs: float = field(default=HYPERPARAMETERS["num_train_epochs"])
    learning_rate: float = field(default=HYPERPARAMETERS["learning_rate"])
    weight_decay: float = field(default=HYPERPARAMETERS["weight_decay"])
    lr_scheduler_type: str = field(default=HYPERPARAMETERS["lr_scheduler_type"])
    warmup_steps: int = field(default=HYPERPARAMETERS["warmup_steps"])
    save_strategy: str = field(default=HYPERPARAMETERS["save_strategy"])
    evaluation_strategy: str = field(default=HYPERPARAMETERS["evaluation_strategy"])
    save_steps: int = field(default=HYPERPARAMETERS["save_steps"])
    eval_steps: int = field(default=HYPERPARAMETERS["eval_steps"])
    load_best_model_at_end: bool = field(default=HYPERPARAMETERS["load_best_model_at_end"])
    metric_for_best_model: str = field(default="eval_loss")
    save_total_limit: int = field(default=HYPERPARAMETERS["save_total_limit"])
    bf16: bool = field(default=HYPERPARAMETERS["bf16"])
    logging_steps: int = field(default=HYPERPARAMETERS["logging_steps"])
    report_to: List[str] = field(default_factory=lambda: HYPERPARAMETERS["report_to"])
    disable_tqdm: bool = field(default=HYPERPARAMETERS["disable_tqdm"])
    dataloader_num_workers: int = field(default=HYPERPARAMETERS["dataloader_num_workers"])
    remove_unused_columns: bool = field(default=HYPERPARAMETERS["remove_unused_columns"])
    ddp_find_unused_parameters: bool = field(default=HYPERPARAMETERS["ddp_find_unused_parameters"])
    ddp_broadcast_buffers: bool = field(default=HYPERPARAMETERS["ddp_broadcast_buffers"])
    max_training_time_hours: float = field(default=HYPERPARAMETERS["max_training_time_hours"])

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

class TensorBoardCopyCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def on_train_end(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", 0)) == 0:
            import shutil
            tb_logs_src = os.path.join(args.output_dir, "runs")
            tb_logs_dst = os.path.join(self.output_dir, "tensorboard_logs")
            if os.path.exists(tb_logs_src):
                try:
                    if os.path.exists(tb_logs_dst): shutil.rmtree(tb_logs_dst)
                    shutil.copytree(tb_logs_src, tb_logs_dst)
                except Exception as e:
                    rank_print(f"Warning: Failed to copy TensorBoard logs: {e}")
            else:
                tb_logs_alt = os.path.join(args.output_dir, ".tensorboard")
                if os.path.exists(tb_logs_alt):
                    try:
                        if os.path.exists(tb_logs_dst): shutil.rmtree(tb_logs_dst)
                        shutil.copytree(tb_logs_alt, tb_logs_dst)
                    except Exception as e: pass

class StratifiedDistributedSampler(Sampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, seed=42, shuffle=True):
        if num_replicas is None:
            if not torch.distributed.is_available(): raise RuntimeError("Requires distributed")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available(): raise RuntimeError("Requires distributed")
            rank = torch.distributed.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle
        self.num_samples = int(np.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
    
    def __len__(self): return self.num_samples
    def set_epoch(self, epoch): self.epoch = epoch


@dataclass
class SmartMultimodalCollator:
    tokenizer: Any
    patch_size: int
    image_size: List[int]
    num_channels: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_pixel = any(b.get('pixel_values') is not None for b in batch)
        has_text = any(b.get('input_ids') is not None for b in batch)
        batch_out = {}

        if has_pixel:
            max_width = 0
            for item in batch:
                if item.get('pixel_values') is not None:
                    max_width = max(max_width, item['pixel_values'].shape[2])
            
            # Cap at hard image_size limit to prevent OOM
            max_width = min(max_width, self.image_size[1])
            if max_width == 0: max_width = self.patch_size # safe fallback
            
            bsz = len(batch)
            height = self.image_size[0]
            
            # Pre-allocate batch with 1.0s (white pixels)
            pixel_batch = torch.ones((bsz, self.num_channels, height, max_width), dtype=torch.float32)
            
            # Calculate max patches for the attention mask
            num_patches = (height // self.patch_size) * (max_width // self.patch_size)
            pixel_mask_batch = torch.zeros((bsz, num_patches), dtype=torch.long)
            
            for i, item in enumerate(batch):
                if item.get('pixel_values') is not None:
                    pixels = item['pixel_values']
                    current_width = min(pixels.shape[2], max_width)
                    
                    # Copy image data directly into the pre-allocated white tensor
                    pixel_batch[i, :, :, :current_width] = pixels[:, :, :current_width]
                    
                    # Create pixel attention mask
                    h = pixels.shape[1]
                    original_patches = (h // self.patch_size) * (current_width // self.patch_size)
                    pixel_mask_batch[i, :original_patches] = 1
            
            batch_out['pixel_values'] = pixel_batch
            batch_out['pixel_attention_mask'] = pixel_mask_batch

        if has_text:
            input_ids_list = []
            labels_list = []
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            for item in batch:
                if item.get('input_ids') is not None:
                    txt = torch.tensor(item['input_ids'], dtype=torch.long)
                    input_ids_list.append(txt)
                    labels_list.append(txt.clone())
                else:
                    input_ids_list.append(torch.tensor([pad_id], dtype=torch.long))
                    labels_list.append(torch.tensor([-100], dtype=torch.long))
            
            batch_out['input_ids'] = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            batch_out['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)
            
            text_mask = (batch_out['input_ids'] != pad_id).long()
            for i, item in enumerate(batch):
                if item.get('input_ids') is None: text_mask[i] = 0
            batch_out['attention_mask'] = text_mask

        return batch_out

def make_lazy_transform(tokenizer, image_col, text_col, max_len, image_size, num_channels):
    """Create lazy transform function for on-the-fly processing."""
    image_transform = transforms.Compose([
        transforms.ToTensor(), # Automatically scales 0-255 uint8 to 0.0-1.0 float32
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
                    if current_ids and current_ids[-1] == eos_token_id: current_ids.pop()
                    if len(current_ids) > max_len - 1: current_ids = current_ids[:max_len - 1]
                    current_ids.append(eos_token_id)
                    p_ids = current_ids
            
            p_img = None
            if mode in [0, 2]:
                raw_img = batch[image_col][i]
                if raw_img is not None:
                    try:
                        np_img = np.array(raw_img, dtype=np.uint8)
                        pil_image = Image.fromarray(np_img)
                        # Duplicates grayscale to 3 channels for VisualTransformer compatibility
                        if num_channels == 3 and pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        p_img = image_transform(pil_image)
                    except Exception as e:
                        rank_print(f"Warning: Failed to process image at index {i}: {e}")
                        p_img = None 
            
            processed_ids.append(p_ids)
            processed_pixels.append(p_img)
        
        return {"input_ids": processed_ids, "pixel_values": processed_pixels}

    return lazy_process

def main():
    rank_print("Script Starting...")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.dataloader_num_workers > 0:
        training_args.dataloader_num_workers = 0

    is_distributed = torch.distributed.is_initialized()
    if is_distributed:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    training_args.remove_unused_columns = False
    set_seed(training_args.seed)
    
    tokenizer_source = model_args.model_name_or_path if model_args.model_name_or_path else model_args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    common_features = Features({
        "pixel_values": Sequence(Sequence(Value("uint8"))),
        data_args.text_column: Sequence(Value("int64")), 
        "mode": Value("int64")
    })

    datasets_to_merge_train = []
    datasets_to_merge_val = []
    dataset_sources = [] 

    def prepare_single_dataset(name, dataset_idx):
        ds = load_dataset(name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        ds = ds.add_column("mode", [0] * len(ds))
        ds = ds.select_columns(["pixel_values", data_args.text_column, "mode"]).cast(common_features)
        
        total_len = len(ds)
        val_size = data_args.validation_samples_per_dataset
        if total_len < val_size * 2: val_size = int(total_len * 0.1) 
            
        all_indices = np.arange(total_len)
        rng = np.random.default_rng(42)
        rng.shuffle(all_indices)
        
        val_ds = ds.select(all_indices[:val_size])
        train_ds = ds.select(all_indices[val_size:])
        return train_ds, val_ds

    train_a, val_a = prepare_single_dataset(data_args.dataset_a_name, 0)
    datasets_to_merge_train.append(train_a)
    datasets_to_merge_val.append(val_a)
    dataset_sources.extend([0] * len(train_a))

    if data_args.dataset_b_name:
        train_b, val_b = prepare_single_dataset(data_args.dataset_b_name, 1)
        datasets_to_merge_train.append(train_b)
        datasets_to_merge_val.append(val_b)
        dataset_sources.extend([1] * len(train_b))
    
    train_dataset = concatenate_datasets(datasets_to_merge_train)
    eval_dataset = concatenate_datasets(datasets_to_merge_val)
    
    weights = [data_args.dataset_a_weight if src == 0 else data_args.dataset_b_weight for src in dataset_sources]

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
    if hasattr(model.config, "use_cache"): model.config.use_cache = False
    
    transform_fn = make_lazy_transform(
        tokenizer, 
        data_args.image_column, 
        data_args.text_column,
        data_args.max_seq_length,
        model_args.image_size,
        model_args.num_channels
    )
    
    train_dataset.set_transform(transform_fn)
    eval_dataset.set_transform(transform_fn)

    data_collator = SmartMultimodalCollator(
        tokenizer=tokenizer,
        patch_size=model_args.patch_size,
        image_size=model_args.image_size,
        num_channels=model_args.num_channels
    )

    class StratifiedTrainer(Trainer):
        def _get_train_sampler(self):
            if is_distributed:
                return StratifiedDistributedSampler(
                    self.train_dataset, weights=weights, num_replicas=world_size,
                    rank=rank, seed=self.args.seed, shuffle=True
                )
            else:
                from torch.utils.data import WeightedRandomSampler
                return WeightedRandomSampler(weights=weights, num_samples=len(self.train_dataset), replacement=True)

    trainer = StratifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5), 
            TimeLimitCallback(time_limit_hours=training_args.max_training_time_hours), 
            TensorBoardCopyCallback(output_dir=training_args.output_dir)
        ],
    )

    # --- NEW LOGGING BLOCK START ---
    if rank == 0:
        # Calculate exactly how the DataLoaders/Trainer will split the data
        samples_per_device = math.ceil(len(train_dataset) / world_size)
        batches_per_device = math.ceil(samples_per_device / training_args.per_device_train_batch_size)
        steps_per_epoch = math.ceil(batches_per_device / training_args.gradient_accumulation_steps)
        total_steps = math.ceil(steps_per_epoch * training_args.num_train_epochs)
        
        effective_batch_size = (
            training_args.per_device_train_batch_size * 
            training_args.gradient_accumulation_steps * 
            world_size
        )
        
        rank_print("=" * 60)
        rank_print("📊 DATASET & TRAINING METRICS 📊")
        rank_print("=" * 60)
        rank_print(f"1. Training Dataset Length : {len(train_dataset):,} samples")
        rank_print(f"2. Eval Dataset Length     : {len(eval_dataset):,} samples")
        rank_print(f"3. Effective Batch Size    : {effective_batch_size} samples/update")
        rank_print(f"   - Per Device Batch Size : {training_args.per_device_train_batch_size}")
        rank_print(f"   - Grad Accumulation     : {training_args.gradient_accumulation_steps}")
        rank_print(f"   - World Size (GPUs)     : {world_size}")
        rank_print(f"4. Expected Training Steps : ~{steps_per_epoch:,} steps/epoch")
        rank_print(f"5. Total Training Steps    : ~{total_steps:,} steps (at {training_args.num_train_epochs} epochs)")
        rank_print("=" * 60)
    # --- NEW LOGGING BLOCK END ---

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    if rank == 0:
        trainer.save_model()
        trainer.save_state()
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()