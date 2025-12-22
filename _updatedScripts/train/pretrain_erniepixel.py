import os
import logging
import time
import sys
import gc
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
from torch.utils.data import Sampler
from torchvision import transforms

from src.ernie_pixel.configuration_ernie_pixel import ErniePixelConfig
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

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
    # Updated default to generic 'token_ids', but shell script will override to 'grapheme_token_ids'
    text_column: str = field(default="token_ids") 
    max_seq_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)
    validation_samples_per_dataset: int = field(default=1000)
    dataset_a_weight: float = field(default=1.0)
    dataset_b_weight: float = field(default=1.0)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/merged-pretrain")
    per_device_train_batch_size: int = field(default=8)
    num_train_epochs: float = field(default=3.0) 
    learning_rate: float = field(default=5e-4)
    weight_decay: float = field(default=0.1)   
    lr_scheduler_type: str = field(default='cosine')
    warmup_steps: int = field(default=1000)
    save_strategy: str = field(default="steps")
    evaluation_strategy: str = field(default="steps")
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    save_total_limit: int = field(default=2) 
    bf16: bool = field(default=True)
    logging_steps: int = field(default=100)
    dataloader_num_workers: int = field(default=0) # Forced 0
    remove_unused_columns: bool = field(default=False) 
    
    # --- CRITICAL FIXES FOR DDP ---
    ddp_find_unused_parameters: bool = field(default=True)
    ddp_broadcast_buffers: bool = field(default=False)
    
    max_training_time_hours: float = field(default=24.0)

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

class StratifiedDistributedSampler(Sampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, seed=42, shuffle=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package")
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
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

@dataclass
class SmartMultimodalCollator:
    tokenizer: Any
    patch_size: int
    image_size: List[int]
    num_channels: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_pixel = any(b['pixel_values'] is not None for b in batch)
        # Note: We rely on the transform to map text_column -> 'input_ids'
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
                # DYNAMIC TEXT COLUMN ACCESS
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
        # Note: We return 'input_ids' here for the Collator to pick up
        return {"input_ids": processed_ids, "pixel_values": processed_pixels}

    return lazy_process


def main():
    rank_print("Script Starting...")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.dataloader_num_workers > 0:
        training_args.dataloader_num_workers = 0

    # Auto-detection
    is_distributed = torch.distributed.is_initialized()
    if is_distributed:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    training_args.remove_unused_columns = False
    set_seed(training_args.seed)
    
    # -------------------------------------------------------------------------
    # TOKENIZER
    # -------------------------------------------------------------------------
    rank_print("Loading Tokenizer...")
    tokenizer_source = model_args.model_name_or_path if model_args.model_name_or_path else model_args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------
    rank_print("Loading and Merging Datasets...")
    
    # DYNAMIC SCHEMA: Use data_args.text_column instead of hardcoded name
    common_features = Features({
        "pixel_values": Sequence(Sequence(Value("uint8"))),
        data_args.text_column: Sequence(Value("int64")), 
        "mode": Value("int64")
    })

    datasets_to_merge_train = []
    datasets_to_merge_val = []
    dataset_sources = [] 

    def prepare_single_dataset(name, dataset_idx):
        rank_print(f"Processing {name}...")
        ds = load_dataset(name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        ds = ds.add_column("mode", [0] * len(ds))
        
        # DYNAMIC COLUMN SELECTION: Drop unused cols (text_id, llama_token_ids, etc)
        ds = ds.select_columns(["pixel_values", data_args.text_column, "mode"])
        
        # CAST to ensure 'grapheme_token_ids' is recognized consistently
        ds = ds.cast(common_features)
        
        # Safe Split (using indices, ignores provided 'test' split to ensure stratification size)
        total_len = len(ds)
        val_size = data_args.validation_samples_per_dataset
        if total_len < val_size * 2:
            val_size = int(total_len * 0.1) 
            
        all_indices = np.arange(total_len)
        rng = np.random.default_rng(42) # Deterministic
        rng.shuffle(all_indices)
        
        val_indices = all_indices[:val_size]
        train_indices = all_indices[val_size:]
        
        val_ds = ds.select(val_indices)
        train_ds = ds.select(train_indices)
        
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
    
    rank_print("Concatenating...")
    train_dataset = concatenate_datasets(datasets_to_merge_train)
    eval_dataset = concatenate_datasets(datasets_to_merge_val)
    
    weights = []
    for source_idx in dataset_sources:
        w = data_args.dataset_a_weight if source_idx == 0 else data_args.dataset_b_weight
        weights.append(w)
    
    rank_print(f"Dataset Ready. Train Size: {len(train_dataset)}")

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    if model_args.model_name_or_path:
        rank_print(f"Loading weights from: {model_args.model_name_or_path}")
        config = ErniePixelConfig.from_pretrained(model_args.model_name_or_path)
        model = ErniePixelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        rank_print(f"Initializing model from scratch.")
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
    
    # -------------------------------------------------------------------------
    # TRANSFORM & TRAINER
    # -------------------------------------------------------------------------
    transform_fn = make_lazy_transform(
        tokenizer, 
        data_args.image_column, 
        data_args.text_column, # This ensures we pick the correct column
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

    early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=5)
    time_limit_cb = TimeLimitCallback(time_limit_hours=training_args.max_training_time_hours)

    class StratifiedTrainer(Trainer):
        def _get_train_sampler(self):
            if is_distributed:
                return StratifiedDistributedSampler(
                    self.train_dataset,
                    weights=weights,
                    num_replicas=world_size,
                    rank=rank,
                    seed=self.args.seed,
                    shuffle=True
                )
            else:
                from torch.utils.data import WeightedRandomSampler
                return WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(self.train_dataset),
                    replacement=True
                )

    trainer = StratifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping_cb, time_limit_cb],
    )

    rank_print("Calling trainer.train()...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    if rank == 0:
        rank_print(f"Saving best model to {training_args.output_dir}")
        trainer.save_model()
        trainer.save_state()
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()