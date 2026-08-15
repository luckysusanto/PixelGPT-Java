import os
import logging
import time
import sys
import gc
import datetime
import math
import csv
import atexit
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
    "per_device_train_batch_size": 4, # PASTIKAN per_device_train_batch_size x gradient_accumulation_steps = 16
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4, # PASTIKAN per_device_train_batch_size x gradient_accumulation_steps = 16
    "num_train_epochs": 5.0,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "save_strategy": "steps",
    "evaluation_strategy": "steps",
    "save_steps": 1000,
    "eval_steps": 1000,
    "load_best_model_at_end": True,
    "save_total_limit": 2,
    "early_stopping_patience": 5,
    "logging_steps": 50,
    "report_to": ["tensorboard"],
    "disable_tqdm": False,
    "max_seq_length": 1024,
    "validation_samples_per_dataset": 1000,
    "bf16": True,
    "dataloader_num_workers": 0,
    "remove_unused_columns": False,
    "ddp_find_unused_parameters": True,
    "ddp_broadcast_buffers": False,
    "max_training_time_hours": 24.0,
}

# =============================================================================
# 1. CRITICAL CLUSTER FIXES
# =============================================================================
os.environ["HF_DATASETS_LOCKING_DISABLED"] = "true"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = False

def rank_print(msg):
    try:
        rank = int(os.environ.get("RANK", 0))
    except:
        rank = 0
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[NO-BARRIER | RANK {rank} | {ts}] {msg}", flush=True)


# =============================================================================
# LOGGING UTILITIES
# =============================================================================
class BatchTokenLogger:
    """Logs per-micro-batch token-length stats to CSV with crash-safe flushing.

    Each row = one collator call (one micro-batch on this rank).
    `effective_step` = micro_step // grad_accum_steps, so you can group later.
    Opened with line buffering + flush-per-row + atexit close so OOM preserves all data.
    """
    def __init__(self, csv_path: str, grad_accum_steps: int, rank: int, world_size: int):
        base, ext = os.path.splitext(csv_path)
        self.csv_path = f"{base}_rank{rank}{ext or '.csv'}"
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.rank = rank
        self.world_size = world_size
        self.micro_step = 0
        self.fh = None
        self.writer = None
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.fh = open(self.csv_path, "w", newline="", buffering=1)
        self.writer = csv.writer(self.fh)
        self.writer.writerow([
            "micro_step", "effective_step", "batch_size",
            "token_min", "token_max", "token_avg",
            "pixel_width_min", "pixel_width_max", "pixel_width_avg",
            "wall_time", "gpu_mem_alloc_gb", "gpu_mem_reserved_gb",
        ])
        self.fh.flush()
        atexit.register(self.close)

    def log(self, input_ids_lengths: List[int], pixel_widths: List[int], batch_size: int):
        if self.writer is None:
            self.micro_step += 1
            return
        eff_step = self.micro_step // self.grad_accum_steps
        tl = input_ids_lengths if input_ids_lengths else [0]
        pw = pixel_widths if pixel_widths else [0]
        wall = datetime.datetime.now().isoformat(timespec="microseconds")
        try:
            mem_alloc_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
            mem_reserved_gb = round(torch.cuda.memory_reserved() / (1024 ** 3), 3)
        except Exception:
            mem_alloc_gb = -1.0
            mem_reserved_gb = -1.0
        self.writer.writerow([
            self.micro_step,
            eff_step,
            batch_size,
            min(tl), max(tl), round(sum(tl) / len(tl), 2),
            min(pw), max(pw), round(sum(pw) / len(pw), 2),
            wall, mem_alloc_gb, mem_reserved_gb,
        ])
        self.fh.flush()
        try:
            os.fsync(self.fh.fileno())
        except Exception:
            pass
        self.micro_step += 1

    def close(self):
        if self.fh is not None:
            try:
                self.fh.flush()
                self.fh.close()
            except Exception:
                pass
            self.fh = None


def compute_token_length_stats(dataset, text_column: str, label: str):
    """Compute min/max/avg/percentile token length over a HF dataset.

    Accesses the raw column directly to avoid triggering set_transform.
    """
    col = dataset[text_column]
    lengths = []
    for ids in col:
        if ids is None:
            continue
        lengths.append(len(ids))
    if not lengths:
        rank_print(f"[TOKEN STATS | {label}] No samples with '{text_column}' found.")
        return
    n = len(lengths)
    mn = min(lengths)
    mx = max(lengths)
    avg = sum(lengths) / n
    arr = np.array(lengths)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    rank_print(
        f"[TOKEN STATS | {label}] n={n:,} | min={mn} | max={mx} | avg={avg:.2f} "
        f"| p50={p50:.0f} | p95={p95:.0f} | p99={p99:.0f}"
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the pretrained ErniePixelForCausalLM checkpoint"})
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

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="../experiment_output/dualgpt-finetune-transliteration")
    per_device_train_batch_size: int = field(default=HYPERPARAMETERS["per_device_train_batch_size"])
    per_device_eval_batch_size: int = field(default=HYPERPARAMETERS["per_device_eval_batch_size"])
    gradient_accumulation_steps: int = field(default=HYPERPARAMETERS["gradient_accumulation_steps"])
    num_train_epochs: float = field(default=HYPERPARAMETERS["num_train_epochs"])
    learning_rate: float = field(default=HYPERPARAMETERS["learning_rate"])
    weight_decay: float = field(default=HYPERPARAMETERS["weight_decay"])
    lr_scheduler_type: str = field(default=HYPERPARAMETERS["lr_scheduler_type"])
    warmup_ratio: float = field(default=HYPERPARAMETERS["warmup_ratio"])
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
            rank_print("Time limit reached.")
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

class MemoryCleanupCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

    def on_save(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

def create_white_patch(patch_size: int, num_channels: int) -> torch.Tensor:
    """Create a white patch (1.0 in all channels) for padding."""
    return torch.ones((num_channels, patch_size, patch_size), dtype=torch.float32)

@dataclass
class TransliterationCollator:
    tokenizer: Any
    patch_size: int
    image_size: List[int]
    num_channels: int
    # Optional logger; None => no logging overhead
    batch_logger: Optional[BatchTokenLogger] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_pixel = any(b['pixel_values'] is not None for b in batch)
        has_text = any(b['input_ids'] is not None for b in batch)
        batch_out = {}

        # Collect per-sample stats for logging
        _per_sample_token_lens: List[int] = []
        _per_sample_pixel_widths: List[int] = []

        if has_pixel:
            pixel_list = []
            pixel_mask_list = []

            max_width = 0
            for item in batch:
                if item.get('pixel_values') is not None:
                    max_width = max(max_width, item['pixel_values'].shape[2])

            max_width = min(max_width, self.image_size[1])
            if max_width == 0: max_width = self.patch_size

            white_patch = create_white_patch(self.patch_size, self.num_channels)

            for item in batch:
                if item.get('pixel_values') is not None:
                    pixels = item['pixel_values']

                    original_width = min(pixels.shape[2], max_width)
                    _per_sample_pixel_widths.append(int(pixels.shape[2]))  # log pre-cap width

                    if pixels.shape[2] > max_width:
                        pixels = pixels[:, :, :max_width]

                    current_width = pixels.shape[2]

                    if current_width < max_width:
                        padding_width = max_width - current_width
                        num_white_patches = padding_width // self.patch_size
                        assert padding_width % self.patch_size == 0, (
                            f"padding_width {padding_width} is not divisible by patch_size {self.patch_size}"
                        )
                        if num_white_patches > 0:
                            white_padding = white_patch.repeat(1, 1, num_white_patches)
                            pixels = torch.cat([pixels, white_padding], dim=2)

                    pixel_list.append(pixels)

                    h = pixels.shape[1]
                    num_patches = (h // self.patch_size) * (max_width // self.patch_size)
                    original_patches = (h // self.patch_size) * (original_width // self.patch_size)

                    mask = torch.zeros(num_patches, dtype=torch.long)
                    mask[:original_patches] = 1
                    pixel_mask_list.append(mask)
                else:
                    placeholder_img = torch.ones(
                        (self.num_channels, self.image_size[0], max_width),
                        dtype=torch.float32
                    )
                    pixel_list.append(placeholder_img)
                    num_patches = (self.image_size[0] // self.patch_size) * (max_width // self.patch_size)
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
                    labels_list.append(txt.clone())
                    _per_sample_token_lens.append(int(txt.shape[0]))  # log token lengths
                else:
                    input_ids_list.append(torch.tensor([pad_id], dtype=torch.long))
                    labels_list.append(torch.tensor([-100], dtype=torch.long))

            batch_out['input_ids'] = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            batch_out['labels'] = pad_sequence(labels_list, batch_first=True, padding_value=-100)

            text_mask = (batch_out['input_ids'] != pad_id).long()
            for i, item in enumerate(batch):
                if item.get('input_ids') is None: text_mask[i] = 0
            batch_out['attention_mask'] = text_mask

        # Flush one row per micro-batch
        if self.batch_logger is not None:
            self.batch_logger.log(
                input_ids_lengths=_per_sample_token_lens,
                pixel_widths=_per_sample_pixel_widths,
                batch_size=len(batch),
            )

        return batch_out

def make_finetune_transform(tokenizer, image_col, text_col, max_len, image_size, num_channels):
    """Create lazy transform function for on-the-fly processing."""
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    eos_token_id = tokenizer.eos_token_id

    def lazy_process(batch):
        batch_len = len(batch[next(iter(batch))])
        processed_pixels = []
        processed_ids = []

        for i in range(batch_len):
            p_ids = None

            raw_ids = batch[text_col][i]
            if raw_ids is not None and len(raw_ids) > 0:
                current_ids = list(raw_ids)
                if current_ids and current_ids[-1] == eos_token_id: current_ids.pop()
                if len(current_ids) > max_len - 1: current_ids = current_ids[:max_len - 1]
                current_ids.append(eos_token_id)
                p_ids = current_ids

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

    rank_print(f"Loading Tokenizer from checkpoint: {model_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    rank_print(f"Loading weights from {model_args.model_name_or_path}...")
    config = ErniePixelConfig.from_pretrained(model_args.model_name_or_path)
    config.dropout = 0.1

    model = ErniePixelForImageTransliteration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"): model.config.use_cache = False

    rank_print("Freezing lm_pixel_head...")
    if hasattr(model, 'lm_pixel_head'):
        for param in model.lm_pixel_head.parameters():
            param.requires_grad = False

    rank_print("Loading Datasets...")

    common_features = Features({
        "pixel_values": Sequence(Sequence(Value("uint8"))),
        data_args.text_column: Sequence(Value("int64")),
    })

    datasets_to_merge_train = []
    datasets_to_merge_val = []

    def prepare_single_dataset(name):
        rank_print(f"Processing {name}...")
        ds = load_dataset(name, split=data_args.train_split, cache_dir=data_args.cache_dir)
        ds = ds.select_columns(["pixel_values", data_args.text_column])
        ds = ds.cast(common_features)

        total_len = len(ds)
        val_size = data_args.validation_samples_per_dataset
        if total_len < val_size * 2:
            val_size = int(total_len * 0.1)

        all_indices = np.arange(total_len)
        rng = np.random.default_rng(42)
        rng.shuffle(all_indices)

        val_ds = ds.select(all_indices[:val_size])
        train_ds = ds.select(all_indices[val_size:])
        return train_ds, val_ds

    train_a, val_a = prepare_single_dataset(data_args.dataset_a_name)
    datasets_to_merge_train.append(train_a)
    datasets_to_merge_val.append(val_a)

    if data_args.dataset_b_name:
        train_b, val_b = prepare_single_dataset(data_args.dataset_b_name)
        datasets_to_merge_train.append(train_b)
        datasets_to_merge_val.append(val_b)

    # Compute token-length stats per dataset BEFORE concatenation/set_transform
    if rank == 0:
        rank_print("=" * 60)
        rank_print("📏 TOKEN LENGTH STATISTICS (pre-concat) 📏")
        rank_print("=" * 60)
        compute_token_length_stats(train_a, data_args.text_column, f"TRAIN-A ({data_args.dataset_a_name})")
        compute_token_length_stats(val_a, data_args.text_column, f"VAL-A ({data_args.dataset_a_name})")
        if data_args.dataset_b_name:
            compute_token_length_stats(train_b, data_args.text_column, f"TRAIN-B ({data_args.dataset_b_name})")
            compute_token_length_stats(val_b, data_args.text_column, f"VAL-B ({data_args.dataset_b_name})")
        rank_print("=" * 60)

    rank_print("Concatenating datasets...")
    train_dataset = concatenate_datasets(datasets_to_merge_train)
    eval_dataset = concatenate_datasets(datasets_to_merge_val)

    # Combined stats on the merged dataset
    if rank == 0:
        compute_token_length_stats(train_dataset, data_args.text_column, "TRAIN-MERGED")
        compute_token_length_stats(eval_dataset, data_args.text_column, "VAL-MERGED")
        rank_print("=" * 60)

    transform_fn = make_finetune_transform(
        tokenizer,
        data_args.image_column,
        data_args.text_column,
        data_args.max_seq_length,
        model_args.image_size,
        model_args.num_channels
    )
    train_dataset.set_transform(transform_fn)
    eval_dataset.set_transform(transform_fn)

    # Per-batch CSV logger (per-rank file: batch_token_stats_rank{N}.csv)
    csv_path = os.path.join(training_args.output_dir, "batch_token_stats.csv")
    batch_logger = BatchTokenLogger(
        csv_path=csv_path,
        grad_accum_steps=training_args.gradient_accumulation_steps,
        rank=rank,
        world_size=world_size,
    )
    rank_print(f"📝 Per-micro-batch stats (rank {rank}) streaming to: {batch_logger.csv_path}")

    data_collator = TransliterationCollator(
        tokenizer=tokenizer,
        patch_size=model_args.patch_size,
        image_size=model_args.image_size,
        num_channels=model_args.num_channels,
        batch_logger=batch_logger,  # pass logger into collator
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=HYPERPARAMETERS["early_stopping_patience"]),
            TimeLimitCallback(time_limit_hours=training_args.max_training_time_hours),
            TensorBoardCopyCallback(output_dir=training_args.output_dir),
            MemoryCleanupCallback(),
        ],
    )

    # Dataset & training metrics summary
    if rank == 0:
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

    rank_print("Starting Fine-Tuning...")
    try:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    finally:
        # Ensure CSV is flushed/closed even on OOM
        batch_logger.close()

    if rank == 0:
        rank_print("Saving Model...")
        trainer.save_model()
        trainer.save_state()
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()