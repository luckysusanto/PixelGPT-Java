"""
Batch Diagnostic Script for DualGPT / mT5 OOM Investigation
============================================================
Simulates the exact DataLoader/Sampler setup from training,
fast-forwards to step 6688, and inspects that minibatch.
Also samples a random batch for comparison.

Key fix vs naive simulation:
  StratifiedDistributedSampler splits the full index list across ranks via
      indices[rank::num_replicas]
  So rank 0 only ever sees every 2nd sampled index. This script replicates
  that slicing exactly so the minibatch sequence matches training rank 0.

Usage:
    python diagnose_batch.py \
        --dataset_a_name <path_or_hf_name_dataset_a> \
        --dataset_b_name <path_or_hf_name_dataset_b> \
        --tokenizer_path <path_or_hf_name_tokenizer> \
        [--target_step 6688] \
        [--effective_batch_size 32] \
        [--per_device_batch_size 2] \
        [--grad_accum_steps 8] \
        [--world_size 2] \
        [--seed 42] \
        [--max_seq_length 1024] \
        [--validation_samples 1000]
"""

import argparse
import gc
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets, Features, Sequence, Value
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk 


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="DualGPT batch diagnostic")
    p.add_argument("--dataset_a_name",        required=True)
    p.add_argument("--dataset_b_name",        required=True)
    p.add_argument("--tokenizer_path",        required=True)
    p.add_argument("--target_step",           type=int,   default=6688)
    p.add_argument("--effective_batch_size",  type=int,   default=32)
    p.add_argument("--per_device_batch_size", type=int,   default=2)
    p.add_argument("--grad_accum_steps",      type=int,   default=8)
    p.add_argument("--world_size",            type=int,   default=2)
    p.add_argument("--seed",                  type=int,   default=42)
    p.add_argument("--max_seq_length",        type=int,   default=1024)
    p.add_argument("--validation_samples",    type=int,   default=1000)
    p.add_argument("--image_size",            type=int,   nargs=2, default=[16, 16384])
    p.add_argument("--patch_size",            type=int,   default=16)
    p.add_argument("--num_channels",          type=int,   default=3)
    p.add_argument("--cache_dir",             type=str,   default=None)
    p.add_argument("--comparison_step",       type=int,   default=None,
                   help="A random step to compare against. Defaults to target_step // 2.")
    return p.parse_args()


# =============================================================================
# Replicated helpers from training script
# =============================================================================
def make_lazy_transform(tokenizer, image_col, text_col, max_len, image_size, num_channels):
    image_transform = transforms.Compose([transforms.ToTensor()])
    eos_token_id = tokenizer.eos_token_id

    def lazy_process(batch):
        batch_len = len(batch[next(iter(batch))])
        processed_pixels, processed_ids = [], []
        modes = batch.get("mode", [0] * batch_len)

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
                        current_ids = current_ids[: max_len - 1]
                    current_ids.append(eos_token_id)
                    p_ids = current_ids

            p_img = None
            if mode in [0, 2]:
                raw_img = batch[image_col][i]
                if raw_img is not None:
                    try:
                        np_img = np.array(raw_img, dtype=np.uint8)
                        pil_image = Image.fromarray(np_img)
                        if num_channels == 3 and pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")
                        p_img = image_transform(pil_image)
                    except Exception as e:
                        print(f"  [WARN] Image processing failed at index {i}: {e}")

            processed_ids.append(p_ids)
            processed_pixels.append(p_img)

        return {"input_ids": processed_ids, "pixel_values": processed_pixels}

    return lazy_process


@dataclass
class SmartMultimodalCollator:
    tokenizer: Any
    patch_size: int
    image_size: List[int]
    num_channels: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_pixel = any(b.get("pixel_values") is not None for b in batch)
        has_text  = any(b.get("input_ids")    is not None for b in batch)
        batch_out = {}

        if has_pixel:
            max_width = 0
            for item in batch:
                if item.get("pixel_values") is not None:
                    max_width = max(max_width, item["pixel_values"].shape[2])
            max_width = min(max_width, self.image_size[1])
            if max_width == 0:
                max_width = self.patch_size

            bsz    = len(batch)
            height = self.image_size[0]
            pixel_batch = torch.ones(
                (bsz, self.num_channels, height, max_width), dtype=torch.float32
            )
            num_patches      = (height // self.patch_size) * (max_width // self.patch_size)
            pixel_mask_batch = torch.zeros((bsz, num_patches), dtype=torch.long)

            for i, item in enumerate(batch):
                if item.get("pixel_values") is not None:
                    pixels        = item["pixel_values"]
                    current_width = min(pixels.shape[2], max_width)
                    pixel_batch[i, :, :, :current_width] = pixels[:, :, :current_width]
                    h = pixels.shape[1]
                    original_patches = (h // self.patch_size) * (current_width // self.patch_size)
                    pixel_mask_batch[i, :original_patches] = 1

            batch_out["pixel_values"]        = pixel_batch
            batch_out["pixel_attention_mask"] = pixel_mask_batch

        if has_text:
            input_ids_list, labels_list = [], []
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            for item in batch:
                if item.get("input_ids") is not None:
                    txt = torch.tensor(item["input_ids"], dtype=torch.long)
                    input_ids_list.append(txt)
                    labels_list.append(txt.clone())
                else:
                    input_ids_list.append(torch.tensor([pad_id], dtype=torch.long))
                    labels_list.append(torch.tensor([-100], dtype=torch.long))

            batch_out["input_ids"] = pad_sequence(
                input_ids_list, batch_first=True, padding_value=pad_id
            )
            batch_out["labels"] = pad_sequence(
                labels_list, batch_first=True, padding_value=-100
            )
            text_mask = (batch_out["input_ids"] != pad_id).long()
            for i, item in enumerate(batch):
                if item.get("input_ids") is None:
                    text_mask[i] = 0
            batch_out["attention_mask"] = text_mask

        return batch_out


# =============================================================================
# Dataset construction (mirrors training script exactly)
# =============================================================================
def build_datasets(args, tokenizer):
    common_features = Features({
        "pixel_values": Sequence(Sequence(Value("uint8"))),
        "tok_mt5":    Sequence(Value("int64")),
        "mode":         Value("int64"),
    })

    def prepare_single_dataset(name):
        print(f"    Loading {name}...", flush=True)
        ds = load_dataset(name, split="train")
        print(f"    Loaded {len(ds):,} samples. Adding mode column...", flush=True)
        
        ds = ds.add_column("mode", [0] * len(ds))
        print(f"    Mode column added. Selecting columns...", flush=True)
        
        ds = ds.select_columns(["pixel_values", "tok_mt5", "mode"])
        print(f"    Columns selected. Casting features...", flush=True)
        
        ds = ds.cast(common_features, keep_in_memory=True)
        print(f"    Cast complete. Splitting train/val...", flush=True)

        total_len = len(ds)
        val_size  = args.validation_samples
        if total_len < val_size * 2:
            val_size = int(total_len * 0.1)

        all_indices = np.arange(total_len)
        rng = np.random.default_rng(42)
        rng.shuffle(all_indices)

        val_ds   = ds.select(all_indices[:val_size])
        train_ds = ds.select(all_indices[val_size:])
        print(f"    Done. Train: {len(train_ds):,}, Val: {len(val_ds):,}", flush=True)
        return train_ds, val_ds

    train_a, val_a = prepare_single_dataset(args.dataset_a_name)
    train_b, val_b = prepare_single_dataset(args.dataset_b_name)

    dataset_sources = [0] * len(train_a) + [1] * len(train_b)
    train_dataset   = concatenate_datasets([train_a, train_b])

    print(f"\n  Dataset A (train): {len(train_a):,} samples")
    print(f"  Dataset B (train): {len(train_b):,} samples")
    print(f"  Combined  (train): {len(train_dataset):,} samples")
    print(f"  Dataset A boundary starts at index 0")
    print(f"  Dataset B boundary starts at index {len(train_a):,}")

    weights = [1.0 if src == 0 else 1.0 for src in dataset_sources]

    transform_fn = make_lazy_transform(
        tokenizer,
        "pixel_values",
        "tok_mt5",
        args.max_seq_length,
        args.image_size,
        args.num_channels,
    )
    train_dataset.set_transform(transform_fn)

    return train_dataset, weights, len(train_a)


# =============================================================================
# Sampler — exact replica of StratifiedDistributedSampler for rank 0
# =============================================================================
def build_rank0_indices(dataset, weights, seed, world_size, epoch=0):
    """
    Replicates StratifiedDistributedSampler.__iter__ for rank=0, epoch=0.

    The distributed sampler:
      1. Generates total_size = ceil(N / world_size) * world_size indices via
         torch.multinomial(weights, total_size, replacement=True)
      2. Pads to total_size
      3. Slices rank i's portion as indices[rank::num_replicas]

    We do exactly that here and return the rank-0 slice as a plain list,
    which is then fed to a SequentialSampler-style DataLoader so the order
    is byte-for-byte identical to what rank 0 saw during training.
    """
    num_replicas = world_size
    rank         = 0  # we always simulate rank 0
    n            = len(dataset)
    num_samples  = math.ceil(n / num_replicas)
    total_size   = num_samples * num_replicas

    g = torch.Generator()
    g.manual_seed(seed + epoch)  # matches: self.seed + self.epoch

    indices = torch.multinomial(
        torch.as_tensor(weights, dtype=torch.double),
        total_size,
        replacement=True,
        generator=g,
    ).tolist()

    # Pad if needed (mirrors the sampler's padding logic)
    indices += indices[: (total_size - len(indices))]
    assert len(indices) == total_size

    # Slice rank 0
    rank0_indices = indices[rank:total_size:num_replicas]
    assert len(rank0_indices) == num_samples

    return rank0_indices


# =============================================================================
# Batch inspector
# =============================================================================
def inspect_batch(batch, label, dataset_a_boundary, collator):
    """Print detailed statistics for a collated batch."""
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  BATCH INSPECTION: {label}")
    print(sep)

    # --- Raw sample indices & language origin ---
    if "sample_indices" in batch:
        indices = batch["sample_indices"]
        lang_a  = sum(1 for idx in indices if idx < dataset_a_boundary)
        lang_b  = len(indices) - lang_a
        print(f"\n  Sample indices  : {indices}")
        print(f"  Language origin : {lang_a} from Dataset A  |  {lang_b} from Dataset B")

    # --- Text stats ---
    if "input_ids" in batch:
        ids      = batch["input_ids"]          # [B, T]
        seq_lens = (ids != collator.tokenizer.pad_token_id).sum(dim=1).tolist()
        print(f"\n  [TEXT]")
        print(f"    input_ids shape : {list(ids.shape)}")
        print(f"    seq lengths     : {seq_lens}")
        print(f"    min / max / mean: {min(seq_lens)} / {max(seq_lens)} / {sum(seq_lens)/len(seq_lens):.1f}")
        text_mem_mb = ids.element_size() * ids.nelement() / 1e6
        print(f"    tensor memory   : {text_mem_mb:.2f} MB")

    # --- Pixel stats ---
    if "pixel_values" in batch:
        pv = batch["pixel_values"]             # [B, C, H, W]
        print(f"\n  [IMAGE]")
        print(f"    pixel_values shape : {list(pv.shape)}")
        per_image_widths = []
        if "pixel_attention_mask" in batch:
            mask = batch["pixel_attention_mask"]   # [B, num_patches]
            for i in range(mask.shape[0]):
                active_patches = mask[i].sum().item()
                patch_cols     = active_patches // (pv.shape[2] // collator.patch_size)
                per_image_widths.append(patch_cols * collator.patch_size)
            print(f"    active widths (px) : {per_image_widths}")
            print(f"    min / max / mean   : "
                  f"{min(per_image_widths)} / {max(per_image_widths)} / "
                  f"{sum(per_image_widths)/len(per_image_widths):.1f}")
        pixel_mem_mb = pv.element_size() * pv.nelement() / 1e6
        print(f"    tensor memory      : {pixel_mem_mb:.2f} MB")
        print(f"    padded width       : {pv.shape[3]} px  "
              f"({'NEAR MAX' if pv.shape[3] > 12000 else 'normal'})")

    # --- Total collated memory ---
    total_mb = sum(
        t.element_size() * t.nelement() / 1e6
        for t in batch.values()
        if isinstance(t, torch.Tensor)
    )
    print(f"\n  Total collated batch memory : {total_mb:.2f} MB")
    print(sep)


# =============================================================================
# Fast-forward to target step
# =============================================================================
def fast_forward_to_step(loader, target_minibatch_idx, label):
    """
    Advance the DataLoader iterator to target_minibatch_idx without
    fully processing each batch (just iterate).
    """
    print(f"\n  Fast-forwarding to minibatch {target_minibatch_idx} ({label})...")
    iterator = iter(loader)
    for i in range(target_minibatch_idx - 1):
        try:
            next(iterator)
        except StopIteration:
            print(f"  [WARN] DataLoader exhausted at minibatch {i+1} before reaching target.")
            return None, i + 1
        if i % 1000 == 0 and i > 0:
            print(f"    ... at minibatch {i}")
    try:
        batch = next(iterator)
    except StopIteration:
        print("  [WARN] DataLoader exhausted exactly at target.")
        return None, target_minibatch_idx
    return batch, target_minibatch_idx


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    # Derived quantities
    # Each "step" consumes grad_accum_steps minibatches per GPU.
    # We simulate rank=0 only, so per_device_batch_size applies directly.
    minibatches_per_step = args.grad_accum_steps
    target_minibatch     = args.target_step * minibatches_per_step
    comparison_step      = args.comparison_step or (args.target_step // 2)
    comparison_minibatch = comparison_step * minibatches_per_step

    print("\n" + "=" * 60)
    print("  DualGPT Batch Diagnostic")
    print("=" * 60)
    print(f"  Target step            : {args.target_step}")
    print(f"  Target minibatch index : {target_minibatch}")
    print(f"  Comparison step        : {comparison_step}")
    print(f"  Comparison minibatch   : {comparison_minibatch}")
    print(f"  Per-device batch size  : {args.per_device_batch_size}")
    print(f"  Grad accum steps       : {args.grad_accum_steps}")
    print(f"  Effective batch size   : {args.effective_batch_size}")

    # --- Tokenizer ---
    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"  Vocab size: {len(tokenizer):,}")

    # --- Datasets ---
    print("\n  Loading datasets...")
    train_dataset, weights, dataset_a_boundary = build_datasets(args, tokenizer)

    # --- Collator ---
    collator = SmartMultimodalCollator(
        tokenizer=tokenizer,
        patch_size=args.patch_size,
        image_size=args.image_size,
        num_channels=args.num_channels,
    )

    # Wrap collator to also record sample indices
    original_call = collator.__call__
    _current_indices = []

    def collator_with_indices(batch):
        result = original_call(batch)
        result["sample_indices"] = list(_current_indices)
        return result

    # --- Sampler & DataLoader ---
    # Build rank-0 index sequence — identical to what StratifiedDistributedSampler
    # would have produced for rank 0 during training.
    rank0_indices = build_rank0_indices(
        train_dataset, weights, args.seed, args.world_size, epoch=0
    )
    print(f"\n  Rank-0 samples per epoch : {len(rank0_indices):,}")
    print(f"  (Full dataset: {len(train_dataset):,}, world_size: {args.world_size})")

    # We need index tracking, so use a custom collate wrapper via a custom Dataset
    class IndexTrackingDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            _current_indices.append(idx)
            return self.ds[idx]

    def make_loader(indices):
        """Build a DataLoader that iterates rank0_indices in exact order."""
        tracked = IndexTrackingDataset(train_dataset)
        # SubsetRandomSampler with shuffle=False preserves our pre-computed order
        sampler = torch.utils.data.SequentialSampler(indices)

        # We wrap indices as a plain dataset so SequentialSampler gives us
        # positions into rank0_indices, then we remap in __getitem__.
        class MappedDataset(torch.utils.data.Dataset):
            def __init__(self, base_ds, idx_list):
                self.base_ds  = base_ds
                self.idx_list = idx_list
            def __len__(self):
                return len(self.idx_list)
            def __getitem__(self, pos):
                real_idx = self.idx_list[pos]
                _current_indices.append(real_idx)
                return self.base_ds[real_idx]

        mapped = MappedDataset(train_dataset, indices)
        return DataLoader(
            mapped,
            batch_size=args.per_device_batch_size,
            shuffle=False,          # order is already determined by rank0_indices
            collate_fn=collator_with_indices,
            num_workers=0,
            drop_last=False,
        )

    total_minibatches = math.ceil(len(rank0_indices) / args.per_device_batch_size)

    print(f"  Total minibatches per epoch   : {total_minibatches:,}")
    print(f"  Dataset A / B boundary        : index {dataset_a_boundary:,}")

    # -------------------------------------------------------------------------
    # 1. Comparison batch (earlier step)
    # -------------------------------------------------------------------------
    _current_indices.clear()
    loader1 = make_loader(rank0_indices)
    comparison_batch, reached = fast_forward_to_step(
        loader1, comparison_minibatch, f"step {comparison_step}"
    )
    if comparison_batch:
        inspect_batch(
            comparison_batch,
            f"COMPARISON — step {comparison_step} (minibatch {comparison_minibatch})",
            dataset_a_boundary,
            collator,
        )
    else:
        print(f"  [WARN] Could not reach comparison minibatch {comparison_minibatch}")

    # -------------------------------------------------------------------------
    # 2. Target batch (OOM step) — fresh loader from same rank0_indices
    # -------------------------------------------------------------------------
    _current_indices.clear()
    loader2 = make_loader(rank0_indices)
    target_batch, reached = fast_forward_to_step(
        loader2, target_minibatch, f"step {args.target_step}"
    )
    if target_batch:
        inspect_batch(
            target_batch,
            f"TARGET — step {args.target_step} (minibatch {target_minibatch})  ← OOM STEP",
            dataset_a_boundary,
            collator,
        )
    else:
        print(f"  [WARN] Could not reach target minibatch {target_minibatch}")

    # -------------------------------------------------------------------------
    # 3. Summary delta
    # -------------------------------------------------------------------------
    if comparison_batch and target_batch:
        print("\n" + "=" * 60)
        print("  DELTA SUMMARY")
        print("=" * 60)

        def get_max_width(b):
            if "pixel_values" in b:
                return b["pixel_values"].shape[3]
            return 0

        def get_max_seq(b, pad_id):
            if "input_ids" in b:
                return (b["input_ids"] != pad_id).sum(dim=1).max().item()
            return 0

        pad_id = tokenizer.pad_token_id or 0

        cmp_width = get_max_width(comparison_batch)
        tgt_width = get_max_width(target_batch)
        cmp_seq   = get_max_seq(comparison_batch, pad_id)
        tgt_seq   = get_max_seq(target_batch, pad_id)

        cmp_mem = sum(t.element_size() * t.nelement() / 1e6
                      for t in comparison_batch.values() if isinstance(t, torch.Tensor))
        tgt_mem = sum(t.element_size() * t.nelement() / 1e6
                      for t in target_batch.values() if isinstance(t, torch.Tensor))

        print(f"\n  {'Metric':<30} {'Comparison (step ' + str(comparison_step) + ')':<25} {'Target (step ' + str(args.target_step) + ')':<25}")
        print(f"  {'-'*78}")
        print(f"  {'Max image width (px)':<30} {cmp_width:<25} {tgt_width:<25} {'⚠ WIDE' if tgt_width > cmp_width * 1.5 else ''}")
        print(f"  {'Max seq length (tokens)':<30} {cmp_seq:<25} {tgt_seq:<25} {'⚠ LONG' if tgt_seq > cmp_seq * 1.5 else ''}")
        print(f"  {'Total batch memory (MB)':<30} {cmp_mem:<25.2f} {tgt_mem:<25.2f} {'⚠ LARGE' if tgt_mem > cmp_mem * 1.5 else ''}")
        print("=" * 60)

    print("\n  Diagnostic complete.\n")


if __name__ == "__main__":
    main()