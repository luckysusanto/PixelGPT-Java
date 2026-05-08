#!/usr/bin/env python
# coding=utf-8
"""
Evaluation script for ErniePixelForImageTransliteration.

Mirrors the finetuning input pipeline EXACTLY:
- Loads pre-trimmed pixel_values from the dataset (no Resize!)
- Casts pixel_values to Sequence(Sequence(Value("uint8"))) like finetuning
- Uses the actual content width for pixel_attention_mask (no hardcoded 1024)
- Compares predictions against the raw `text` column (no tokenizer round-trip)

Generation protocol (per-tokenizer, matching training-time conventions):
- BOS-having tokenizers (Llama-2, Komodo, grapheme):
    Step 0: feed [[bos_token_id]] with attention_mask=[[1]]
- mT5 (no BOS):
    Step 0: feed [[pad_token_id]] with attention_mask=[[0]] (masked-pad approximation)
    NOTE: This is a known inference/training mismatch since mT5 was trained without
    any sentinel at position 0. Acknowledged in paper methodology.

Metrics:
- chrF++ (primary, reported)
- CER (secondary, reported)
- WER (computed but acknowledged as unreliable on un-normalized references;
  saved for internal/rebuttal use only)

Outputs (written into the model directory):
- eval_result.json        : aggregate metrics + metadata
- eval_verbose.csv        : per-sample (text_id, text, pred, chrf++, CER, WER, ...)
"""

import os
import json
import csv
import logging
import argparse
from typing import Optional, List, Dict, Any

import torch
import numpy as np
from datasets import load_dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import jiwer
import sacrebleu

from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForImageTransliteration

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants (match finetuning) ---
PATCH_SIZE = 16
NUM_CHANNELS = 3
IMAGE_HEIGHT = 16
MAX_IMAGE_WIDTH = 16384  # cap; per-sample width should be much smaller


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a finetuned ErniePixel transliteration model.")

    # Required: model + tokenizer + dataset
    p.add_argument("--model_path", required=True, help="Path to the finetuned model checkpoint.")
    p.add_argument("--tokenizer_path", required=True, help="Path/HF-id of the tokenizer used for finetuning.")
    p.add_argument("--dataset_path", required=True,
                   help="HF dataset name (or local path) for the eval dataset. "
                        "Loaded with load_dataset(...) to mirror finetuning.")
    p.add_argument("--lang_code", required=True,
                   help="Language code for the eval set (used for log/output naming).")

    # Optional: where to write
    p.add_argument("--output_dir", default=None,
                   help="Directory to write eval_result.json and eval_verbose.csv. Defaults to model_path.")

    # Optional: dataset config
    p.add_argument("--eval_split", default="test")
    p.add_argument("--image_column", default="pixel_values")
    p.add_argument("--reference_column", default="text",
                   help="Raw reference text column (canonical, not tokenizer round-tripped).")
    p.add_argument("--text_id_column", default="text_id")
    p.add_argument("--chunk_id_column", default="chunk_id")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="If set, evaluate only the first N samples (smoke test).")

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help=">1.0 penalizes already-generated tokens (helps with degenerate repeat).")

    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--print_first_n", type=int, default=5,
                   help="Print first N predictions to stdout for sanity-checking.")

    return p.parse_args()


# -------------------------------------------------------------------
# Dataset loading (mirrors finetuning's prepare_single_dataset)
# -------------------------------------------------------------------

def load_eval_dataset(
    dataset_path: str,
    split: str,
    cache_dir: Optional[str],
    image_column: str,
    reference_column: str,
    text_id_column: str,
    chunk_id_column: str,
):
    """
    Load eval split using the same pattern as finetuning:
        ds = load_dataset(name, split=..., cache_dir=...)
        ds = ds.select_columns([...])
        ds = ds.cast(common_features)  # for pixel_values

    The reference text column is REQUIRED here (eval-specific; finetuning doesn't need it).
    text_id / chunk_id are pulled in opportunistically for traceability.
    """
    logger.info(f"Loading dataset: {dataset_path} (split={split})")
    ds = load_dataset(dataset_path, split=split, cache_dir=cache_dir)

    available = ds.column_names

    # Required columns for evaluation
    required = [image_column, reference_column]
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(
            f"Dataset {dataset_path} (split={split}) missing required columns: {missing}. "
            f"Available: {available}"
        )

    # Opportunistic columns (kept if present, ignored otherwise)
    optional_present = [c for c in (text_id_column, chunk_id_column) if c in available]
    columns_to_keep = required + optional_present
    ds = ds.select_columns(columns_to_keep)

    # Cast pixel_values to the same uint8-of-uint8 schema finetuning uses.
    # We can't cast `text` (it's already string) — only pixel_values needs schema enforcement
    # so prepare_pixels receives the exact same array layout finetuning's lazy_process does.
    cast_features = {col: ds.features[col] for col in ds.column_names}
    cast_features[image_column] = Sequence(Sequence(Value("uint8")))
    ds = ds.cast(Features(cast_features))

    logger.info(f"Loaded {len(ds):,} samples. Columns kept: {ds.column_names}")
    return ds


# -------------------------------------------------------------------
# Pixel preparation (mirrors finetuning's lazy_process EXACTLY)
# -------------------------------------------------------------------

_image_transform = transforms.Compose([transforms.ToTensor()])


def prepare_pixels(raw_img, device, dtype) -> Optional[torch.Tensor]:
    """
    Convert a raw uint8 pixel array (height=16, width=16*n) to a model-ready tensor.
    Mirrors `make_finetune_transform` in the finetuning code: no Resize, just ToTensor + RGB.
    Returns shape [1, 3, 16, width].
    """
    if raw_img is None:
        return None
    try:
        np_img = np.array(raw_img, dtype=np.uint8)
        pil_image = Image.fromarray(np_img)
        # Match finetuning: only convert if NUM_CHANNELS == 3 and not already RGB
        if NUM_CHANNELS == 3 and pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        tensor = _image_transform(pil_image)  # [3, H, W], float32 in [0,1]

        # Defensive: trim to nearest patch boundary. Finetuning data is patch-aligned
        # by construction (collator asserts), but eval is single-sample and we want robustness.
        if tensor.shape[2] % PATCH_SIZE != 0:
            new_w = (tensor.shape[2] // PATCH_SIZE) * PATCH_SIZE
            tensor = tensor[:, :, :new_w]

        # Also enforce the same hard cap finetuning's collator applies (image_size[1])
        if tensor.shape[2] > MAX_IMAGE_WIDTH:
            tensor = tensor[:, :, :MAX_IMAGE_WIDTH]

        return tensor.unsqueeze(0).to(device, dtype=dtype)
    except Exception as e:
        logger.warning(f"Failed to process image: {e}")
        return None


# -------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------

@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    max_new_tokens: int,
    device,
    has_bos: bool,
    repetition_penalty: float,
):
    """
    Manual autoregressive generation with KV cache.

    Per-tokenizer start protocol:
    - has_bos=True  : seed with [[bos_token_id]], attention_mask [[1]]
    - has_bos=False : seed with [[pad_token_id]], attention_mask [[0]] (mT5)

    Returns:
      generated_token_ids (list[int])  -- excludes the seed token, includes EOS if produced
      hit_max_tokens (bool)
    """
    batch_size = 1

    # Number of pixel patches (model concatenates these before text in the sequence)
    pixel_width = pixel_values.shape[3]
    num_pixel_patches = pixel_width // PATCH_SIZE
    pixel_attention_mask = torch.ones(
        (batch_size, num_pixel_patches), dtype=torch.long, device=device
    )

    # Seed the text side
    if has_bos:
        seed_id = tokenizer.bos_token_id
        seed_mask_value = 1
    else:
        # mT5 path: use pad_token_id with attention_mask=0 to approximate "no real token"
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer has no bos_token_id AND no pad_token_id; cannot seed generation. "
                "Set tokenizer.pad_token before calling."
            )
        seed_id = tokenizer.pad_token_id
        seed_mask_value = 0

    generated_ids = torch.full(
        (batch_size, 1), seed_id, dtype=torch.long, device=device
    )
    text_attention_mask = torch.full(
        (batch_size, 1), seed_mask_value, dtype=torch.long, device=device
    )

    past_key_values = None
    eos_token_id = tokenizer.eos_token_id
    hit_max = True  # set False if we break on EOS
    new_tokens: List[int] = []

    for _ in range(max_new_tokens):
        if past_key_values is None:
            # Step 0: pixels + seed text token
            outputs = model(
                pixel_values=pixel_values,
                input_ids=generated_ids,
                pixel_attention_mask=pixel_attention_mask,
                attention_mask=text_attention_mask,  # mask only over text positions
                use_cache=True,
                return_dict=True,
            )
        else:
            # Step N+1: only the last new token; attention mask spans pixel + all prior text
            full_mask = torch.cat(
                [pixel_attention_mask, text_attention_mask], dim=1
            )
            outputs = model(
                pixel_values=None,
                input_ids=generated_ids[:, -1:],
                attention_mask=full_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        logits = outputs.logits_token[:, -1, :]  # [batch, vocab]

        # Repetition penalty (GPT-2 style: divide positive logits, multiply negative)
        if repetition_penalty != 1.0 and len(new_tokens) > 0:
            already = torch.tensor(new_tokens, dtype=torch.long, device=device).unique()
            penalize_logits = logits[0, already]
            penalize_logits = torch.where(
                penalize_logits > 0,
                penalize_logits / repetition_penalty,
                penalize_logits * repetition_penalty,
            )
            logits[0, already] = penalize_logits

        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)  # [1, 1]

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        # Newly emitted text tokens are real and should attend (mask = 1)
        text_attention_mask = torch.cat(
            [text_attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
            dim=1,
        )
        past_key_values = outputs.past_key_values

        tok_id = next_token.item()
        new_tokens.append(tok_id)

        if eos_token_id is not None and tok_id == eos_token_id:
            hit_max = False
            break

    return new_tokens, hit_max


# -------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------

def safe_cer(ref: str, pred: str) -> float:
    """Per-sample CER. jiwer requires non-empty reference; handle edge case."""
    if not ref:
        return 1.0 if pred else 0.0
    try:
        return jiwer.cer(ref, pred)
    except Exception:
        return 1.0


def safe_wer(ref: str, pred: str) -> float:
    if not ref.strip():
        return 1.0 if pred.strip() else 0.0
    try:
        return jiwer.wer(ref, pred)
    except Exception:
        return 1.0


def per_sample_chrf(ref: str, pred: str) -> float:
    """Per-sample chrF++ (sentence-level). Uses sacrebleu's sentence_chrf with word_order=2."""
    try:
        return sacrebleu.sentence_chrf(pred, [ref], word_order=2).score
    except Exception:
        return 0.0


def detect_repetition(token_ids: List[int], min_repeat_len: int = 4) -> bool:
    """Crude detector: True if some n-gram (n>=min_repeat_len) repeats >=3 times back-to-back."""
    if len(token_ids) < min_repeat_len * 3:
        return False
    for n in range(min_repeat_len, max(min_repeat_len + 1, len(token_ids) // 3 + 1)):
        for i in range(len(token_ids) - n * 3 + 1):
            chunk = token_ids[i : i + n]
            if token_ids[i + n : i + 2 * n] == chunk and token_ids[i + 2 * n : i + 3 * n] == chunk:
                return True
    return False


# -------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = args.output_dir or args.model_path
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load tokenizer (mirrors finetuning's tokenizer setup)
    logger.info(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        # match finetuning fallback
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    has_bos = tokenizer.bos_token_id is not None
    protocol = "BOS" if has_bos else "MASKED-PAD (mT5-style)"
    logger.info(f"Start-token protocol: {protocol}")
    logger.info(
        f"  bos_token_id={tokenizer.bos_token_id}, "
        f"eos_token_id={tokenizer.eos_token_id}, "
        f"pad_token_id={tokenizer.pad_token_id}"
    )

    # 2. Load model
    logger.info(f"Loading model from: {args.model_path}")
    dtype = torch.bfloat16 if (args.device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32
    model = ErniePixelForImageTransliteration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    # 3. Load dataset (single unified path; mirrors finetuning's load_dataset + select_columns + cast)
    dataset = load_eval_dataset(
        dataset_path=args.dataset_path,
        split=args.eval_split,
        cache_dir=args.cache_dir,
        image_column=args.image_column,
        reference_column=args.reference_column,
        text_id_column=args.text_id_column,
        chunk_id_column=args.chunk_id_column,
    )

    if args.max_eval_samples is not None:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    logger.info(f"Eval set size: {len(dataset):,}")

    # 4. Inference loop
    all_records: List[Dict[str, Any]] = []
    skipped = 0

    for idx, item in enumerate(tqdm(dataset, desc=f"Eval {args.lang_code}")):
        ref_text = (item.get(args.reference_column) or "").strip()
        raw_img = item[args.image_column]

        pixel_values = prepare_pixels(raw_img, args.device, dtype)
        if pixel_values is None:
            skipped += 1
            continue

        gen_ids, hit_max = generate_one(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            has_bos=has_bos,
            repetition_penalty=args.repetition_penalty,
        )

        # Strip trailing EOS from generated for decode
        gen_ids_for_decode = gen_ids
        if gen_ids_for_decode and gen_ids_for_decode[-1] == tokenizer.eos_token_id:
            gen_ids_for_decode = gen_ids_for_decode[:-1]

        pred_text = tokenizer.decode(gen_ids_for_decode, skip_special_tokens=True).strip()

        chrf_score = per_sample_chrf(ref_text, pred_text)
        cer_score = safe_cer(ref_text, pred_text)
        wer_score = safe_wer(ref_text, pred_text)
        repeats = detect_repetition(gen_ids)

        record = {
            "idx": idx,
            "text_id": item.get(args.text_id_column, ""),
            "chunk_id": item.get(args.chunk_id_column, ""),
            "text": ref_text,
            "pred": pred_text,
            "chrf++": round(chrf_score, 4),
            "cer": round(cer_score, 4),
            "wer": round(wer_score, 4),
            "pred_token_len": len(gen_ids),
            "hit_max_tokens": int(hit_max),
            "has_repetition": int(repeats),
        }
        all_records.append(record)

        if idx < args.print_first_n:
            logger.info(f"--- Sample {idx} ---")
            logger.info(f"  REF:  {ref_text[:120]}")
            logger.info(f"  PRED: {pred_text[:120]}")
            logger.info(
                f"  chrF++={chrf_score:.2f} CER={cer_score:.3f} WER={wer_score:.3f} "
                f"len={len(gen_ids)} hit_max={hit_max} repeat={repeats}"
            )

    if not all_records:
        logger.error("No samples evaluated; aborting metric computation.")
        return

    # 5. Aggregate metrics
    refs = [r["text"] for r in all_records]
    preds = [r["pred"] for r in all_records]

    # Corpus-level chrF++
    corpus_chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    # Corpus-level CER & WER (concatenate-style averaging is what jiwer does by default)
    corpus_cer = jiwer.cer(refs, preds)
    corpus_wer = jiwer.wer(refs, preds)

    # Also compute simple mean of per-sample scores (useful diagnostic)
    mean_chrf = float(np.mean([r["chrf++"] for r in all_records]))
    mean_cer = float(np.mean([r["cer"] for r in all_records]))
    mean_wer = float(np.mean([r["wer"] for r in all_records]))

    hit_max_frac = float(np.mean([r["hit_max_tokens"] for r in all_records]))
    repeat_frac = float(np.mean([r["has_repetition"] for r in all_records]))

    # 6. Write outputs into the model directory
    metrics = {
        "lang_code": args.lang_code,
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "dataset_path": args.dataset_path,
        "eval_split": args.eval_split,
        "n_samples": len(all_records),
        "n_skipped": skipped,
        "start_token_protocol": protocol,
        # === Reported metrics ===
        "chrf++": round(corpus_chrf, 4),       # PRIMARY
        "cer": round(corpus_cer * 100, 4),     # SECONDARY (as %)
        # === Computed but not reported in paper (un-normalized refs make this unreliable) ===
        "wer": round(corpus_wer * 100, 4),
        # === Per-sample-mean variants (diagnostic) ===
        "chrf++_per_sample_mean": round(mean_chrf, 4),
        "cer_per_sample_mean": round(mean_cer * 100, 4),
        "wer_per_sample_mean": round(mean_wer * 100, 4),
        # === Generation health diagnostics ===
        "hit_max_tokens_fraction": round(hit_max_frac, 4),
        "has_repetition_fraction": round(repeat_frac, 4),
        # Generation config
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
    }

    metrics_path = os.path.join(output_dir, "eval_result.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote aggregate metrics to: {metrics_path}")

    verbose_path = os.path.join(output_dir, "eval_verbose.csv")
    with open(verbose_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx", "text_id", "chunk_id", "text", "pred",
                "chrf++", "cer", "wer",
                "pred_token_len", "hit_max_tokens", "has_repetition",
            ],
        )
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)
    logger.info(f"Wrote verbose per-sample CSV to: {verbose_path}")

    # 7. Final summary print
    logger.info("=" * 60)
    logger.info(f"FINAL [{args.lang_code}] @ {args.model_path}")
    logger.info(f"  chrF++ (corpus): {corpus_chrf:.2f}    [PRIMARY]")
    logger.info(f"  CER    (corpus): {corpus_cer*100:.2f} %  [SECONDARY]")
    logger.info(f"  WER    (corpus): {corpus_wer*100:.2f} %  [internal-only]")
    logger.info(f"  hit_max_tokens={hit_max_frac:.1%} | has_repetition={repeat_frac:.1%}")
    logger.info(f"  protocol={protocol} | n={len(all_records):,} (skipped {skipped})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()