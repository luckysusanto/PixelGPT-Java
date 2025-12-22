#!/usr/bin/env python
# coding=utf-8
import os
import logging
import json
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import jiwer
import sacrebleu

from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForImageTransliteration

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
IMAGE_SIZE = [16, 16384]
NUM_PATCHES = (IMAGE_SIZE[0] // 16) * (IMAGE_SIZE[1] // 16)

@dataclass
class ModelArguments:
    model_path: str = field(metadata={"help": "Path to the fine-tuned model checkpoint."})
    tokenizer_path: str = field(default="izzako/javanese-llama-tokenizer")

@dataclass
class DataArguments:
    # Dataset A (Required)
    dataset_a_name: str = field(metadata={"help": "Path/Name of the first dataset."})
    dataset_a_lang: str = field(metadata={"help": "Language code/Identifier for Dataset A (e.g., 'javanese'). Used for output filenames."})
    
    # Dataset B (Optional)
    dataset_b_name: Optional[str] = field(default=None, metadata={"help": "Path/Name of the second dataset."})
    dataset_b_lang: Optional[str] = field(default=None, metadata={"help": "Language code for Dataset B."})
    
    # Configuration
    eval_split: str = field(default="test", metadata={"help": "The split to evaluate on."})
    image_column: str = field(default="pixel_values")
    text_column: str = field(default="grapheme_token_ids")
    cache_dir: Optional[str] = field(default=None)

@dataclass
class EvaluationArguments:
    output_dir: str = field(default="./evaluation_results")
    max_new_tokens: int = field(default=256)
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")

def evaluate_single_dataset(
    model, 
    tokenizer, 
    dataset_path, 
    lang_code, 
    data_args, 
    eval_args, 
    dtype
):
    """
    Helper function to process a single dataset and save results with the language prefix.
    """
    logger.info(f"--- Processing Dataset: {lang_code} ({dataset_path}) ---")
    
    # 1. Load Dataset
    try:
        dataset = load_dataset(dataset_path, split=data_args.eval_split, cache_dir=data_args.cache_dir)
        dataset = dataset.select_columns([data_args.image_column, data_args.text_column])
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        return

    # 2. Transform Setup
    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    all_predictions = []
    all_references = []

    logger.info(f"Starting inference on {len(dataset)} samples...")

    # 3. Inference Loop
    for item in tqdm(dataset, desc=f"Eval {lang_code}"):
        raw_image = item[data_args.image_column]
        reference_ids = item[data_args.text_column]

        # Prepare Image
        try:
            pil_image = Image.fromarray(np.array(raw_image, dtype=np.uint8))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            pixel_values = image_transform(pil_image).unsqueeze(0).to(eval_args.device, dtype=dtype)
        except Exception as e:
            logger.warning(f"Skipping corrupt image in {lang_code}: {e}")
            continue

        # Generate
        with torch.no_grad():
            past_key_values = None
            generated_ids = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=eval_args.device)
            pixel_attention_mask = torch.ones((1, NUM_PATCHES), dtype=torch.long, device=eval_args.device)

            for _ in range(eval_args.max_new_tokens):
                if past_key_values is None:
                    model_inputs = {
                        "pixel_values": pixel_values,
                        "input_ids": generated_ids,
                        "pixel_attention_mask": pixel_attention_mask,
                        "attention_mask": torch.ones_like(generated_ids),
                        "use_cache": True
                    }
                else:
                    current_mask = torch.cat([pixel_attention_mask, torch.ones((1, generated_ids.shape[1]), device=eval_args.device)], dim=1)
                    model_inputs = {
                        "pixel_values": None,
                        "input_ids": generated_ids[:, -1:],
                        "attention_mask": current_mask,
                        "past_key_values": past_key_values,
                        "use_cache": True
                    }

                outputs = model(**model_inputs)
                logits = outputs.logits_token[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                past_key_values = outputs.past_key_values

                if next_token.item() == tokenizer.eos_token_id:
                    break

        # Decode
        pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        ref_text = tokenizer.decode(reference_ids, skip_special_tokens=True).strip()
        
        all_predictions.append(pred_text)
        all_references.append(ref_text)

    # 4. Metrics
    logger.info(f"Calculating metrics for {lang_code}...")
    wer = jiwer.wer(all_references, all_predictions)
    # Wrap all_references in a list because sacrebleu expects a list of reference streams
    bleu = sacrebleu.corpus_bleu(all_predictions, [all_references])
    chrf = sacrebleu.corpus_chrf(all_predictions, [all_references], word_order=2)

    metrics = {
        "wer": wer * 100, 
        "bleu": bleu.score, 
        "chrf": chrf.score
    }

    logger.info(f"[{lang_code}] WER: {metrics['wer']:.2f} | BLEU: {metrics['bleu']:.2f} | CHRF: {metrics['chrf']:.2f}")

    # 5. Save Files
    os.makedirs(eval_args.output_dir, exist_ok=True)
    
    # Save Metrics
    metric_file = os.path.join(eval_args.output_dir, f"{lang_code}_metrics.json")
    with open(metric_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Samples
    sample_file = os.path.join(eval_args.output_dir, f"{lang_code}_samples.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Ref : {ref}\n")
            f.write(f"  Pred: {pred}\n")
            f.write("-" * 40 + "\n")
            
    logger.info(f"Saved results for {lang_code} to {eval_args.output_dir}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvaluationArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    # 1. Load Model Once
    logger.info(f"Loading model from: {model_args.model_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    model = ErniePixelForImageTransliteration.from_pretrained(
        model_args.model_path, 
        torch_dtype=dtype
    ).to(eval_args.device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.eos_token_id

    # 2. Evaluate Dataset A (Required)
    evaluate_single_dataset(
        model, tokenizer, 
        data_args.dataset_a_name, 
        data_args.dataset_a_lang, 
        data_args, eval_args, dtype
    )

    # 3. Evaluate Dataset B (Optional)
    if data_args.dataset_b_name and data_args.dataset_b_lang:
        evaluate_single_dataset(
            model, tokenizer, 
            data_args.dataset_b_name, 
            data_args.dataset_b_lang, 
            data_args, eval_args, dtype
        )
    else:
        logger.info("Dataset B not provided. Skipping.")

    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()