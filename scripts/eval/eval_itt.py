#!/usr/bin/env python
# coding=utf-8
"""
Final Evaluation Script for Image Transliteration (Single-Item Processing).
This version uses a manual generation loop and includes debugging output.
"""

import os
import logging
import json
from dataclasses import dataclass, field

import torch
import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import jiwer
import sacrebleu

# Import the specific model class
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
IMAGE_SIZE = [16, 16384]
NUM_PATCHES = (IMAGE_SIZE[0] // 16) * (IMAGE_SIZE[1] // 16)

# --- Argument Classes ---
@dataclass
class ModelArguments:
    model_path: str = field(metadata={"help": "Path to the fine-tuned model checkpoint."})
    tokenizer_path: str = field(default="izzako/javanese-llama-tokenizer")

@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "Name or path of the evaluation dataset."})
    eval_split: str = field(default="test", metadata={"help": "The split to evaluate on."})
    image_column: str = field(default="pixel_values")
    text_column: str = field(default="token_ids")
    cache_dir: str = field(default=None)

@dataclass
class EvaluationArguments:
    output_dir: str = field(default="./evaluation_results")
    max_new_tokens: int = field(default=256)

# --- Main Evaluation Logic ---
def run_evaluation(model_args, data_args, eval_args):
    """
    Loads model and dataset, then runs the evaluation loop one item at a time.
    """
    # 1. Load Model and Tokenizer
    logger.info(f"Loading model from: {model_args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = ErniePixelForCausalLM.from_pretrained(model_args.model_path, torch_dtype=torch_dtype).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None: tokenizer.bos_token_id = tokenizer.eos_token_id

    # 2. Load Dataset
    logger.info(f"Loading dataset '{data_args.dataset_name}' split '{data_args.eval_split}'")
    dataset = load_dataset(data_args.dataset_name, cache_dir=data_args.cache_dir, download_mode='force_redownload')[data_args.eval_split]

    # 3. Initialize lists and image transform
    all_predictions = []
    all_references = []
    
    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    # 4. Single-Item Processing Loop
    logger.info("Starting evaluation (using manual generation loop)...")
    for index, item in enumerate(tqdm(dataset, desc="Evaluating Samples")):
        
        # --- Prepare a single item ---
        raw_image = item[data_args.image_column]
        reference_ids = item[data_args.text_column]

        pil_image = Image.fromarray(np.array(raw_image, dtype=np.uint8))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pixel_values_tensor = image_transform(pil_image).unsqueeze(0).to(device, dtype=torch_dtype)

        # --- MANUAL GENERATION LOOP ---
        with torch.no_grad():
            past_key_values = None
            generated_ids = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
            pixel_attention_mask = torch.ones((1, NUM_PATCHES), dtype=torch.long, device=device)

            for _ in range(eval_args.max_new_tokens):
                if past_key_values is None:
                    text_attention_mask = torch.ones_like(generated_ids)
                    model_inputs = dict(
                        pixel_values=pixel_values_tensor,
                        input_ids=generated_ids,
                        pixel_attention_mask=pixel_attention_mask,
                        attention_mask=text_attention_mask,
                        use_cache=True,
                    )
                else:
                    full_attention_mask = torch.cat(
                        [pixel_attention_mask, torch.ones_like(generated_ids)], dim=1
                    )
                    model_inputs = dict(
                        pixel_values=None,
                        input_ids=generated_ids[:, -1:],
                        attention_mask=full_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                outputs = model(**model_inputs)
                
                logits = outputs.logits_token[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                past_key_values = outputs.past_key_values

                if next_token.item() == tokenizer.eos_token_id:
                    break

        # Decode and store results
        prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reference = tokenizer.decode(reference_ids, skip_special_tokens=True)
        
        # --- ADDED FOR DEBUGGING ---
        if index >= len(dataset) - 5:
            print("\n" + "="*80)
            print(f"SAMPLE {index + 1}")
            print(f"  REFERENCE : '{reference}'")
            print(f"  GENERATED : '{prediction}'")
            print("="*80)
        # ---------------------------
        
        all_predictions.append(prediction)
        all_references.append(reference)

    # 5. Calculate and Report Metrics
    logger.info("Calculating final metrics...")
    wer = jiwer.wer(all_references, all_predictions)
    bleu = sacrebleu.corpus_bleu(all_predictions, [[ref] for ref in all_references])
    chrfpp = sacrebleu.corpus_chrf(all_predictions, [[ref] for ref in all_references], word_order=2)
    metrics = {"wer": wer * 100, "bleu": bleu.score, "chrf": chrfpp.score}

    logger.info("--- Evaluation Results ---")
    logger.info(f"Word Error Rate (WER): {metrics['wer']:.2f}%")
    logger.info(f"BLEU Score: {metrics['bleu']:.2f}")

    # 6. Save Results
    os.makedirs(eval_args.output_dir, exist_ok=True)
    results_path = os.path.join(eval_args.output_dir, "evaluation_metrics.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {results_path}")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, EvaluationArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    run_evaluation(model_args, data_args, eval_args)