#!/usr/bin/env python
# coding=utf-8
"""
Evaluation Script for Text-to-Image (Script Rendering) Model using SSIM.
This version saves the first 5 generated/reference image pairs for visual debugging.
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

from torchmetrics.image import StructuralSimilarityIndexMeasure
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForImageGeneration

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
IMAGE_SIZE = [16, 16384]

# --- Helper Function for Saving Images ---
def save_tensor_as_image(tensor: torch.Tensor, filepath: str):
    """Converts a [0, 1] range tensor to a PNG image and saves it."""
    # Squeeze the batch dimension if it exists (shape [1, C, H, W] -> [C, H, W])
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Move to CPU, convert to uint8 numpy array, and change to HWC format for PIL
    # Tensor is already normalized, so just multiply by 255
    img_np = tensor.mul(255).byte().cpu().permute(1, 2, 0).numpy()
    
    # Create and save the image
    pil_img = Image.fromarray(img_np)
    pil_img.save(filepath)
    logger.info(f"Saved debug image to {filepath}")

# --- Argument Classes ---
@dataclass
class ModelArguments:
    model_path: str = field(metadata={"help": "Path to the fine-tuned text-to-image model checkpoint."})
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
    output_dir: str = field(default="./evaluation_results_ssim")

# --- Main Evaluation Logic ---
def run_evaluation(model_args, data_args, eval_args):
    logger.info(f"Loading model from: {model_args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = ErniePixelForImageGeneration.from_pretrained(model_args.model_path, torch_dtype=torch_dtype).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)

    logger.info(f"Loading dataset '{data_args.dataset_name}' split '{data_args.eval_split}'")
    dataset = load_dataset(data_args.dataset_name, cache_dir=data_args.cache_dir)[data_args.eval_split]

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    patch_size = model.config.patch_size
    num_channels = model.config.num_channels

    logger.info("Starting evaluation (generating images and calculating SSIM)...")
    for index, item in enumerate(tqdm(dataset, desc="Evaluating Samples")):
        input_ids = torch.tensor([item[data_args.text_column]]).to(device)
        
        reference_image_raw = item[data_args.image_column]
        pil_image = Image.fromarray(np.array(reference_image_raw, dtype=np.uint8))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        reference_image_tensor = image_transform(pil_image).unsqueeze(0).to(device, dtype=torch_dtype)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            predicted_patches = outputs.logits_pixel

            h_patches = 1
            w_patches = predicted_patches.shape[1]

            unpatched = predicted_patches.reshape(
                predicted_patches.shape[0], h_patches, w_patches, patch_size, patch_size, num_channels
            )
            unpatched = torch.einsum("nhwpqc->nchpwq", unpatched)
            generated_image_tensor = unpatched.reshape(
                unpatched.shape[0], num_channels, h_patches * patch_size, w_patches * patch_size
            )

            gen_width = generated_image_tensor.shape[3]
            cropped_reference_tensor = reference_image_tensor[:, :, :, :gen_width]
            
            g_min, g_max = generated_image_tensor.min(), generated_image_tensor.max()
            if g_max > g_min:
                generated_image_tensor = (generated_image_tensor - g_min) / (g_max - g_min)
            generated_image_tensor = torch.clamp(generated_image_tensor, 0, 1)

        # --- ADDED FOR DEBUGGING ---
        # Save the first 5 pairs of images
        if index >= len(dataset) - 5:
            # Ensure the output directory exists
            os.makedirs(eval_args.output_dir, exist_ok=True)
            
            # Define filepaths
            ref_path = os.path.join(eval_args.output_dir, f"img_label_{index + 1}.png")
            gen_path = os.path.join(eval_args.output_dir, f"img_generated_{index + 1}.png")
            
            # Save the images
            save_tensor_as_image(cropped_reference_tensor, ref_path)
            save_tensor_as_image(generated_image_tensor, gen_path)
        # ---------------------------

        ssim_metric.update(generated_image_tensor.float(), cropped_reference_tensor.float())

    logger.info("Calculating final average SSIM score...")
    final_ssim = ssim_metric.compute().item()
    metrics = {"ssim": final_ssim}

    logger.info("--- Evaluation Results ---")
    logger.info(f"Average Structural Similarity Index (SSIM): {metrics['ssim']:.4f}")

    os.makedirs(eval_args.output_dir, exist_ok=True)
    results_path = os.path.join(eval_args.output_dir, "evaluation_metrics_ssim.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {results_path}")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, EvaluationArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    run_evaluation(model_args, data_args, eval_args)