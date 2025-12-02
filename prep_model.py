import os
import argparse
import logging
from transformers import AutoTokenizer
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Loads a pretrained model, adapts it to a new tokenizer by resizing its
    token embedding and LM head layers (cold start), and saves the result.
    """
    parser = argparse.ArgumentParser(
        description="Adapt a pretrained Hugging Face model to a new tokenizer."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path or Hub name of the base model to load (e.g., 'ernie-research/DualGPT')."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path or Hub name of the new tokenizer to adapt the model to."
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        required=True,
        help="Path to save the new, adapted model."
    )
    args = parser.parse_args()

    # --- 1. Load the Original Model and New Tokenizer ---
    logging.info(f"Loading base model from: {args.base_model_path}")
    model = ErniePixelForCausalLM.from_pretrained(args.base_model_path)

    logging.info(f"Loading new tokenizer from: {args.tokenizer_path}")
    new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    original_vocab_size = model.config.vocab_size
    new_vocab_size = len(new_tokenizer)

    logging.info(f"Original model vocabulary size: {original_vocab_size}")
    logging.info(f"New tokenizer vocabulary size: {new_vocab_size}")

    # --- 2. Perform the "Cold Start" Resizing ---
    if original_vocab_size != new_vocab_size:
        logging.info("Vocabulary sizes differ. Resizing model token embeddings...")
        model.resize_token_embeddings(new_vocab_size)
        
        # Important: Also update the model's configuration to reflect the change.
        if new_tokenizer.pad_token_id is not None:
            model.config.pad_token_id = new_tokenizer.pad_token_id
        
        logging.info(f"Model successfully resized to new vocabulary size: {new_vocab_size}")
    else:
        logging.info("Vocabulary sizes already match. No resizing needed.")

    # --- 3. Save the Adapted Model and Tokenizer ---
    logging.info(f"Ensuring output directory exists: {args.output_model_path}")
    os.makedirs(args.output_model_path, exist_ok=True)

    logging.info("Saving adapted model and tokenizer...")
    model.save_pretrained(args.output_model_path)
    new_tokenizer.save_pretrained(args.output_model_path)

    logging.info(f"Process complete. Adapted model saved to: {args.output_model_path}")


if __name__ == "__main__":
    main()