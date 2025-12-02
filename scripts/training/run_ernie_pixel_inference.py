import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from src.ernie_pixel.modeling_ernie_pixel import ErniePixelForCausalLM
from PIL import Image
import os


def save_image(pixel_values, title="Image", save_path="output.png"):
    """
    Renders and saves an image from pixel values.
    Uses aspect='equal' to keep pixels square, but calculates figure size
    to accommodate long rectangular images (strips).
    """
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.cpu().detach().numpy()

    if pixel_values.min() < 0 or pixel_values.max() > 1:
        pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())
    pixel_values = (pixel_values * 255).astype(np.uint8)

    if pixel_values.shape[0] in [1, 3]:
        img_to_show = np.transpose(pixel_values, (1, 2, 0))
    else:
        img_to_show = pixel_values

    h, w = img_to_show.shape[:2]
    aspect_ratio = w / h

    # Define a base height (in inches) that is large enough to see the title
    fig_height = 3 
    # Calculate width proportionally
    fig_width = fig_height * aspect_ratio

    # Cap width at 100 inches to prevent Matplotlib crashing on extremely long images,
    # though for 5% of 16k pixels, it will be fine (~25 inches).
    if fig_width > 100:
        fig_width = 100

    plt.figure(figsize=(fig_width, fig_height))
    
    # aspect='equal' ensures the 16x800 strip looks like a strip, not a square.
    plt.imshow(img_to_show, aspect='equal', interpolation='nearest')
    
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Image saved to {save_path}")


def inference_pipeline(model_path, tokenizer_path, output_dir, token_ids=None, pixel_values=None, flag=False):
    """
    Inference pipeline for ErniePixelForCausalLM.
    """
    print("=" * 50)

    # 1. Load Model + Tokenizer
    print(f"Loading model from: {model_path}")
    model = ErniePixelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    IMG_HEIGHT = 16

    # MODE 1 — TEXT ONLY
    if token_ids is not None and pixel_values is None:
        print("\n--- Running in Text Completion Mode ---")

        input_ids = torch.tensor([token_ids]).to(device)
        print(f"Input Prompt: '{tokenizer.decode(token_ids)}'")

        generated_ids = model.generate(
            input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        output_filepath = os.path.join(output_dir, "completion_text.txt")
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(generated_text)

        print(f"Generated text saved to {output_filepath}")
        print("=" * 50)
        return

    # MODE 2 — IMAGE ONLY
    if token_ids is None and pixel_values is not None:
        print("\n--- Running in Image Completion Mode ---")

        num_pixels = len(pixel_values)
        if num_pixels % IMG_HEIGHT != 0:
            raise ValueError(f"pixel_values length must be divisible by {IMG_HEIGHT}")

        width = num_pixels // IMG_HEIGHT

        input_image_tensor = torch.tensor(pixel_values).reshape(
            1, 1, IMG_HEIGHT, width
        ).float().repeat(1, 3, 1, 1).to(device)

        num_patches = num_pixels // (16 * 16)
        attention_mask = torch.ones(1, num_patches, dtype=torch.long, device=device)
        
        # Save prompt image normally
        save_path_prompt = os.path.join(output_dir, "prompt_image.png")
        save_image(input_image_tensor.squeeze(0), "Input Image Prompt", save_path_prompt)
        
        # --- SPECIAL FLAG HANDLING ---
        if flag:
            save_path_prompt = os.path.join(output_dir, "full_image_5percent.png")
            save_image(input_image_tensor.squeeze(0), "Full Image (5% Slice)", save_path_prompt)
            print("Flag detected: 5% slice saved. Exiting pipeline.")
            print("=" * 50)
            return

        with torch.no_grad():
            outputs = model(pixel_values=input_image_tensor, attention_mask=attention_mask)
            generated_patches = outputs.logits_pixel

        patch_size = model.model.embed_patches.patch_embeddings.patch_size[0]
        num_channels = model.config.num_channels

        h_patches = 1
        w_patches = generated_patches.shape[1]

        unpatched = generated_patches.reshape(
            generated_patches.shape[0],
            h_patches,
            w_patches,
            patch_size,
            patch_size,
            num_channels,
        )
        unpatched = torch.einsum("nhwpqc->nchpwq", unpatched)

        generated_image = unpatched.reshape(
            unpatched.shape[0],
            num_channels,
            h_patches * patch_size,
            w_patches * patch_size
        )

        save_path_output = os.path.join(output_dir, "completion_image.png")
        save_image(generated_image.squeeze(0), "Model Output Image", save_path_output)

        print("=" * 50)
        return

    # MODE 3 — IMAGE + TEXT
    if token_ids is not None and pixel_values is not None:
        print("\n--- Running in Image-Conditioned Text Generation Mode ---")

        num_pixels = len(pixel_values)
        width = num_pixels // IMG_HEIGHT

        input_image_tensor = torch.tensor(pixel_values).reshape(
            1, 1, IMG_HEIGHT, width
        ).float().repeat(1, 3, 1, 1).to(device)

        input_ids = torch.tensor([token_ids]).to(device)

        num_patches = num_pixels // (16 * 16)
        pixel_attention_mask = torch.ones(1, num_patches, device=device, dtype=torch.long)
        text_attention_mask = torch.ones_like(input_ids)

        save_image(
            input_image_tensor.squeeze(0),
            "Input Image Prompt (Full Context)",
            os.path.join(output_dir, "prompt_image_text_gen.png")
        )

        with torch.no_grad():
            past_key_values = None
            generated_ids = input_ids

            for _ in range(50):
                if past_key_values is None:
                    model_inputs = dict(
                        pixel_values=input_image_tensor,
                        input_ids=generated_ids,
                        pixel_attention_mask=pixel_attention_mask,
                        attention_mask=text_attention_mask,
                        past_key_values=None,
                        use_cache=True,
                    )
                else:
                    current_text_attention_mask = torch.ones_like(generated_ids)
                    combined_mask = torch.cat([pixel_attention_mask, current_text_attention_mask], dim=1)

                    model_inputs = dict(
                        pixel_values=None,
                        input_ids=generated_ids[:, -1:],
                        attention_mask=combined_mask,
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

        txt = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        outfile = os.path.join(output_dir, "generated_text_from_image.txt")
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(txt)

        print(f"Generated conditioned text saved to {outfile}")
        print("=" * 50)
        return


if __name__ == "__main__":

    CHKPTS = [
        i for i in range(65500, 75500, 500)
    ]
    CHKPTS = [str(c) for c in CHKPTS]
    mflag = False

    for chkpt in CHKPTS:
        MODEL_PATH = f"/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/mixedPretrain/checkpoint-{chkpt}"
        TOKENIZER_PATH = "izzako/javanese-llama-tokenizer"
        DATASET_PATH = "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-poc-2/default/0.0.0"
        OUTPUT_DIR = f"pretrain_infer_output/chpt{chkpt}-mixedPretrain"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("Loading a full sample from the dataset...")
        dataset = load_dataset(DATASET_PATH, split="train", streaming=True)
        sample_data = next(iter(dataset))

        full_token_ids = sample_data["token_ids"]
        raw_pixel_values = sample_data["pixel_values"]

        IMG_HEIGHT = 16
        FULL_IMG_WIDTH = 16384

        full_image_array = np.array(raw_pixel_values).reshape(IMG_HEIGHT, FULL_IMG_WIDTH)

        # Flatten here (correct)
        full_pixel_values = full_image_array.flatten().tolist()

        # FIX: Type comparison (string vs string)
        if chkpt == "500" and mflag == False:
            print(f"--- Processing Checkpoint 500 Special Visualization ---")
            
            # Calculate 5% width
            target_width = int(FULL_IMG_WIDTH * 0.05)
            
            # FIX: Ensure width is divisible by 16 (patch size)
            slice_width = (target_width // 16) * 16
            
            print(f"Original Width: {FULL_IMG_WIDTH}, 5% Target: {target_width}, Final Slice Width: {slice_width}")

            slice_image_array = full_image_array[:, :slice_width]
            slice_pixel_values = slice_image_array.flatten().tolist()

            # 2) IMAGE SAVE (Pass sliced pixels)
            inference_pipeline(
                model_path=MODEL_PATH,
                tokenizer_path=TOKENIZER_PATH,
                output_dir=OUTPUT_DIR,
                token_ids=None,
                pixel_values=slice_pixel_values,
                flag=True
            )
            mflag = True 
            continue

        # Token prompt
        considered_token = int(len(full_token_ids) * 0.75) + 1
        prompt_token_ids = full_token_ids[:considered_token]

        # Image prompt
        prompt_img_width = 16 * len(prompt_token_ids) // 2
        prompt_img_width += 16 - (prompt_img_width % 16)
        prompt_image_array = full_image_array[:, :prompt_img_width]
        prompt_pixel_values = prompt_image_array.flatten().tolist()

        print("=" * 50)
        print(f"--- Running Inference for Checkpoint {chkpt} ---")
        print(f"Using {len(prompt_token_ids)} tokens for prompts")

        # 1) TEXT ONLY
        inference_pipeline(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            output_dir=OUTPUT_DIR,
            token_ids=prompt_token_ids,
            pixel_values=None
        )

        # 2) IMAGE ONLY
        inference_pipeline(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            output_dir=OUTPUT_DIR,
            token_ids=None,
            pixel_values=prompt_pixel_values
        )

        # 3) TEXT + IMAGE
        inference_pipeline(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            output_dir=OUTPUT_DIR,
            token_ids=prompt_token_ids,
            pixel_values=full_pixel_values
        )