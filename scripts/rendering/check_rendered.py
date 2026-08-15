import os
import numpy as np
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

# 1. Load your HF Token if the repository is private
load_dotenv()
token = os.environ.get('HF_TOKEN')

# 2. Load the dataset (using the path you provided)
dataset_name = "Exqrch/Rebuttal-javanese-pixelgpt-debug"
print(f"Loading dataset: {dataset_name}...")
ds = load_dataset(dataset_name, split="train", token=token, download_mode="force_redownload")

# 3. Process the first 3 entries
for i in range(min(3, len(ds))):
    sample = ds[i]
    
    # The dataset stores pixel_values as a list of lists.
    # We convert it back to a numpy array (uint8).
    pixels = np.array(sample['pixel_values'], dtype=np.uint8)
    
    # Verification: The height should be 16, width is variable.
    print(f"Sample {i+1}:")
    print(f"  - Text:  {sample['text']}")
    print(f"  - Shape: {pixels.shape} (Height x Width)")
    
    # 4. Convert numpy array to PIL Image
    # Since it's grayscale (16, width), PIL handles it directly as 'L' (Luminance)
    img = Image.fromarray(pixels)
    
    # 5. Save the image
    filename = f"jv-debug-{i+1}.png"
    img.save(filename)
    print(f"  - Saved to: {filename}\n")

print("Done! Check your folder for the .png files.")