#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import logging
from huggingface_hub import HfApi, login, create_repo

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def upload_model_to_hub(
    local_model_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload model weights"
):
    """
    Uploads a local model directory to Hugging Face Hub.
    Creates the repository if it doesn't exist.
    """
    
    # 1. Validation
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"The local path '{local_model_path}' does not exist.")
    
    if not os.path.isdir(local_model_path):
        raise NotADirectoryError(f"The path '{local_model_path}' is not a directory.")

    # 2. Authentication
    # If a token is provided explicitly, use it. 
    # Otherwise, it falls back to the cached token (run `huggingface-cli login` previously).
    if token:
        login(token=token)
        logger.info("Logged in with provided token.")
    else:
        logger.info("Using cached Hugging Face credentials.")

    api = HfApi()

    # 3. Create Repository (if not exists)
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True, # This prevents error if repo already exists
            repo_type="model"
        )
        logger.info(f"Repository ready at: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create/access repository: {e}")
        return

    # 4. Upload Folder
    logger.info(f"Starting upload from '{local_model_path}' to '{repo_id}'...")
    
    try:
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            # --- MODIFIED HERE ---
            ignore_patterns=[
                ".git", 
                ".DS_Store", 
                "__pycache__", 
                "*.ipynb_checkpoints", 
                "checkpoint-*"  # <--- THIS blocks all checkpoint folders
            ], 
            # ---------------------
        )
        logger.info("Upload complete successfully!")
        print(f"\nModel is live at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload a local model directory to Hugging Face Hub.")
    
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Local path to the model directory (e.g., ./output_dir)"
    )
    parser.add_argument(
        "--repo", 
        type=str, 
        required=True, 
        help="Target Repo ID (e.g., username/my-bert-model)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None, 
        help="Hugging Face Write Token. If not provided, assumes local login."
    )
    parser.add_argument(
        "--private", 
        action="store_true", 
        help="If set, the repository will be created as Private."
    )
    parser.add_argument(
        "--message", 
        type=str, 
        default="Upload model weights", 
        help="Commit message for the upload."
    )

    args = parser.parse_args()

    upload_model_to_hub(
        local_model_path=args.path,
        repo_id=args.repo,
        token=args.token,
        private=args.private,
        commit_message=args.message
    )

if __name__ == "__main__":
    main()