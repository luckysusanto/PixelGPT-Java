from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PixelAksara/aksarapixellm-komodo",
    local_dir="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/rebuttal_experiment_output/komodo",
    local_dir_use_symlinks=False
)