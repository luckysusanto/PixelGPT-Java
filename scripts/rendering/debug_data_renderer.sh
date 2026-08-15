#!/bin/bash

# 1. Load your .env
set -a
[ -f .env ] && source .env
set +a

# 2. Setup Python Path
export PYTHONPATH="../../src:$PYTHONPATH"
export HF_HOME="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated"

# 3. Define the list of languages to process
languages=("javanese" "sundanese" "pbali" "balinese" "lampung")
# languages="pbali"

# 4. Loop through each language
for lang in "${languages[@]}"
do
    echo "----------------------------------------------------"
    echo "STARTING PROCESSING FOR: $lang"
    echo "----------------------------------------------------"

    # Map the language to its specific tokenizer AND renderer path
    case $lang in
        "javanese")
            tokenizer="izzako/javanese-llama-tokenizer"
            renderer="../../renderers/m4_renderer"
            ;;
        "sundanese")
            tokenizer="izzako/sunda-llama-tokenizer"
            renderer="../../renderers/m4_renderer"
            ;;
        "balinese")
            tokenizer="izzako/javanese-llama-tokenizer"
            renderer="../../renderers/m4_renderer"
            ;;
        "pbali")
            tokenizer="izzako/balinese-llama-tokenizer"
            renderer="../../renderers/m4_renderer"
            ;;
        "lampung")
            tokenizer="izzako/sunda-llama-tokenizer"
            renderer="../../renderers/lampung_renderer"
            ;;
        *)
            echo "Unknown language: $lang. Skipping..."
            continue
            ;;
    esac

    echo "Using Tokenizer: $tokenizer"
    echo "Using Renderer:  $renderer"

    # 5. Run the script
    # We now use the ${renderer} variable for the config path
    python data_renderer.py \
            --renderer_config_path "${renderer}" \
            --lang "${lang}" \
            --tokenizer_path "${tokenizer}" \
            2>&1 | tee "render_${lang}.log"

    echo "FINISHED: $lang"
    echo "----------------------------------------------------"
done

echo "ALL LANGUAGES PROCESSED!"