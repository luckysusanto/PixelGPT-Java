#!/bin/bash

set -a
source "../../.env"
set +a

lang="javanese"
python data_renderer2.py \
        --renderer_config_path "../../renderers/m4_renderer" \
        --lang "${lang}" \
        --tokenizer_path "izzako/javanese-llama-tokenizer" \
        --debug \
        &> "render_$lang.log"