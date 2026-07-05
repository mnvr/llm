#!/bin/sh

set -o errexit
set -o xtrace

mkdir -p models/qwen3-4b-base
cd models/qwen3-4b-base

get () {
    curl -fLO -C - "https://huggingface.co/Qwen/Qwen3-4B-Base/resolve/main/$1"
}

get config.json
get generation_config.json
get tokenizer_config.json
get tokenizer.json
get vocab.json
get merges.txt
get model.safetensors.index.json
get model-00001-of-00003.safetensors
get model-00002-of-00003.safetensors
get model-00003-of-00003.safetensors

shasum -a 256 model-*.safetensors
