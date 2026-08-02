import json

import torch
from safetensors import safe_open

index = json.load(open("models/qwen3-4b-base/model.safetensors.index.json"))
file = index["weight_map"]["model.embed_tokens.weight"]

with safe_open(f"models/qwen3-4b-base/{file}", framework="pt") as f:
    w = f.get_tensor("model.embed_tokens.weight")
    print(w.shape)
    for id in [9707, 1879]:
        print(id, w[id][:4].to(torch.float32).tolist())
