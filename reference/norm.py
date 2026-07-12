import torch
from safetensors import safe_open

with safe_open("models/qwen3-4b-base/model-00003-of-00003.safetensors", framework="pt") as f:
    print(f.get_tensor("model.norm.weight")[:8].to(torch.float32).tolist())
