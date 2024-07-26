from safetensors.torch import safe_open
import fastllm
import torch

# Open and load tensors from the file
with safe_open("/data/haiyang.wang/workspace/Qwen2-0.5B-Instruct/model.safetensors", framework="pt") as f:
    #print(f.keys())
    #loaded_tensors = {key: f.get_tensor(key) for key in f.keys()}

    down = f.get_tensor("model.layers.1.mlp.down_proj.weight")
    up = f.get_tensor("model.layers.1.mlp.up_proj.weight")

    mlp = fastllm.layers.MLPLayer(down, up)

    x = torch.randn(1, 896, dtype=torch.float32)

    bf16_x = x.to(dtype=torch.bfloat16)

    y = mlp.forward(bf16_x)

    print(y.shape)
