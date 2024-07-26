from safetensors.torch import safe_open
import fastllm
import torch

# Open and load tensors from the file
with safe_open("/data/haiyang.wang/workspace/Qwen2-0.5B-Instruct/model.safetensors", framework="pt") as f:
    #print(f.keys())
    #loaded_tensors = {key: f.get_tensor(key) for key in f.keys()}

    q_weight = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    q_bias = f.get_tensor("model.layers.0.self_attn.q_proj.bias")
    k_weight = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
    k_bias = f.get_tensor("model.layers.0.self_attn.k_proj.bias")
    v_weight = f.get_tensor("model.layers.0.self_attn.v_proj.weight")
    v_bias = f.get_tensor("model.layers.0.self_attn.v_proj.bias")
    o_weight = f.get_tensor("model.layers.0.self_attn.o_proj.weight")

    attn = fastllm.layers.AttentionLayer(
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        896,
        14,
        2
    )

    x = torch.randn(1, 896, dtype=torch.float32)
    x = x.to(dtype=torch.bfloat16)

    k_cache = torch.randn(2, 128, dtype=torch.float32)
    k_cache = k_cache.to(dtype=torch.bfloat16)

    v_cache = torch.randn(2, 128, dtype=torch.float32)
    v_cache = v_cache.to(dtype=torch.bfloat16)

    y = attn.forward(x, k_cache, v_cache)

    print(y.shape)
    #print(y)