import torch
from typing import List, Union

base = 10000
head_size = 64
rotary_dim = 64
max_position_embeddings = 2048

def _compute_inv_freq(base: Union[int, float]) -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
    # However, we use `torch.arange(..., dtype=torch.float)` instead to
    # avoid numerical issues with large base values (e.g., 10000000).
    # This may cause a slight numerical difference between the HF
    # implementation and ours.
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    inv_freq = 1.0 / (base**(torch.arange(
        0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    return inv_freq

def _compute_cos_sin_cache() -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = _compute_inv_freq(base)
    t = torch.arange(max_position_embeddings, dtype=torch.float)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

cache = _compute_cos_sin_cache()
