#pragma once

#include <torch/extension.h>

class RotaryEmbeddingLayer {
    public:
    RotaryEmbeddingLayer(int head_size, int rotary_dim, int max_position_embeddings, int base);

    torch::Tensor forward(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key);

    private:
    void init_cos_sin_cache();
    torch::Tensor compute_inv_freq();

    private:
    torch::Tensor cos_sin_cache_;
    int head_size_;
    int rotary_dim_;
    int max_position_embeddings_;
    int base_;
};