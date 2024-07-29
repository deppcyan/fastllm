#include "rotary_embedding.h"

RotaryEmbeddingLayer::RotaryEmbeddingLayer(int head_size, int rotary_dim, int max_position_embeddings, int base) :
head_size_(head_size),
rotary_dim_(rotary_dim),
max_position_embeddings_(max_position_embeddings),
base_(base)
{

}

torch::Tensor RotaryEmbeddingLayer::forward(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key) {

}

void RotaryEmbeddingLayer::init_cos_sin_cache() {
    torch::Tensor inv_freq = compute_inv_freq();
    torch::Tensor t = torch::arange(max_position_embeddings_, torch::dtype(torch::kFloat));

    torch::Tensor freqs = torch::einsum("i,j->ij", {t, inv_freq});

    torch::Tensor cos = freqs.cos();
    torch::Tensor sin = freqs.sin();

    cos_sin_cache_ = torch::cat({cos, sin}, -1);
}

torch::Tensor RotaryEmbeddingLayer::compute_inv_freq() {
    torch::Tensor arange_tensor = torch::arange(0, rotary_dim_, 2, torch::dtype(torch::kFloat));
    torch::Tensor inv_freq = 1.0 / torch::pow(base_, arange_tensor / rotary_dim_);

    return inv_freq;
}