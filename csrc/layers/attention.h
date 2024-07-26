#pragma once

#include <torch/extension.h>

class AttentionLayer {
    public:
    AttentionLayer(torch::Tensor q_proj_weight,
    torch::Tensor q_proj_bias,
    torch::Tensor k_proj_weight,
    torch::Tensor k_proj_bias,
    torch::Tensor v_proj_weight,
    torch::Tensor v_proj_bias,
    torch::Tensor o_proj_weight,
    int hidden_size,
    int query_heads_num,
    int kv_heads_num);

    torch::Tensor forward(torch::Tensor& hidden, torch::Tensor& k_cache, torch::Tensor& v_cache);

    private:
    torch::Tensor q_proj_weight_;
    torch::Tensor q_proj_bias_;
    torch::Tensor k_proj_weight_;
    torch::Tensor k_proj_bias_;
    torch::Tensor v_proj_weight_;
    torch::Tensor v_proj_bias_;
    torch::Tensor o_proj_weight_;
    int hidden_size_;
    int query_heads_num_;
    int kv_heads_num_;
};