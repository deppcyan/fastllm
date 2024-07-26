#include "attention.h"
#include <torch/extension.h>

AttentionLayer::AttentionLayer(torch::Tensor q_proj_weight,
torch::Tensor q_proj_bias,
torch::Tensor k_proj_weight,
torch::Tensor k_proj_bias,
torch::Tensor v_proj_weight,
torch::Tensor v_proj_bias,
torch::Tensor o_proj_weight,
int hidden_size,
int query_heads_num,
int kv_heads_num) :
q_proj_weight_(q_proj_weight),
q_proj_bias_(q_proj_bias),
k_proj_weight_(k_proj_weight),
k_proj_bias_(k_proj_bias),
v_proj_weight_(k_proj_weight),
v_proj_bias_(k_proj_bias),
o_proj_weight_(o_proj_weight),
hidden_size_(hidden_size),
query_heads_num_(query_heads_num),
kv_heads_num_(kv_heads_num)
{

}

torch::Tensor AttentionLayer::forward(torch::Tensor& hidden, torch::Tensor& k_cache, torch::Tensor& v_cache) {
    int head_dim = hidden_size_/query_heads_num_;
    
    torch::Tensor q = torch::addmm(q_proj_bias_, hidden, q_proj_weight_.t());
    torch::Tensor k = torch::addmm(k_proj_bias_, hidden, k_proj_weight_.t());
    torch::Tensor v = torch::addmm(v_proj_bias_, hidden, v_proj_weight_.t());

    k_cache = torch::cat({k_cache, k}, 0);
    v_cache = torch::cat({v_cache, v}, 0);

    auto q_reshape = q.reshape({-1, kv_heads_num_*head_dim});
    auto k_cache_reshape = k_cache.reshape({-1, kv_heads_num_*head_dim}).transpose(0, 1);
    auto v_cache_reshape = v_cache.reshape({-1, kv_heads_num_*head_dim});
        
    torch::Tensor t = q_reshape.matmul(k_cache_reshape)/sqrt(hidden_size_);
    torch::Tensor s = torch::softmax(t, 0);

    return s.matmul(v_cache).reshape({1, -1});
}