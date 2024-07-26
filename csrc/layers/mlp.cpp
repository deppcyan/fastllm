#include "mlp.h"

MLPLayer::MLPLayer(torch::Tensor& down_proj, torch::Tensor& up_proj) :
down_proj_(down_proj), up_proj_(up_proj){

}

torch::Tensor MLPLayer::forward(torch::Tensor x) {
    torch::Tensor gate_up = silu(x.matmul(up_proj_.t()));
    return gate_up.matmul(down_proj_.t());
}

torch::Tensor MLPLayer::silu(torch::Tensor x) {
    return x * torch::sigmoid(x);
}
