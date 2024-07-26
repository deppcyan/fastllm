#pragma once

#include <torch/extension.h>

class MLPLayer {
    public:
    MLPLayer(torch::Tensor& down_proj, torch::Tensor& up_proj);
    torch::Tensor forward(torch::Tensor x);

    private:
    torch::Tensor silu(torch::Tensor x);

    private:
    torch::Tensor down_proj_;
    torch::Tensor up_proj_;
};