#pragma once

#include <torch/extension.h>

void printTensorShape(const torch::Tensor& tensor);