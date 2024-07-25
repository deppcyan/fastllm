#pragma once

#include <torch/extension.h>

void softmax(torch::Tensor input, torch::Tensor output);