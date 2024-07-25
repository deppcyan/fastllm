#include "ops.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::module ops = m.def_submodule("ops", "fastllm operators");

    ops.def("softmax", &softmax, "CUDA Softmax function");
}