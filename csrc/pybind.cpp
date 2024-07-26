#include "ops.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "layers/mlp.h"
#include "layers/attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::module ops = m.def_submodule("ops", "fastllm operators");

    ops.def("softmax", &softmax, "CUDA Softmax function");

    pybind11::module layers = m.def_submodule("layers", "fastllm layers");

    py::class_<MLPLayer>(layers, "MLPLayer")
        .def(py::init<torch::Tensor&, torch::Tensor&>()) // Constructor
        .def("forward", &MLPLayer::forward); // Method

    py::class_<AttentionLayer>(layers, "AttentionLayer")
       .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                            torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>(),
             pybind11::arg("q_proj_weight"),
             pybind11::arg("q_proj_bias"),
             pybind11::arg("k_proj_weight"),
             pybind11::arg("k_proj_bias"),
             pybind11::arg("v_proj_weight"),
             pybind11::arg("v_proj_bias"),
             pybind11::arg("o_proj_weight"),
             pybind11::arg("hidden_size"),
             pybind11::arg("query_heads_num"),
             pybind11::arg("kv_heads_num"))
        .def("forward", &AttentionLayer::forward);
}