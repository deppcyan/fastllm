#include "tensor_util.h"

void printTensorShape(const torch::Tensor& tensor) {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < tensor.sizes().size(); ++i) {
        std::cout << tensor.size(i);
        if (i < tensor.sizes().size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}