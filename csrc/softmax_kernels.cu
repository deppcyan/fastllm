// softmax_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void softmax_kernel(float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float max_val = -1e20;
        // 找到每行的最大值
        for (int i = 0; i < num_cols; ++i) {
            max_val = fmaxf(max_val, input[row * num_cols + i]);
        }

        // 计算指数和
        float sum_exp = 0.0;
        for (int i = 0; i < num_cols; ++i) {
            sum_exp += expf(input[row * num_cols + i] - max_val);
        }

        // 计算softmax输出
        for (int i = 0; i < num_cols; ++i) {
            output[row * num_cols + i] = expf(input[row * num_cols + i] - max_val) / sum_exp;
        }
    }
}

// Wrapper function
void softmax(torch::Tensor input, torch::Tensor output) {
    const auto num_rows = input.size(0);
    const auto num_cols = input.size(1);

    // 检查输入张量是否在 CUDA 上
    if (!input.is_cuda() || !output.is_cuda()) {
        throw std::invalid_argument("Input and output tensors must be on CUDA device");
    }

    // 检查输入和输出张量的大小
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2-dimensional");
    TORCH_CHECK(output.dim() == 2, "Output tensor must be 2-dimensional");
    TORCH_CHECK(input.size(0) == output.size(0) && input.size(1) == output.size(1),
                "Input and output tensors must have the same shape");

    // 设置 CUDA 网格和块尺寸
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

    // 启动 CUDA 内核
    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        num_cols
    );
}
