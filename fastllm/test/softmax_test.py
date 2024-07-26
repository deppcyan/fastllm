import torch
import fastllm

# 创建一个随机输入张量
input_tensor = torch.randn(2, 3, device='cuda')

# 创建一个输出张量来保存结果
output_tensor = torch.empty_like(input_tensor)

# 使用CUDA实现计算Softmax
fastllm.ops.softmax(input_tensor, output_tensor)

# 验证结果
print("Input:", input_tensor)
print("CUDA Softmax Output:", output_tensor)

# 对比PyTorch的Softmax
torch_output = torch.nn.functional.softmax(input_tensor, dim=1)
print("PyTorch Softmax Output:", torch_output)

# 检查差异
assert torch.allclose(output_tensor, torch_output, atol=1e-6), "Outputs do not match!"
