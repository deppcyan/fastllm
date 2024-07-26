import torch

# Create example tensors
a = torch.randn(14, 1, 64)  # Tensor a of shape (14, 64, 986)
b = torch.randn(2, 64, 1)   # Tensor b of shape (2, 64, 896)

# Transpose tensor b to align dimensions for multiplication
b_transposed = b.transpose(0, 1)  # Transpose to shape (2, 896, 64)

# Perform matrix multiplication
result = torch.matmul(a, b) 

print(result.shape) 
