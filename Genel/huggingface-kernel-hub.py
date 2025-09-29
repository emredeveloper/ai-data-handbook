import torch

from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
activation = get_kernel("kernels-community/activation")

# Random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="auto")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)