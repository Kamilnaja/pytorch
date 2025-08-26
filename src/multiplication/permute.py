import torch

x_original = torch.rand(size=(244, 244, 3))
x_permuted = x_original.permute(2, 0, 1)
# print(f"Previous shape: {x_original.shape}")
# print(f"New shape: {x_permuted.shape}")

x = torch.arange(1, 10).reshape(1, 3, 3)
print(x[:, 0])
