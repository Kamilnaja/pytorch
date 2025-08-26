import torch
import random

RANDOM_SEED = 42

torch.random.manual_seed(seed=RANDOM_SEED)  # type: ignore
random_tensor_C = torch.rand(3, 4)

torch.random.manual_seed(seed=RANDOM_SEED)  # type: ignore
random_tensor_D = torch.rand(3, 4)


print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
print(random_tensor_C == random_tensor_D)
