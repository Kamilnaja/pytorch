import torch
random_tensor = torch.rand([7, 7])
random_2 = torch.rand([1, 7]).T

rand_mm = torch.mm(random_tensor, random_2)
print(rand_mm)

torch.manual_seed(seed=42)  # type: ignore
