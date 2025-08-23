
import torch
tensor = torch.tensor([1, 2, 3])

tensor_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
tensor_B = torch.tensor([[7, 10], [8, 11], [9, 12]])

# print(torch.matmul(tensor_A, tensor_B))

# print(tensor_A)
# print(tensor_B)

# print(torch.matmul(tensor_A, tensor_B.T))
tensor_C = torch.tensor([[1, 2], [3, 4]])
tensor_D = torch.tensor([[2, 3], [4, 5]])

# print(torch.mm(tensor_C, tensor_D))

M_1 = torch.tensor([[3, 1, 4]])
M_2 = torch.tensor([[4, 3], [2, 5], [6, 8]])

linear = torch.nn.Linear(in_features=2, out_features=6)
x = tensor_A
output = linear(x)

print(f"Input shape: {x.shape}\n")
print(f"Output: \n{output}\n\nOutput shape: {output.shape}")
