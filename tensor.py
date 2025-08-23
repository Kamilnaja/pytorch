import torch
import numpy as np


t1 = torch.tensor([[1., -1.], [1., -1.]])
# print(t1)
t2 = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
# print(t2)
t3 = torch.zeros([2, 4], dtype=torch.int32)


print(torch.tensor(2.5000))
t4 = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)

# Scalar
scalar = torch.tensor(7)

vector = torch.tensor([[7, 7], [2, 2]])
print(vector.shape)

MATRIX = torch.tensor([[7, 8], [9, 10]])
# print(matrix)

TENSOR = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])

random_tensor = torch.rand(size=(3, 4))
# print(random_tensor, random_tensor.dtype)

zeros = torch.zeros(size=(3, 4))
# print(zeros)

# print(torch.ones(size=(4, 5)))

# print(torch.arange(1, 20, 3))
# print(torch.zeros_like(input=random_tensor))

float_32_tensor = torch.tensor(
    [3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False)
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)
