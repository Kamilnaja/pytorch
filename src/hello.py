import torch
torch.backends.mps.is_available()
# print("helo world")
# print(torch.__version__)
scalar = torch.tensor(7)
print(scalar.item())    

vector = torch.tensor([7, 7])
# print(vector.ndim)
# print(vector.size)

matrix = torch.tensor([[7, 7], [8, 8]])
# print(matrix.ndim)
# print(matrix.shape)

random_tensor = torch.rand(size=(3, 4))
random_7_tensor = torch.rand(size=(7, 7))
# print(random_tensor)
print(torch.max(random_7_tensor))
random_image_size = torch.rand(size=(224, 224, 3))
# print(random_image_size.shape, random_image_size.ndim)

zeros = torch.zeros(size=(2, 5));
# print(zeros)

ones = torch.ones(size=(3, 2))
# print(ones)

zero_to_ten = torch.arange(start=0, end=30, step=2)
# print(zero_to_ten)

ten_zeros = torch.zeros_like(input=ones)
# print(ten_zeros)

float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=None, device=None, requires_grad=False)
# print(float_32_tensor)

some_tensor = torch.rand(3, 4)
# print(f"Shape of tensor:, {some_tensor.shape}")
# print(f"Datatype of tensor: {some_tensor.dtype}")
# print(f"device: {some_tensor.device}")

tensor = torch.tensor([1,2,3])
n_tensor = tensor + 10
n_tensor = n_tensor * 10
# print(n_tensor)
mini_tensor = torch.tensor([2, 4, 7])
mult = torch.multiply(mini_tensor, 10)
# print(mult)
# print(mini_tensor * mini_tensor)

tens_1 = torch.tensor([3,2])
# print(torch.mm(torch.tensor([1, 2]), torch.tensor([2, 1])))
# print(torch.mm(tens_1, tens_1.T))

first_tens = torch.tensor([1,3,4, 9]);
sec_tens = torch.tensor([2, 3, 3, 9])
# print(first_tens * sec_tens)

# print(torch.tensor([4,2]) @ torch.tensor([2, 4]))
ms = torch.manual_seed(42)
# print(ms)
linear = torch.nn.Linear(in_features = 2, out_features=6)
# print(linear)
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

# print(len(X), len(X_train), len(y_train), len(X_test), len(y_test))

import matplotlib.pyplot as plt

def plot_predictions(
        train_data=X_train, 
        train_labels=y_train, 
        test_data=X_test, 
        test_labels=y_test, 
        predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=10, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()


