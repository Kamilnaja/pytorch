# NumPy array to tensor
import torch
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # type: ignore
print(array, tensor)

array = array + 1
print(array, tensor)
