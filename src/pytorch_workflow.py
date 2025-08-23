import torch

from linear_regression_model import LinearRegressionModel
from model import X_test

torch.manual_seed(42)
model_0 = LinearRegressionModel()

model_0_params = list(model_0.parameters())
print(model_0_params)
print(model_0.state_dict())

with torch.inference_mode():
    y_preds = model_0(X_test)