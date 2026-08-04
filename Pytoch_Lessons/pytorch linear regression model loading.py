import torch
from pathlib import Path
from pytorch_linear_regression import LinearRegressionModel, MODEL_SAVE_PATH, X_test, y_predicted

# 1. To load in a saved state_dict, we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel().to(device="cuda")
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_0.eval()

with torch.inference_mode():
    loaded_model_0_prediction = loaded_model_0(X_test)

print(loaded_model_0_prediction == y_predicted)
