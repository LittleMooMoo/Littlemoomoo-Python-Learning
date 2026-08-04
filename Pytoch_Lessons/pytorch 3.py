import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# A common PyTorch workflow:
# 1. Prepare data.
# 2. Build a model or load a pretrained model.
# 3. Train the model on training data.
# 4. Evaluate it on unseen test data.
# 5. Improve it through controlled experiments.
# 6. Save the useful model parameters.

# ========================================================
# check pytorch version
# print(torch.__version__)

# =======================================================
# data preparation and loading

# ======================================================
# create known parameters

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

# arange() creates shape [50]. unsqueeze(dim=1) changes it to [50, 1]:
# 50 samples with one input feature in each sample.
X = torch.arange(start, end, step).unsqueeze(dim=1)

# These are the correct labels generated from parameters that we already know.
y = weight * X + bias

# Use the first 80% of the samples for learning and the remaining 20% for
# checking how the learned model behaves on data it did not train on.
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]

# visualize data


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()

# Build a model that learns the weight and bias of a linear function.

# nn.Module is the base class for PyTorch models. Inheriting from it gives this
# class parameter tracking, device movement, train/eval modes, and model calls.
class LinearRegressionModel(nn.Module):
    def __init__(self):
        # Initialize nn.Module before assigning layers or nn.Parameter objects.
        super().__init__()
        # nn.Parameter registers these tensors as model parameters. Because
        # requires_grad=True, backward() can calculate gradients for them.
        self.weights = nn.Parameter(torch.randn(1,  # Start with a random weight.
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,  # Start with a random bias.
                                             requires_grad=True,
                                             dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the calculation performed by this model.
        return self.weights * x + self.bias


# Set the seed before constructing the model so its random initial parameters
# are reproducible each time the complete file is run.
torch.manual_seed(42)

model_0 = LinearRegressionModel()

# Calling model_0(x) invokes nn.Module.__call__(), which then invokes the
# forward(x) method above. Prefer model_0(x) over model_0.forward(x).

# The loss function reduces prediction error to one scalar value. L1Loss
# calculates the mean absolute difference between predictions and labels.
loss_fn = nn.L1Loss()

# The optimizer holds references to model_0's registered parameters.
# optimizer.step() later updates them using the gradients in parameter.grad.
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)  # Learning rate scales each update.

# One epoch is one complete pass through X_train in this full-batch example.
epochs = 2000

for epoch in range(epochs):
    # train() selects training behavior for layers such as Dropout and
    # BatchNorm. It does not enable gradients or change requires_grad.
    model_0.train()

    # Forward pass: __call__() invokes forward() and autograd records the
    # operations because gradient tracking is currently enabled.
    y_pred = model_0(X_train)

    # Compare the current predictions against the correct training labels.
    train_loss = loss_fn(y_pred, y_train)

    # PyTorch accumulates gradients by default, so clear gradients left from
    # the previous epoch before calculating the new ones.
    optimizer.zero_grad()

    # Backpropagation calculates d(loss)/d(parameter) and stores each result
    # in the corresponding parameter's .grad attribute.
    train_loss.backward()

    # SGD updates each parameter approximately as:
    # parameter = parameter - learning_rate * parameter.grad
    optimizer.step()

    # eval() selects evaluation behavior for Dropout and BatchNorm. It does
    # not disable gradient tracking. This linear model behaves identically in
    # train and eval modes because it has neither of those layer types.
    model_0.eval()

    # Test predictions do not need backward(), so inference_mode() disables
    # autograd graph construction and reduces evaluation memory use.
    with torch.inference_mode():
        test_prediction = model_0(X_test)
        test_loss = loss_fn(test_prediction, y_test)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch:3d} | "
            f"Train loss: {train_loss.item():.5f} | "
            f"Test loss: {test_loss.item():.5f}"
        )

print("Learned parameters:")
# state_dict() maps every registered parameter name to its learned tensor.
print(model_0.state_dict())

print("Correct weight:", weight)
print("Correct bias:", bias)

# Make one final prediction from the trained parameters for visualization.
model_0.eval()
with torch.inference_mode():
    y_predicted = model_0(X_test)

plot_predictions(predictions=y_predicted)
