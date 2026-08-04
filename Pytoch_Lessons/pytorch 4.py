import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.datasets import make_circles
import pandas as pd
from sklearn.model_selection import train_test_split


# use machine learning to solve data classification function

# 0. classification type :
# 1 binary classification
# 2 multiclass classification(multiple classes, same label)
# 3 multilabel classification(multiple labels for one class)

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# print(f"first 5 samples of X: \n {X[:5]}")
# print(f"first 5 samples of y: \n {y[:5]}")
# print(y)
circles = pd.DataFrame({"X1": X[:, 0],
                       "X2": X[:, 1],
                        "label": y})

# # print(circles)
# print(X.shape)

# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.show()

# task : code a neural network so that it can classify the dots with given x
# y is the color label of the dot

X_tensor = torch.from_numpy(X).to(torch.float32).to(device="cuda")
y_tensor = torch.from_numpy(y).to(torch.float32).to(device="cuda")

# split data into training and test set:
X_train, X_test, y_train, y_test = train_test_split(X_tensor,
                                                    y_tensor,
                                                    test_size=0.2,
                                                    random_state=42)

# build a model


# class CircleModelV0(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = nn.Linear(in_features=2,
#                                  out_features=5)  # <- 5 hiden features
#         self.layer_2 = nn.Linear(in_features=5,
#                                  out_features=1)

#     def forward(self, x):
#         return self.layer_2(self.layer_1(x))

# instantiate an instance of our model class and send to target device

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device="cuda")


loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.005)

# training the model:
torch.cuda.manual_seed(42)

X_train, y_train = X_train.to(device="cuda"), y_train.to(device="cuda")
X_test, y_test = X_test.to(device="cuda"), y_test.to(device="cuda")


epoch_history = []

train_loss_history = []
test_loss_history = []

train_accuracy_history = []
test_accuracy_history = []

epochs = 8000

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        # y_preb_label = torch.round(torch.sigmoid(
        #     model_0(X_test.to(device="cuda"))[:5]))
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    epoch_history.append(epoch)
    train_loss_history.append(loss.item())
    test_loss_history.append(test_loss.item())
    train_accuracy_history.append(acc)
    test_accuracy_history.append(test_acc)

    if epoch % 100 == 0:
        print(
            f" Epoch: {epoch} | Loss: {loss:.5f}, ACC: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 5),
)

# Loss graph
axes[0].plot(
    epoch_history,
    train_loss_history,
    color="blue",
    label="Training loss",
)

axes[0].plot(
    epoch_history,
    test_loss_history,
    color="red",
    label="Testing loss",
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss During Training")
axes[0].legend()
axes[0].grid(True)

# Accuracy graph
axes[1].plot(
    epoch_history,
    train_accuracy_history,
    color="blue",
    label="Training accuracy",
)

axes[1].plot(
    epoch_history,
    test_accuracy_history,
    color="green",
    label="Testing accuracy",
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy During Training")
axes[1].set_ylim(0, 100)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
