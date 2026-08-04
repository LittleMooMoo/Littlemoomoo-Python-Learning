import torch
import numpy as np

# print(x)

# y = torch.arange(1, 10).reshape(1, 3, 3)

# print(y)

# print(y[0, 0, :])

# =======================================================
# pytorch & numpy

# arrray = np.array([[1, 1, 2], [2, 2, 1]])

# x = torch.from_numpy(arrray).type(torch.float32)
# # when convertin from numpy to pytorch, pytorch uses default float64
# print(x)
# print(x.dtype)

# y = x.numpy()
# print(y)

# z = torch.rand(3, 3)
# print(z)

# ==========================================================
# reduce the randomness in a neural network

# moo = torch.rand(3, 4)
# goo = torch.rand(3, 4)

# print(moo)
# print(goo)

# print(moo == goo)
# RANDOM_SEED = 42
# torch.manual_seed(42)

# random_tensor_C = torch.rand(3, 4)
# torch.manual_seed(42)
# random_tensor_D = torch.rand(3, 4)

# print(random_tensor_C)
# print(random_tensor_D)

# ==============================================================
# running pytorch and tensors on GPU

# print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Selected device:", device)

# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

# x = torch.tensor([[[3, 2, 1], [3, 2, 1], [0, 6, 5]],
#                  [[1, 12, 3], [0, 0, 4], [5, 5, 7]]])
# x = x.to(device)
# print(x)

# ===============================================================

# tensor = torch.tensor([1, 2, 3])
# print(tensor, tensor.device)

# tensor_on_gpu = tensor.to(device)

# print(tensor_on_gpu)

# moo = tensor_on_gpu.cpu().numpy()

# print(moo, moo.device)
