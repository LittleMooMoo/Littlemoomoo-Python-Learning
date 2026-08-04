import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x = torch.tensor(7)

# print(x.item())

# print(x)

# vector = torch.tensor([3, 2])
# print(vector.ndim)

# print(vector.shape)

# print(vector)

# matrix = torch.tensor([[7, 8], [4, 5]])

# print(matrix.ndim)
# print(matrix.shape)

# tensor = torch.tensor([[[2, 1, 2], [2, 3, 2], [3, 2, 7]],
#                       [[3, 4, 42], [3, 221, 1], [3, 21, 2]]])

# print(tensor)
# print(tensor.shape)

# random = torch.rand(3, 2, 2)
# print(random)

# zeros = torch.zeros(size=(3, 2, 2))
# print(zeros)

# x = torch.arange(0, 10)
# x = torch.add(x, 10)
# y = torch.arange(0, 10)
# z = torch.mul(x, y)
# w = torch.dot(z, y)
# # print(x)

# # print(z)

# print(w)

# y = torch.rand(2, 2)
# x = torch.rand(2, 2)
# z = torch.matmul(x, y)
# print(z)

# tensor_1 = torch.tensor([[2, 3], [3, 2], [5, 5]])
# tensor_2 = torch.tensor([[2, 2, 3], [5, 5, 7]])
# tensor_3 = torch.matmul(tensor_1, tensor_2)
# print(tensor_3)

# tensor_4 = torch.tensor([[3, 2], [3, 21], [9, 7]])
# tensor_4_transposed = tensor_4.transpose(1, 0)
# print(tensor_4_transposed)
# print(torch.matmul(tensor_1, tensor_4_transposed))

# goo = torch.mean(moo)
# goo1 = torch.max(moo)
# goo2 = torch.min(moo)
# print(goo)
# print(goo1)
# print(goo2)

# meow = torch.argmin(moo)
# meow1 = torch.argmax(moo)
# print(meow1)

moo = torch.tensor([[1], [3], [32]], dtype=torch.float32)

r = torch.tensor([[2, 2, 1], [2, 1, 1], [5, 5, 8]])

# moo_reshpaed = moo.reshape(9, 1)
# moo_reviewed = moo.view(1, 9)

# z = r.view(1, 9)
# z[:, 4] = 5

# print(z)
# print(r)
# z = torch.stack([moo, r], dim=2)
# print(z[0, 0, 0])  # first page, first row, first column

y = torch.squeeze(moo)

z = y.unsqueeze(dim=1)

w = z.permute(1, 0)

print(w)
print(w.shape)
