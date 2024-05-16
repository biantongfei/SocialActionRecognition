import torch

a = [[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], [[[5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8]]]]
a = torch.Tensor(a)
print(a)
a = a.reshape(-1, 3)
print(a)
a = a.reshape(2, 2, 2, 3)
print(a)
