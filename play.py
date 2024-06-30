import torch

a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(a)
a = a.reshape(-1, 1)
print(a)
a = a.reshape(2, 3)
print(a)
