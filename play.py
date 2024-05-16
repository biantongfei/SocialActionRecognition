import torch

a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
a = a.reshape(2, 2, 2)
print(a)
print(a[0])
