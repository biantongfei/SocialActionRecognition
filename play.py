import torch

a = torch.randn((2, 2, 3, 2)).to(torch.int64)
print(a)
a = a.reshape((2, 2, 6))
print(a)
