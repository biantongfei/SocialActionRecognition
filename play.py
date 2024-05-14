import torch

a = torch.zeros((2, 3))

a = a + torch.full((2, 3), 1)
print(a)
