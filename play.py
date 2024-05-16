import torch

a = torch.zeros((2, 2))
a = a + torch.full((2, 2), fill_value=1)
print(a)
