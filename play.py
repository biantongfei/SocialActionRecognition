import torch

a = torch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a)
print(torch.mean(a, dim=(0, 1)))
