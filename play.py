import torch

x = torch.randn(2, 3)
y = torch.randn(3, 3)
print(torch.cat((x, y), 0))
