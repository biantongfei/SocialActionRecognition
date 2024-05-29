import torch

a = [1]
a = torch.Tensor(a)
print(a[0])
if a[0] == 1:
    print(a)
