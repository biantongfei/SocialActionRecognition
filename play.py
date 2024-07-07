import torch

# 示例张量
tensor1 = torch.tensor([1, 2, 3, 4, 5])
tensor2 = torch.tensor([1, 3, 3, 0, 5])

# 找到值不同的索引
different_indices = torch.nonzero(torch.ne(tensor1, tensor2)).squeeze()

print(different_indices)
print(different_indices.shape)