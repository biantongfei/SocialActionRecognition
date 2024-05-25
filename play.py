import torch
from torch_geometric.utils import add_self_loops

a = [[0, 1, 2, 3], [1, 2, 3, 4]]
a = torch.Tensor(a)
a = torch.cat([a, a.flip([0])], dim=1)
a, _ = add_self_loops(a, num_nodes=4)
print(a)
