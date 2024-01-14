import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

train_x = [torch.tensor([2, 2, 2, 2, 2, 2]),
           torch.tensor([1, 1, 1, 1, 1, 1, 1]),
           torch.tensor([3, 3, 3, 3, 3]),
           torch.tensor([4, 4, 4, 4]),
           torch.tensor([5, 5, 5]),
           torch.tensor([6, 6]),
           torch.tensor([7])]
x = rnn_utils.pad_sequence(train_x, batch_first=True)
print(x)
