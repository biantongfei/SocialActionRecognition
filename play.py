import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

train_x = [torch.tensor([[2, 2]]),
           torch.tensor([[1, 1], [2, 2]])]
x = rnn_utils.pad_sequence(train_x, batch_first=True)
print(x.shape)
print(x)
print(type('none'))
print(type('none') is )