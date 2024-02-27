import torch
from torch import nn
# import numpy as np
import math

m1 = nn.Conv1d(266, 64, 7, stride=3, padding=3)
m2 = nn.Conv1d(64, 32, 7, stride=3, padding=3)
m3 = nn.Conv1d(32, 16, 7, stride=3, padding=3)
input = torch.randn(32, 3, 266)
input = torch.transpose(input, 1, 2)
output = m1(input)
print(output.shape)
output = nn.MaxPool1d(2, stride=2, padding=1)(output)
print(output.shape)
output = m2(output)
print(output.shape)
output = nn.MaxPool1d(2, stride=2, padding=1)(output)
print(output.shape)
output = m3(output)
print(output.shape)
output = nn.MaxPool1d(2, stride=2, padding=1)(output)
print(output.shape)
print(int((math.ceil(int((math.ceil(int((math.ceil(3 / 3) + 2) / 2) / 3) + 2) / 2) / 3) + 2) / 2))
180
360
1080

1058
529
265
60
30
15
8
3
