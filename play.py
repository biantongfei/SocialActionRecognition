import torch
from torch import nn
import numpy as np
import math

m1 = nn.Conv1d(130, 128, 7, stride=3, padding=3)
m2 = nn.Conv1d(128, 64, 5, stride=2, padding=2)
m3 = nn.Conv1d(64, 32, 5, stride=2, padding=2)
m4 = nn.Conv1d(32, 16, 3, stride=1, padding=1)
m5 = nn.Conv1d(16, 8, 3, stride=1, padding=1)
input = torch.randn(32, 30, 130)
input = torch.transpose(input, 1, 2)
output = m1(input)
print(output.shape)
output = m2(output)
print(output.shape)
output = m3(output)
print(output.shape)
output = m4(output)
print(output.shape)
output = m5(output)
print(output.shape)
print(int((math.ceil(int((math.ceil(int((math.ceil(3 / 3) + 2) / 2) / 3) + 2) / 2) / 3) + 2) / 2))