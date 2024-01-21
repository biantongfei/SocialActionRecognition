import torch
from torch import nn
import numpy as np

m1 = nn.Conv1d(266, 64, 7, stride=3, padding=3)
m2 = nn.Conv1d(64, 32, 7, stride=3, padding=3)
m3 = nn.Conv1d(32, 16, 7, stride=3, padding=3)
input = torch.randn(32, 1044, 266)
input = torch.transpose(input, 1, 2)
output = m1(input)
print(output.shape)
output = nn.MaxPool1d(2, stride=2)(output)
print(output.shape)
output = nn.BatchNorm1d(64)(output)
print(output.shape)
output = m2(output)
print(output.shape)
output = nn.MaxPool1d(2, stride=2)(output)
print(output.shape)
output = m3(output)
print(output.shape)
output = nn.MaxPool1d(2, stride=2)(output)
print(output.shape)
print(np.append(np.zeros((2, 2)), np.zeros((2, 2)), axis=0))
