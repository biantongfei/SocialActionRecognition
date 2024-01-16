import torch
from torch import nn

m = nn.Conv1d(1058, 256, 3, stride=1, padding='same')
input = torch.randn(32, 1058, 266)
output = m(input)

print(output.shape)
output = nn.MaxPool2d((2, 1), stride=(2, 1))(output)
print(output.shape)
