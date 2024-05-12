import torch

# 假设你的输入数据尺寸为 (30, 30, 26, 3)
input_data = torch.randn(30, 30, 26, 3)

# 使用 torch.permute() 函数调整输入数据的维度顺序
input_data = input_data.permute(0, 3, 1, 2, 4)

print("调整后的输入数据尺寸:", input_data.shape)