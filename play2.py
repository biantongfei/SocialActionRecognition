import numpy as np

features = np.array([1.2, 3.4, 5.6, 7.8])  # 特征序列
binary_sequence = np.array([1, 0, 1, 0])    # 二元序列

# 将二元序列为 0 的位置的特征置为 0
filtered_features = features * binary_sequence
print(filtered_features)