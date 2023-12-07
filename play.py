import numpy as np

list2 = [1, 3, 2, 4, 2, 6, 2, 3, 3]
counts2 = np.bincount(list2)
Mode1 = np.argmax(np.bincount(list2))
print(Mode1)
