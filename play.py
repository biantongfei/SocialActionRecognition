import matplotlib.pyplot as plt
import numpy as np
import cv2

heatmap_data = np.zeros((400, 400))
points = [(100, 150), (200, 250), (300, 350)]
for point in points:
    heatmap_data[point] = 1.0
points = [(150, 100), (250, 200), (350, 300)]
for point in points:
    heatmap_data[point] = 0.3
heatmap_data = cv2.GaussianBlur(heatmap_data, (21, 21), sigmaX=0, sigmaY=0)
plt.imshow(heatmap_data, cmap='jet', alpha=1)  # Use alpha to make the heatmap semi-transparent
plt.colorbar()
plt.axis('off')
plt.show()
