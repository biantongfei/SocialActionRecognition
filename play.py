import csv
import os
import pickle
import numpy as np

performance = {}
# with open('wandb_export_2024-12-07T15_17_33.419+00_00.csv') as csvfile:
#     for row in csv.DictReader(csvfile):
#         key = 'e%s_loss%s_T%s_lr%s' % (row['epochs'], row['loss_weight'], row['T'], row['learning_rate'])
#         if key in performance.keys():
#             performance[key] += float(row['avg_f1']) / 10
#         else:
#             performance[key] = float(row['avg_f1']) / 10
# print(len(performance.keys()))
# print(sorted(performance.items(), key=lambda kv: (kv[1], kv[0])))

files = os.listdir('../HARPER_data/data/harper/train/annotations/')
# files.sort()
for file in files:
    with open('../HARPER_data/data/harper/train/annotations/' + file, 'rb') as f:
        data = pickle.load(f)
        for i in data.keys():
            seq1 = np.array(data[i]['human_joints_3d'])
            seq2 = np.array(data[i]['spot_joints_3d'])
            # center1 = np.mean(seq1, axis=0)
            # center2 = np.mean(seq2, axis=0)
            # min_distance = ((center1[0] - center2[0]) ** 2 + (center1[2] - center2[2]) ** 2) ** 0.5
            # distance = ((center1[0]) ** 2 + (center1[2]) ** 2) ** 0.5
            # min_distance = float('inf')
            # for point1 in seq1:
            #     for point2 in seq2:
            #         distance = ((point1[0] - point2[0]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5
            #         if distance < min_distance:
            #             min_distance = distance
            # print(file, i, min_distance)
            # print(data[i].keys())
            # print(data[i])
            print(len(data[i]['human_joints_2d']))
            # print(file, len(data.keys()))
            # print(data[0]['rgb_camera_name'])
        break
