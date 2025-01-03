import csv
import os
import pickle
import numpy as np

performance = {}
with open('wandb_export_2025-01-03T19_27_18.440+00_00.csv') as csvfile:
    for row in csv.DictReader(csvfile):
        key = 'e%s_time%s_w%s' % (
            row['epochs'], row['time_hidden_dim'], row['loss_weight'])
        if key in performance.keys():
            performance[key] += float(row['avg_f1']) / 10
        else:
            performance[key] = float(row['avg_f1']) / 10
print(len(performance.keys()))
print(sorted(performance.items(), key=lambda kv: (kv[1], kv[0])))
# data_path = ['../HARPER_data/data/harper/train/annotations/', '../HARPER_data/data/harper/test/annotations/']
# for d_p in data_path:
#     files = os.listdir(d_p)
#     files.sort()
#     for file in files:
#         with open(d_p + file, 'rb') as f:
#             contact_frame = 0
#             contact_time = 0
#             data = pickle.load(f)
#             # print(data[0]['action'])
#             if data[0]['action'] in ['act1_0', 'act1_45', 'act1_90', 'act1_180', 'act2', 'act3', 'act4', 'act9',
#                                      'act11', 'act12']:
#                 for i in data.keys():
#                     seq1 = np.array(data[i]['human_joints_3d'])
#                     seq2 = np.array(data[i]['spot_joints_3d'])
#                     # center1 = np.mean(seq1, axis=0)
#                     # center2 = np.mean(seq2, axis=0)
#                     # min_distance = ((center1[0] - center2[0]) ** 2 + (center1[2] - center2[2]) ** 2) ** 0.5
#                     min_distance = float('inf')
#                     for point1 in seq1:
#                         for point2 in seq2:
#                             distance = ((point1[0] - point2[0]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5
#                             if distance < min_distance:
#                                 min_distance = distance
#                     if min_distance < 0.1:
#                         contact_frame += 1
#                     elif min_distance > 0.1:
#                         contact_time += 1 if contact_frame >= 5 else 0
#                         contact_frame = 0
#                 if contact_time > 0:
#                     print(file, contact_time)
#
#                 # print(data[i].keys())
#                 # print(data[i])
#                 # print(len(data[i]['human_joints_2d']))
#                 # print(file, len(data.keys()))
#                 # print(data[0]['rgb_camera_name'])
#                 # break
