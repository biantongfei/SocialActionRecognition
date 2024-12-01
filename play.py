import csv
from train_val import draw_confusion_martix
import random

# performance = {}
# # with open('wandb_export_2024-11-20T17_23_33.230+00_00.csv') as csvfile:
# with open('wandb_export_2024-11-24T02_11_52.183+00_00.csv') as csvfile:
#     for row in csv.DictReader(csvfile):
#         key = 'e%s_loss%s' % (row['epochs'], row['loss_type'])
#         if key in performance.keys():
#             performance[key] += float(row['avg_f1']) / 10
#         else:
#             performance[key] = float(row['avg_f1']) / 10
# print(len(performance.keys()))
# print(sorted(performance.items(), key=lambda kv: (kv[1], kv[0])))

print(random.randint(0,100))