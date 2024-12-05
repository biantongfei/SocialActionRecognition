import csv

performance = {}
# with open('wandb_export_2024-11-20T17_23_33.230+00_00.csv') as csvfile:
with open('wandb_export_2024-12-05T13_45_40.691+00_00.csv') as csvfile:
    for row in csv.DictReader(csvfile):
        key = 'e%s_loss%s_T%s_lr%s' % (row['epochs'], row['loss_type'], row['T'], row['learning_rate'])
        if key in performance.keys():
            performance[key] += float(row['avg_f1']) / 10
        else:
            performance[key] = float(row['avg_f1']) / 10
print(len(performance.keys()))
print(sorted(performance.items(), key=lambda kv: (kv[1], kv[0])))
