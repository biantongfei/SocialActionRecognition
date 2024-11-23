import csv

performance = {}
# with open('wandb_export_2024-11-20T17_23_33.230+00_00.csv') as csvfile:
with open('wandb_export_2024-11-23T14_00_07.492+00_00.csv') as csvfile:
    for row in csv.DictReader(csvfile):
        key = 'e%s_loss%s_k%s_t_%s' % (
            row['epochs'], row['loss_type'], row['keypoints_hidden_dim'], row['time_hidden_dim'])
        if key in performance.keys():
            performance[key] += float(row['avg_f1']) / 8
        else:
            performance[key] = float(row['avg_f1']) / 8
print(len(performance.keys()))
print(sorted(performance.items(), key=lambda kv: (kv[1], kv[0])))

