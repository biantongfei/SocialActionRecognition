import csv

performance = {}
# with open('wandb_export_2024-11-20T17_23_33.230+00_00.csv') as csvfile:
with open('wandb_export_2024-11-20T17_02_50.686+00_00.csv') as csvfile:
    for row in csv.DictReader(csvfile):
        key = 'e%s_fc1%s_fc2%s_k%s_t%s' % (
            row['epochs'], row['fc_hidden1'], row['fc_hidden2'], row['keypoint_hidden_dim'], row['time_hidden_dim'])
        if key in performance.keys():
            performance[key] += float(row['avg_f1']) / 4
        else:
            performance[key] = float(row['avg_f1']) / 4
print(len(performance.keys()))
print(sorted(performance.items(), key=lambda kv: (kv[1], kv[0])))
