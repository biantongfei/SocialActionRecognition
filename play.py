import os
import shutil

train = ['Lingming', 'Ilaria', 'Birgi', 'Aggarwal']
val = ['Song', 'Lu']
test = ['Wan', 'Wang']

data_path = '../JPL_Augmented_Posefeatures/mixed/coco_wholebody/'
files = os.listdir(data_path)
for file in files:
    print(file)
    try:
        if file.split('test')[0] in train:
            shutil.copyfile(data_path + file, data_path + 'train/' + file)
        else:
            if '-ori_' in file:
                if file.split('test')[0] in val:
                    shutil.copyfile(data_path + file, data_path + 'validation/' + file)
                elif file.split('test')[0] in test:
                    shutil.copyfile(data_path + file, data_path + 'test/' + file)
    except ValueError:
        continue
