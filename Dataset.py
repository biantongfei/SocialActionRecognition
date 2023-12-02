import os
import json
import random

import numpy
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

ori_video_dir = '../JPL/jpl_interaction_segmented_iyuv/'
sigma = '100501'
testset_rate = 0.1
coco_point_num = 133
halpe_point_num = 136


def get_data_path(is_crop, is_coco):
    if is_crop:
        if is_coco:
            data_path = '../jpl_augmented/features/crop/coco_wholebody/'
        else:
            data_path = '../jpl_augmented/features/crop/halpe136/'
    else:
        if is_coco:
            data_path = '../jpl_augmented/features/gaussian/%s/coco_wholebody/' % (sigma)
        else:
            data_path = '../jpl_augmented/features/gaussian/%s/halpe136/' % (sigma)
    return data_path


def get_tra_test_files(is_crop, is_coco):
    data_path = get_data_path(is_crop, is_coco)
    files = os.listdir(data_path)
    ori_videos_dict = {}
    for file in files:
        if '-ori_' in file:
            with open(data_path + file, 'r') as f:
                feature_json = json.load(f)
            if feature_json['action_class'] in ori_videos_dict.keys():
                ori_videos_dict[feature_json['action_class']].append(file)
            else:
                ori_videos_dict[feature_json['action_class']] = [file]
    f.close()
    test_videos_dict = {}
    for action_class in ori_videos_dict.keys():
        random.shuffle(ori_videos_dict[action_class])
        test_video_list = ori_videos_dict[action_class][:int(len(ori_videos_dict[action_class]) * testset_rate)]
        for test_video in test_video_list:
            if test_video.split('-')[0] in test_videos_dict.keys():
                test_videos_dict[test_video.split('-')[0]].append(test_video.split('_p')[-1].split('.')[0])
            else:
                test_videos_dict[test_video.split('-')[0]] = [test_video.split('_p')[-1].split('.')[0]]

    tra_files = []
    test_files = []
    for file in files:
        if file.split('-')[0] not in test_videos_dict.keys() or file.split('_p')[-1].split('.')[0] not in \
                test_videos_dict[file.split('-')[0]]:
            tra_files.append(file)
        elif '-ori_' in file:
            test_files.append(file)
    return tra_files, test_files


class AvgDataset(Dataset):
    def __init__(self, data_files, action_recognition, is_crop, is_coco, dimension):
        self.files = data_files
        self.data_path = get_data_path(is_crop=is_crop, is_coco=is_coco)
        self.action_recognition = action_recognition
        self.is_crop = is_crop
        self.is_coco = is_coco
        self.dimension = dimension

    def __getitem__(self, idx):
        with open(self.data_path + self.files[idx], 'r') as f:
            feature_json = json.load(f)
        if self.dimension == 1:
            features = numpy.zeros(
                (len(feature_json['frames']), 2 * coco_point_num + 4)) if self.is_coco else numpy.zeros(
                (len(feature_json['frames']), 2 * halpe_point_num + 4))
        else:
            features = numpy.zeros(
                (len(feature_json['frames']), coco_point_num + 2, 2)) if self.is_coco else numpy.zeros(
                (len(feature_json['frames']), halpe_point_num + 2, 2))
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        for index, frame in enumerate(feature_json['frames']):
            box_width, box_height = frame['box'][2], frame['box'][3]
            frame_feature = numpy.array(frame['keypoints'])[:, :2]
            frame_feature[:, 0] = frame_feature[:, 0] / box_width
            frame_feature[:, 1] = frame_feature[:, 1] / box_height
            frame_feature = np.append(frame_feature, [[frame['box'][0] / frame_width, frame['box'][1] / frame_height],
                                                      [box_width / frame_width, box_height / frame_height]], axis=0)
            if self.dimension == 1:
                frame_feature = frame_feature.reshape(1, frame_feature.size)[0]
            features[index] = frame_feature
        if self.action_recognition:
            label = np.float32(feature_json['action_class'])
        else:
            if np.float32(feature_json['action_class']) == 7:
                label = 1
            elif np.float32(feature_json['action_class']) == 8:
                label = 2
            else:
                label = 0
        return features.mean(axis=0), np.float32(feature_json['action_class'])

        def __len__(self):
            return len(self.files)


if __name__ == '__main__':
    tra_files, test_files = get_tra_test_files(is_crop=True, is_coco=True)
    dataset = AvgDataset(data_files=tra_files, action_recognition=True, is_crop=True, is_coco=True, dimension=2)
    features, labels = dataset.__getitem__(0)
    print(features.shape, labels)
