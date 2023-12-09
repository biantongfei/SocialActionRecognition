import os
import json
import random

import numpy as np
from torch.utils.data import Dataset

testset_rate = 0.5
coco_point_num = 133
halpe_point_num = 136


def get_data_path(is_crop, is_coco, sigma):
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


def get_tra_test_files(is_crop, is_coco, sigma, not_add_class):
    data_path = get_data_path(is_crop, is_coco, sigma)
    files = os.listdir(data_path)
    ori_videos_dict = {}
    for file in files:
        if '-ori_' in file:
            with open(data_path + file, 'r') as f:
                feature_json = json.load(f)
            if not_add_class and feature_json['action_class'] in [7, 8]:
                continue
            elif feature_json['action_class'] in ori_videos_dict.keys():
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
        if 'json' not in file:
            continue
        elif file.split('-')[0] not in test_videos_dict.keys() or file.split('_p')[-1].split('.')[0] not in \
                test_videos_dict[file.split('-')[0]]:
            if not_add_class:
                with open(data_path + file, 'r') as f:
                    feature_json = json.load(f)
                if feature_json['action_class'] in [7, 8]:
                    continue
                f.close()
            tra_files.append(file)
        elif '-ori_' in file:
            test_files.append(file)
    return tra_files, test_files


def get_body_part(feature, is_coco, body_part):
    """
    :param body_part: 1 for only body, 2 for head and body, 3 for hands and body, 4 for head, hands and body
    :return:
    """
    coco_body_part = [23, 91]
    halpe_body_part = [26, 94]
    if body_part == 1:
        feature = feature[:coco_body_part[0], :] if is_coco else feature[:halpe_body_part[0], :]
    elif body_part == 2:
        feature = feature[:coco_body_part[1], :] if is_coco else feature[:coco_body_part[1], :]
    elif body_part == 3:
        if is_coco:
            feature = np.append(feature[:coco_body_part[0], :], feature[coco_body_part[1]:, :], axis=0)
        else:
            feature = np.append(feature[:halpe_body_part[0], :], feature[halpe_body_part[1]:, :], axis=0)
    return feature


class AvgDataset(Dataset):
    def __init__(self, data_files, action_recognition, is_crop, is_coco, sigma, dimension, body_part):
        super(AvgDataset, self).__init__()
        self.files = data_files
        self.data_path = get_data_path(is_crop=is_crop, is_coco=is_coco, sigma=sigma)
        self.action_recognition = action_recognition  # 0 for origin 7 classes; 1 for add not interested and interested; False for attitude recognition
        self.is_crop = is_crop
        self.is_coco = is_coco
        self.dimension = dimension
        self.body_part = body_part  # 1 for only body, 2 for head and body, 3 for hands and body, 4 for head, hands and body

    def __getitem__(self, idx):
        with open(self.data_path + self.files[idx], 'r') as f:
            feature_json = json.load(f)
        if self.dimension == 1:
            features = np.zeros(
                (len(feature_json['frames']), 2 * coco_point_num + 4)) if self.is_coco else np.zeros(
                (len(feature_json['frames']), 2 * halpe_point_num + 4))
        else:
            features = np.zeros(
                (1, len(feature_json['frames']), coco_point_num + 2, 2)) if self.is_coco else np.zeros(
                (1, len(feature_json['frames']), halpe_point_num + 2, 2))
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]

        for index, frame in enumerate(feature_json['frames']):
            box_x, box_y, box_width, box_height = frame['box'][0], frame['box'][1], frame['box'][2], frame['box'][3]
            frame_feature = np.array(frame['keypoints'])[:, :2]
            frame_feature[:, 0] = (frame_feature[:, 0] - box_x) / box_width
            frame_feature[:, 1] = (frame_feature[:, 1] - box_y) / box_height
            frame_feature = np.append(frame_feature, [
                [(box_x - (frame_width / 2)) / frame_width, (box_y - (frame_height / 2)) / frame_height],
                [box_width / frame_width, box_height / frame_height]], axis=0)
            if self.dimension == 1:
                frame_feature = frame_feature.reshape(1, frame_feature.size)[0]
                features[index] = frame_feature
            else:
                features[0, index] = frame_feature

        if self.action_recognition:
            label = feature_json['action_class']
        else:
            if feature_json['action_class'] == 7:
                label = 1
            elif feature_json['action_class'] == 8:
                label = 2
            else:
                label = 0
        feature = features.mean(axis=0) if self.dimension == 1 else features.mean(axis=1)
        feature = get_body_part(feature, self.is_coco, self.body_part)
        return feature, label

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    tra_files, test_files = get_tra_test_files(is_crop=True, is_coco=True)
    dataset = AvgDataset(data_files=tra_files, action_recognition=1, is_crop=True, is_coco=True, dimension=2)
    features, labels = dataset.__getitem__(0)
    print(features.shape, labels)
