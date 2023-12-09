import os

from Datasets.AvgDataset import get_data_path, get_body_part

import json
import random

import numpy as np
from torch.utils.data import Dataset

testset_rate = 0.1
coco_point_num = 133
halpe_point_num = 136


class PerFrameDataset(Dataset):
    def __init__(self, data_files, action_recognition, is_crop, is_coco, sigma, dimension, body_part):
        super(PerFrameDataset, self).__init__()
        self.files = data_files
        self.data_path = get_data_path(is_crop=is_crop, is_coco=is_coco, sigma=sigma)
        self.action_recognition = action_recognition
        self.is_crop = is_crop
        self.is_coco = is_coco
        self.dimension = dimension
        self.body_part = body_part
        self.frame_list = self.get_all_frames_id()

    def __getitem__(self, idx):
        frame = self.frame_list[idx]
        with open(self.data_path + frame.split('~')[0], 'r') as f:
            feature_json = json.load(f)

        index = int(frame.split('~')[1])
        feature = np.array(feature_json['frames'][index]['keypoints'])
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        box_x, box_y, box_width, box_height = frame['box'][0], frame['box'][1], frame['box'][2], frame['box'][3]
        feature[:, 0] = (feature[:, 0] - box_x) / box_width
        feature[:, 1] = (feature[:, 1] - box_y) / box_height
        feature = feature[:, :2]
        feature = np.append(feature, [
            [(box_x - (frame_width / 2)) / frame_width, (box_y - (frame_height / 2)) / frame_height],
            [box_width / frame_width, box_height / frame_height]], axis=0)
        if self.dimension == 1:
            feature = feature.reshape(1, feature.size)[0]
        else:
            feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        if self.action_recognition:
            label = feature_json['action_class']
        else:
            if feature_json['action_class'] == 7:
                label = 1
            elif feature_json['action_class'] == 8:
                label = 2
            else:
                label = 0
        feature = get_body_part(feature, self.is_coco, self.body_part)
        return feature, label

    def __len__(self):
        return len(self.frame_list)

    def get_all_frames_id(self):
        frame_list = []
        for file in self.files:
            with open(self.data_path + file, 'r') as f:
                feature_json = json.load(f)
            for index, frame in enumerate(feature_json['frames']):
                frame_list.append('%s~%d' % (file, index))
        random.shuffle(frame_list)
        return frame_list


if __name__ == '__main__':
    files = os.listdir('../jpl_augmented/features/crop/coco_wholebody/')
    dataset = PerFrameDataset(files, 2, True, True, '', 2)
    print(dataset.__len__())
