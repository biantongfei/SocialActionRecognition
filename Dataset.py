import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import coco_body_point_num, halpe_body_point_num, head_point_num, hands_point_num, valset_rate, \
    testset_rate, coco_body_l_pair, coco_head_l_pair, coco_hand_l_pair, halpe_body_l_pair, halpe_head_l_pair, \
    halpe_hand_l_pair


def get_data_path(augment_method, is_coco):
    if augment_method == 'crop':
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/crop/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/crop/halpe136/'
    elif augment_method == 'noise':
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/gaussian/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/gaussian/halpe136/'
    elif augment_method in ['mixed', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/mixed/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/mixed/halpe136/'
    return data_path


def get_tra_test_files(augment_method, is_coco, ori_videos=False):
    if augment_method.split('+')[0] not in ['mixed', 'crop', 'noise']:
        return get_tra_test_files_generalisation(augment_method)
    data_path = get_data_path(augment_method, is_coco)
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
    validation_videos_dict = {}
    test_videos_dict = {}
    for action_class in ori_videos_dict.keys():
        random.shuffle(ori_videos_dict[action_class])
        val_video_list = ori_videos_dict[action_class][int(len(ori_videos_dict[action_class]) * (1 - valset_rate)):]
        test_video_list = ori_videos_dict[action_class][:int(len(ori_videos_dict[action_class]) * testset_rate)]
        for test_video in test_video_list:
            if test_video.split('-')[0] in test_videos_dict.keys():
                test_videos_dict[test_video.split('-')[0]].append(test_video.split('_p')[-1].split('.')[0])
            else:
                test_videos_dict[test_video.split('-')[0]] = [test_video.split('_p')[-1].split('.')[0]]
        for val_video in val_video_list:
            if val_video.split('-')[0] in validation_videos_dict.keys():
                validation_videos_dict[val_video.split('-')[0]].append(val_video.split('_p')[-1].split('.')[0])
            else:
                validation_videos_dict[val_video.split('-')[0]] = [val_video.split('_p')[-1].split('.')[0]]
    tra_files = []
    val_files = []
    test_files = []
    for file in files:
        if 'json' not in file:
            continue
        elif file.split('-')[0] not in test_videos_dict.keys() or file.split('_p')[-1].split('.')[0] not in \
                test_videos_dict[file.split('-')[0]]:
            if file.split('-')[0] not in validation_videos_dict.keys() or file.split('_p')[-1].split('.')[0] not in \
                    validation_videos_dict[file.split('-')[0]]:
                if ori_videos and '-ori_' not in file:
                    continue
                tra_files.append(file)
            else:
                if ori_videos and '-ori_' not in file:
                    continue
                val_files.append(file)
        elif '-ori_' in file:
            test_files.append(file)
    return tra_files, val_files, test_files


def get_tra_test_files_generalisation(augment_method):
    tra_files, val_files, test_files = [], [], []
    data_path = get_data_path('mixed', True)
    files = os.listdir(data_path)
    for file in files:
        with open(data_path + file, 'r') as f:
            feature_json = json.load(f)
            if feature_json['action_class'] == int(augment_method):
                if '-ori_' in file:
                    test_files.append(file)
            else:
                tra_files.append(file)
    random.shuffle(tra_files)
    val_files = tra_files[:int(valset_rate * len(tra_files))]
    tra_files = tra_files[int(valset_rate * len(tra_files)):]
    return tra_files, val_files, test_files


def get_body_part(feature, is_coco, body_part):
    """
    :param body_part: list, index0 for body, index1 for face, index2 for hands
    :return:
    """
    coco_body_part = [23, 91]
    halpe_body_part = [26, 94]
    new_features = []
    if body_part[0]:
        new_features += feature[:coco_body_part[0]].tolist() if is_coco else feature[:halpe_body_part[0]].tolist()
    if body_part[1]:
        new_features += feature[coco_body_part[0]:coco_body_part[1]].tolist() if is_coco else feature[
                                                                                              halpe_body_part[0]:
                                                                                              halpe_body_part[
                                                                                                  1]].tolist()
    if body_part[2]:
        new_features += feature[coco_body_part[1]:].tolist() if is_coco else feature[halpe_body_part[1]:].tolist()
    return np.array(new_features)


def get_inputs_size(is_coco, body_part):
    input_size = 0
    if body_part[0]:
        input_size += coco_body_point_num if is_coco else halpe_body_point_num
    if body_part[1]:
        input_size += head_point_num
    if body_part[2]:
        input_size += hands_point_num
    return 3 * input_size


def get_l_pair(is_coco, body_part):
    l_pair = []
    if is_coco:
        if body_part[0]:
            l_pair += coco_body_l_pair
        if body_part[1]:
            l_pair += coco_head_l_pair
        if body_part[2]:
            l_pair += coco_hand_l_pair
    else:
        if body_part[0]:
            l_pair += halpe_body_l_pair
        if body_part[1]:
            l_pair += halpe_head_l_pair
        if body_part[2]:
            l_pair += halpe_hand_l_pair
    return l_pair


class Dataset(Dataset):
    def __init__(self, data_files, augment_method, is_coco, body_part, model, frame_sample_hop, sequence_length=99999):
        super(Dataset, self).__init__()
        self.files = data_files
        self.data_path = get_data_path(augment_method=augment_method, is_coco=is_coco)
        self.is_coco = is_coco
        self.body_part = body_part  # 1 for only body, 2 for head and body, 3 for hands and body, 4 for head, hands and body
        self.model = model
        self.frame_sample_hop = frame_sample_hop
        self.sequence_length = sequence_length

        self.features, self.labels = [], []
        index = 0
        for file in self.files:
            if self.model in ['stgcn', 'msgcn']:
                x, label = self.get_stgraph_data_from_file(file)
                self.features.append(x)
            elif 'gcn_' in self.model:
                x_list, label = self.get_graph_data_from_file(file)
                if type(x_list) == int:
                    continue
                else:
                    self.features.append(x_list)
            else:
                feature, label = self.get_data_from_file(file)
                if type(feature) == int or feature.size == 0 or feature.ndim == 0:
                    continue
                elif index == 0:
                    index += 1
                    if self.model in ['avg', 'perframe']:
                        self.features = feature
                    elif self.model in ['lstm', 'conv1d']:
                        self.features = [feature]
                else:
                    if self.model in ['avg', 'perframe']:
                        self.features = np.append(self.features, feature, axis=0)
                    elif self.model in ['lstm', 'conv1d']:
                        self.features.append(feature)
            if model == 'perframe':
                self.labels += label
            else:
                self.labels.append(label)

    def get_data_from_file(self, file):
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        features = []
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        first_id = -1
        for frame in feature_json['frames']:
            if frame['frame_id'] % self.frame_sample_hop == 0:
                first_id = frame['frame_id']
                break
        if first_id == -1:
            return 0, 0
        index = 0
        while len(features) < int(self.sequence_length / self.frame_sample_hop):
            if index == video_frame_num:
                break
            else:
                frame = feature_json['frames'][index]
                if frame['frame_id'] > first_id and frame['frame_id'] > len(features) * self.frame_sample_hop:
                    features.append(features[-1])
                else:
                    index += 1
                    if frame['frame_id'] - first_id > int(self.sequence_length / self.frame_sample_hop):
                        break
                    elif frame['frame_id'] % self.frame_sample_hop == 0:
                        frame_feature = np.array(frame['keypoints'])
                        frame_feature = get_body_part(frame_feature, self.is_coco, self.body_part)
                        frame_feature[:, 0] = (2 * frame_feature[:, 0] / frame_width) - 1
                        frame_feature[:, 1] = (2 * frame_feature[:, 1] / frame_height) - 1
                        frame_feature = frame_feature.reshape(1, frame_feature.size)[0]
                        features.append(frame_feature)
        if len(features) == 0:
            return 0, None
        features = np.array(features)
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        if self.model == 'avg':
            features = np.mean(features, axis=0)
            features = features.reshape(1, features.size)
        elif self.model == 'perframe':
            label = [label for _ in range(int(features.shape[0]))]
        elif self.model in ['lstm', 'conv1d']:
            features = torch.from_numpy(features)
        return features, label

    def get_graph_data_from_file(self, file):
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        x_list = [0, 0, 0]
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        first_id = -1
        for frame in feature_json['frames']:
            if frame['frame_id'] % self.frame_sample_hop == 0:
                first_id = frame['frame_id']
                break
        if first_id == -1:
            return 0, 0
        for index_body, body in enumerate(self.body_part):
            if body:
                index = 0
                b_p = [False, False, False]
                b_p[index_body] = True
                input_size = get_inputs_size(self.is_coco, b_p)
                x_tensor = torch.zeros((int(self.sequence_length / self.frame_sample_hop), int(input_size / 3), 3))
                frame_num = 0
                while frame_num < int(self.sequence_length / self.frame_sample_hop):
                    if index == video_frame_num:
                        x_tensor[frame_num] = x
                        frame_num += 1
                    else:
                        frame = feature_json['frames'][index]
                        if frame['frame_id'] > first_id and frame['frame_id'] > frame_num * self.frame_sample_hop:
                            x_tensor[frame_num] = x
                            frame_num += 1
                        else:
                            index += 1
                            if frame['frame_id'] - first_id > int(self.sequence_length / self.frame_sample_hop):
                                break
                            elif frame['frame_id'] % self.frame_sample_hop == 0:
                                frame_feature = np.array(frame['keypoints'])
                                frame_feature = get_body_part(frame_feature, self.is_coco, b_p)
                                frame_feature[:, 0] = (2 * frame_feature[:, 0] / frame_width) - 1
                                frame_feature[:, 1] = (2 * frame_feature[:, 1] / frame_height) - 1
                                x = torch.tensor(frame_feature)
                                x_tensor[frame_num] = x
                                frame_num += 1
                if frame_num == 0:
                    return 0, 0
                x_list[index_body] = x_tensor
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        return x_list, label

    def get_stgraph_data_from_file(self, file):
        x_list = [0, 0, 0]
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        for index_body, body in enumerate(self.body_part):
            if body:
                bp = [False, False, False]
                bp[index_body] = True
                x_l = np.zeros((3, int(self.sequence_length / self.frame_sample_hop),
                                int(get_inputs_size(self.is_coco, bp) / 3), 1))
                first_id = -1
                for frame in feature_json['frames']:
                    if frame['frame_id'] % self.frame_sample_hop == 0:
                        first_id = frame['frame_id']
                        break
                if first_id == -1:
                    return 0, 0, 0
                index = 0
                frame_num = 0
                while frame_num < int(self.sequence_length / self.frame_sample_hop):
                    if index == video_frame_num:
                        break
                    else:
                        frame = feature_json['frames'][index]
                        if frame['frame_id'] > first_id and frame['frame_id'] > frame_num * self.frame_sample_hop:
                            x_l[:, frame_num, :, 0] = frame_feature.T
                            frame_num += 1
                        else:
                            index += 1
                            if frame['frame_id'] - first_id > int(self.sequence_length / self.frame_sample_hop):
                                break
                            elif frame['frame_id'] % self.frame_sample_hop == 0:
                                frame_feature = np.array(frame['keypoints'])
                                frame_feature = get_body_part(frame_feature, self.is_coco, bp)
                                frame_feature[:, 0] = (2 * frame_feature[:, 0] / frame_width) - 1
                                frame_feature[:, 1] = (2 * frame_feature[:, 1] / frame_height) - 1
                                x_l[:, frame_num, :, 0] = frame_feature.T
                                frame_num += 1
                x_list[index_body] = x_l
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        return x_list, label

    def feature_transform(self, features, frame_width, frame_height):
        l_pair = get_l_pair(self.is_coco, self.body_part)
        frame_feature = np.zeros((len(l_pair), 2))
        for index, pair in enumerate(l_pair):
            frame_feature[index][0] = (features[pair[0]][0] - features[pair[1]][0]) / frame_width
            frame_feature[index][1] = (features[pair[0]][1] - features[pair[1]][1]) / frame_height
        return frame_feature

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        if self.model in ['avg', 'perframe']:
            return self.features.shape[0]
        elif self.model in ['lstm', 'gru', 'conv1d'] or 'gcn' in self.model:
            return len(self.features)


if __name__ == '__main__':
    augment_method = 'crop'
    is_coco = True
    tra_files, val_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco)
    print(len(tra_files), len(val_files), len(test_files))
    # dataset = Dataset(data_files=tra_files[int(len(tra_files) * 0.2):], action_recognition=1,
    #                   augment_method=augment_method, is_coco=is_coco, body_part=[True, True, True], model='lstm',
    #                   sample_fps=30)
    # features, labels = dataset.__getitem__(9)
    # print(features.shape, labels)
    # dataset = Dataset(data_files=tra_files[int(len(tra_files) * 0.2):], action_recognition=1,
    #                   augment_method=augment_method, is_coco=is_coco, body_part=[True, True, True], model='lstm',
    #                   sample_fps=30)
    # features, labels = dataset.__getitem__(9)
    # print(features.shape, labels)
