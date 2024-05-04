import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

coco_body_point_num = 23
halpe_body_point_num = 26
head_point_num = 68
hands_point_num = 42
valset_rate = 0.1
testset_rate = 0.4
coco_point_num = 133
halpe_point_num = 136
video_fps = 30


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
    elif augment_method == 'mixed':
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/mixed/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/mixed/halpe136/'
    return data_path


def get_tra_test_files(augment_method, is_coco, ori_videos=False):
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


def get_inputs_size(is_coco, body_part, gcn=False):
    input_size = 0
    if body_part[0]:
        input_size += coco_body_point_num if is_coco else halpe_body_point_num
    if body_part[1]:
        input_size += head_point_num
    if body_part[2]:
        input_size += hands_point_num
    if not gcn:
        input_size += len(get_l_pair(is_coco, body_part))
        print(len(get_l_pair(is_coco, body_part)))
    return 2 * input_size


def get_l_pair(is_coco, body_part):
    l_pair = []
    if is_coco:
        if body_part[0]:
            l_pair += [[0, 1], [0, 2], [1, 3], [2, 4],  # Head
                       [5, 7], [7, 9], [6, 8], [8, 10],  # Body
                       [5, 6], [11, 12], [5, 11], [6, 12],
                       [11, 13], [12, 14], [13, 15], [14, 16],
                       [15, 17], [15, 18], [15, 19], [16, 20], [16, 21], [16, 22]]
        if body_part[1]:
            l_pair += [[23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
                       [32, 33], [33, 34], [34, 35],
                       [35, 36], [36, 37], [37, 38], [38, 39], [40, 41], [41, 42], [42, 43], [43, 44], [45, 46],
                       [46, 47], [47, 48], [48, 49],
                       [50, 51], [51, 52], [52, 53], [54, 55], [55, 56], [56, 57], [57, 58], [59, 60], [60, 61],
                       [61, 62], [62, 63], [63, 64],
                       [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [71, 72], [72, 73], [73, 74], [74, 75],
                       [75, 76], [76, 77], [77, 78],
                       [78, 79], [79, 80], [80, 81], [81, 82], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87],
                       [87, 88], [88, 89], [89, 90]]
        if body_part[2]:
            l_pair += [[91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
                       [100, 101], [101, 102],
                       [102, 103], [91, 104], [104, 105], [105, 106], [106, 107], [91, 108], [108, 109], [109, 110],
                       [110, 111], [112, 113],
                       [113, 114], [114, 115], [115, 116], [112, 117], [117, 118], [118, 119], [119, 120], [112, 121],
                       [121, 122], [122, 123],
                       [123, 124], [112, 125], [125, 126], [126, 127], [127, 128], [112, 129], [129, 130], [130, 131],
                       [131, 132]]
    else:
        if body_part[0]:
            l_pair += [[0, 1], [0, 2], [1, 3], [2, 4],  # Head
                       [5, 18], [6, 18], [5, 7], [7, 9], [6, 8], [8, 10],  # Body
                       [17, 18], [18, 19], [19, 11], [19, 12],
                       [11, 13], [12, 14], [13, 15], [14, 16],
                       [20, 24], [21, 25], [23, 25], [22, 24], [15, 24], [16, 25],  # Foot
                       ]
        if body_part[1]:
            l_pair += [[26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [34, 35],
                       [35, 36], [36, 37], [37, 38],  # Face
                       [38, 39], [39, 40], [40, 41], [41, 42], [43, 44], [44, 45], [45, 46], [46, 47], [48, 49],
                       [49, 50], [50, 51], [51, 52],  # Face
                       [53, 54], [54, 55], [55, 56], [57, 58], [58, 59], [59, 60], [60, 61], [62, 63], [63, 64],
                       [64, 65], [65, 66], [66, 67],  # Face
                       [68, 69], [69, 70], [70, 71], [71, 72], [72, 73], [74, 75], [75, 76], [76, 77], [77, 78],
                       [78, 79], [79, 80], [80, 81],  # Face
                       [81, 82], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87], [87, 88], [88, 89], [89, 90],
                       [90, 91], [91, 92], [92, 93]  # Face
                       ]
        if body_part[2]:
            l_pair += [[94, 95], [95, 96], [96, 97], [97, 98], [94, 99], [99, 100], [100, 101], [101, 102], [94, 103],
                       [103, 104], [104, 105],  # LeftHand
                       [105, 106], [94, 107], [107, 108], [108, 109], [109, 110], [94, 111], [111, 112], [112, 113],
                       [113, 114],  # LeftHand
                       [115, 116], [116, 117], [117, 118], [118, 119], [115, 120], [120, 121], [121, 122], [122, 123],
                       [115, 124], [124, 125],  # RightHand
                       [125, 126], [126, 127], [115, 128], [128, 129], [129, 130], [130, 131], [115, 132], [132, 133],
                       [133, 134], [134, 135]  # RightHand
                       ]
    return l_pair


class Dataset(Dataset):
    def __init__(self, data_files, augment_method, is_coco, body_part, model, sample_fps, video_len=99999):
        super(Dataset, self).__init__()
        self.files = data_files
        self.data_path = get_data_path(augment_method=augment_method, is_coco=is_coco)
        self.augment_method = augment_method
        self.is_coco = is_coco
        self.body_part = body_part  # 1 for only body, 2 for head and body, 3 for hands and body, 4 for head, hands and body
        self.model = model
        self.sample_fps = sample_fps
        self.video_len = video_len

        self.features, self.labels, self.frame_number_list = 0, [], []
        index = 0
        for file in self.files:
            if self.model == 'stgcn':
                x, edge_index, edge_attr, label = self.get_stgraph_data_from_file(file)
                if type(x) == int:
                    continue
                elif type(self.features) == int:
                    self.features = [(x, edge_index, edge_attr)]
                else:
                    self.features.append((x, edge_index, edge_attr))
            elif self.model in ['gcn_lstm', 'gcn_gru', 'gcn_conv1d', 'gcn_gcn']:
                x, edge_index, edge_attr, label = self.get_graph_data_from_file(file)
                if type(x) == int:
                    continue
                elif type(self.features) == int:
                    self.features = [(x, edge_index, edge_attr)]
                else:
                    self.features.append((x, edge_index, edge_attr))
            else:
                feature, label = self.get_data_from_file(file)
                if type(feature) == int or feature.size == 0 or feature.ndim == 0:
                    continue
                elif index == 0:
                    index += 1
                    if self.model in ['avg', 'perframe']:
                        self.features = feature
                    elif self.model in ['lstm', 'gru', 'conv1d']:
                        self.features = [feature]
                else:
                    if self.model in ['avg', 'perframe']:
                        self.features = np.append(self.features, feature, axis=0)
                    elif self.model in ['lstm', 'gru', 'conv1d']:
                        self.features.append(feature)
            if model == 'perframe':
                self.labels += label
            else:
                self.labels.append(label)
            self.frame_number_list.append(
                int(feature.shape[0]) if model not in ['gcn_conv1d', 'gcn_lstm', 'gcn_gru', 'gcn_gcn',
                                                       'stgcn'] else int(x.shape[0]))
        self.max_length = max(self.frame_number_list)

    def get_data_from_file(self, file):
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        features = []
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        first_id = -1
        for frame in feature_json['frames']:
            if frame['frame_id'] % (video_fps / self.sample_fps) == 0:
                first_id = frame['frame_id']
                break
        if first_id == -1:
            return 0, 0
        index = 0
        while len(features) < int(self.video_len * self.sample_fps):
            if index == video_frame_num:
                break
            else:
                frame = feature_json['frames'][index]
                if frame['frame_id'] > first_id and frame['frame_id'] > len(features) * (video_fps / self.sample_fps):
                    features.append(features[-1])
                else:
                    index += 1
                    if frame['frame_id'] - first_id > int(video_fps * self.video_len):
                        break
                    elif frame['frame_id'] % int(video_fps / self.sample_fps) == 0:
                        frame_feature = np.array(frame['keypoints'])[:, :2]
                        frame_feature = get_body_part(frame_feature, self.is_coco, self.body_part)
                        frame_feature[:, 0] = (2 * frame_feature[:, 0] / frame_width) - 1
                        frame_feature[:, 1] = (2 * frame_feature[:, 1] / frame_height) - 1
                        frame_feature = np.append(frame_feature,
                                                  self.feature_transform(frame_feature, frame_width, frame_height),
                                                  axis=0)
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
        elif self.model in ['lstm', 'gru', 'conv1d']:
            features = torch.from_numpy(features)
        return features, label

    def get_graph_data_from_file(self, file):
        edge_index = torch.tensor(np.array(get_l_pair(self.is_coco, self.body_part)), dtype=torch.long).t().contiguous()
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        first_id = -1
        for frame in feature_json['frames']:
            if frame['frame_id'] % (video_fps / self.sample_fps) == 0:
                first_id = frame['frame_id']
                break
        if first_id == -1:
            return 0, 0
        index = 0
        while len(x_list) < int(self.video_len * self.sample_fps):
            if index == video_frame_num:
                break
            else:
                frame = feature_json['frames'][index]
                if frame['frame_id'] > first_id and frame['frame_id'] > len(x_list) * (video_fps / self.sample_fps):
                    x_list.append(x_list[-1])
                    edge_index_list.append(edge_index_list[-1])
                    edge_attr_list.append(edge_attr_list[-1])
                else:
                    index += 1
                    if frame['frame_id'] - first_id > int(video_fps * self.video_len):
                        break
                    elif frame['frame_id'] % int(video_fps / self.sample_fps) == 0:
                        frame_feature = np.array(frame['keypoints'])[:, :2]
                        frame_feature = get_body_part(frame_feature, self.is_coco, self.body_part)
                        frame_feature[:, 0] = (2 * frame_feature[:, 0] / frame_width) - 1
                        frame_feature[:, 1] = (2 * frame_feature[:, 1] / frame_height) - 1
                        x = torch.tensor(frame_feature)
                        edge_attr = torch.tensor(self.feature_transform(frame_feature, frame_width, frame_height))
                        x_list.append(x)
                        edge_index_list.append(edge_index)
                        edge_attr_list.append(edge_attr)
        while len(x_list) < self.sample_fps * self.video_len:
            x_list.append(x_list[-1])
            edge_index_list.append(edge_index_list[-1])
            edge_attr_list.append(edge_attr_list[-1])
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        if len(x_list) == 0:
            return 0
        return np.array(x_list), np.array(edge_index_list), np.array(edge_attr_list), label

    def get_stgraph_data_from_file(self, file):
        input_size = get_inputs_size(self.is_coco, self.body_part, gcn=True)
        x_list, edge_index_list, edge_attr_list = [], [], []
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        first_id = -1
        for frame in feature_json['frames']:
            if frame['frame_id'] % (video_fps / self.sample_fps) == 0:
                first_id = frame['frame_id']
                break
        if first_id == -1:
            return 0, 0, 0
        index = 0
        frame_num = 0
        while frame_num < int(self.video_len * self.sample_fps):
            if index == video_frame_num:
                break
            else:
                frame = feature_json['frames'][index]
                if frame['frame_id'] > first_id and frame['frame_id'] > frame_num * (video_fps / self.sample_fps):
                    edge_index = [
                        [i[0] + int(frame_num * input_size / 2), i[1] + int(len(x_list) * input_size / 2)] for i
                        in get_l_pair(self.is_coco, self.body_part)]
                    x_list += x
                    edge_index_list += edge_index
                    edge_attr_list += edge_attr
                    frame_num += 1
                else:
                    index += 1
                    if frame['frame_id'] - first_id > int(video_fps * self.video_len):
                        break
                    elif frame['frame_id'] % int(video_fps / self.sample_fps) == 0:
                        frame_feature = np.array(frame['keypoints'])[:, :2]
                        frame_feature = get_body_part(frame_feature, self.is_coco, self.body_part)
                        frame_feature[:, 0] = (2 * frame_feature[:, 0] / frame_width) - 1
                        frame_feature[:, 1] = (2 * frame_feature[:, 1] / frame_height) - 1
                        x = frame_feature.tolist()
                        edge_index = [
                            [i[0] + int(frame_num * input_size / 2), i[1] + int(len(x_list) * input_size / 2)] for i
                            in get_l_pair(self.is_coco, self.body_part)]
                        edge_attr = self.feature_transform(frame_feature, frame_width, frame_height).tolist()
                        x_list += x
                        edge_index_list += edge_index
                        edge_attr_list += edge_attr
                        frame_num += 1
        while frame_num < self.video_len * self.sample_fps:
            edge_index = [
                [i[0] + int(frame_num * input_size / 2), i[1] + int(len(x_list) * input_size / 2)] for i
                in get_l_pair(self.is_coco, self.body_part)]
            x_list += x
            edge_index_list += edge_index
            edge_attr_list += edge_attr
            frame_num += 1
        for index in range(len(x_list) - int(input_size / 2)):
            edge_index_list.append([index, int(index + input_size / 2)])
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        if len(x_list) == 0:
            return 0
        return np.array(x_list), np.array(edge_index_list), np.array(edge_attr_list), label

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
