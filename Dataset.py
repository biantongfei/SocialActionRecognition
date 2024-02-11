import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torch
from torch_geometric.data import InMemoryDataset, Data

from Models import get_points_num

testset_rate = 0.5
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
    elif augment_method == 'mixed_same':
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/mixed/same/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/mixed/same/halpe136/'
    elif augment_method == 'mixed_large':
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/mixed/large/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/mixed/large/halpe136/'
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
            if ori_videos and '-ori_' not in file:
                continue
            tra_files.append(file)
        elif '-ori_' in file:
            test_files.append(file)
    return tra_files, test_files


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


def get_labels(att_class, act_class):
    if att_class in [0, 2]:
        intent_class = 0
        attitude_class = att_class if att_class == 0 else 1
    elif att_class == 1:
        intent_class = 1
        attitude_class = 2
    else:
        intent_class = 2
        attitude_class = 2
    return intent_class, attitude_class, act_class


class Dataset(Dataset):
    def __init__(self, data_files, augment_method, is_coco, body_part, model, sample_fps, video_len=99999,
                 empty_frame=False):
        super(Dataset, self).__init__()
        self.files = data_files
        self.data_path = get_data_path(augment_method=augment_method, is_coco=is_coco)
        self.augment_method = augment_method
        self.is_coco = is_coco
        self.body_part = body_part  # 1 for only body, 2 for head and body, 3 for hands and body, 4 for head, hands and body
        self.model = model
        self.sample_fps = sample_fps
        self.video_len = video_len
        self.empty_frame = empty_frame  # how to deal with empty frames: 'zero' for zero padding; 'same' for last frame padding

        self.features, self.labels, self.frame_number_list = 0, [], []
        index = 0
        for file in self.files:
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
            self.frame_number_list.append(int(feature.shape[0]))
        self.max_length = max(self.frame_number_list)

    def get_data_from_file(self, file):
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        features = []
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        first_id = feature_json['frames'][0]['frame_id']
        index = 0
        while len(features) < int(self.video_len * self.sample_fps):
            if index == video_frame_num:
                break
            else:
                frame = feature_json['frames'][index]
                if self.empty_frame and frame['frame_id'] > len(features) * (video_fps / self.sample_fps):
                    if self.empty_frame == 'zero':
                        features.append(np.zeros((2 * get_points_num(is_coco=self.is_coco, body_part=self.body_part))))
                    elif self.empty_frame == 'same':
                        features.append(features[-1])
                else:
                    index += 1
                    if frame['frame_id'] - first_id > int(video_fps * self.video_len):
                        break
                    elif frame['frame_id'] % int(video_fps / self.sample_fps) == 0:
                        # box_x, box_y, box_width, box_height = frame['box'][0], frame['box'][1], frame['box'][2], \
                        #     frame['box'][3]
                        frame_feature = np.array(frame['keypoints'])[:, :2]
                        # frame_feature[:, 0] = (frame_feature[:, 0] - box_x) / box_width
                        # frame_feature[:, 1] = (frame_feature[:, 1] - box_y) / box_height
                        frame_feature[:, 0] = (frame_feature[:, 0] / frame_width) - 0.5
                        frame_feature[:, 1] = (frame_feature[:, 1] / frame_height) - 0.5
                        frame_feature = get_body_part(frame_feature, self.is_coco, self.body_part)
                        # frame_feature = np.append(frame_feature, [
                        #     [(box_x - (frame_width / 2)) / frame_width, (box_y - (frame_height / 2)) / frame_height],
                        #     [box_width / frame_width, box_height / frame_height]], axis=0)
                        frame_feature = frame_feature.reshape(1, frame_feature.size)[0]
                        features.append(frame_feature)
        if len(features) == 0:
            return 0, None
        features = np.array(features)
        label = get_labels(feature_json['attitude_class'], feature_json['action_class'])
        if self.model == 'avg':
            features = np.mean(features, axis=0)
            features = features.reshape(1, features.size)
        elif self.model == 'perframe':
            label = [label for _ in range(int(features.shape[0]))]
        elif self.model in ['lstm', 'gru', 'conv1d']:
            features = torch.from_numpy(features)
        return features, label

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        if self.model in ['avg', 'perframe']:
            return self.features.shape[0]
        elif self.model in ['lstm', 'gru', 'conv1d']:
            return len(self.features)

#
# class MyCustomDataset(InMemoryDataset):
#     def __init__():
#         self.filename =..  # List of raw files, in your case point cloud
#         super(MyCustomDataset, self).__init()
#
#     @property
#     def raw_file_names(self):
#         return self.filename
#
#     @property
#     def processed_file_names(self):
#         """ return list of files should be in processed dir, if found - skip processing."""
#         processed_filename = []
#         return processed_filename
#
#     def download(self):
#         pass
#
#     def process(self):
#         for file in self.raw_paths:
#             self._process_one_step(file)
#
#     def _process_one_step(self, path):
#         out_path = (self.processed_dir, "some_unique_filename.pt")
#         # read your point cloud here,
#         # convert point cloud to Data object
#         data = Data(x=node_features,
#                     edge_index=edge_index,
#                     edge_attr=edge_attr,
#                     y=label  # you can add more arguments as you like
#                     )
#         torch.save(data, out_path)
#         return
#
#     def __len__(self):
#         return len(self.processed_file_names)
#
#     def __getitem__(self, idx):
#         data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
#         return data


if __name__ == '__main__':
    augment_method = 'crop'
    is_coco = True
    tra_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco, not_add_class=False)
    print(len(tra_files))
    dataset = Dataset(data_files=tra_files[int(len(tra_files) * 0.2):], action_recognition=1,
                      augment_method=augment_method, is_coco=is_coco, body_part=[True, True, True], model='lstm',
                      sample_fps=30)
    features, labels = dataset.__getitem__(9)
    print(features.shape, labels)
    dataset = Dataset(data_files=tra_files[int(len(tra_files) * 0.2):], action_recognition=1,
                      augment_method=augment_method, is_coco=is_coco, body_part=[True, True, True], model='lstm',
                      sample_fps=30)
    features, labels = dataset.__getitem__(9)
    print(features.shape, labels)
