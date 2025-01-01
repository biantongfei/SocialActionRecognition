import os
import json
import random
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import coco_body_point_num, head_point_num, hands_point_num, coco_body_l_pair, head_l_pair, hand_l_pair, \
    visible_threshold_score, harper_l_pair, harper_body_point_num, camera_name_list, contact_min_distance, attack_class

video_path = '../jpl_augmented_videos/'


def get_data_path(augment_method):
    if augment_method == 'crop':
        data_path = '../JPL_Augmented_Posefeatures/crop/coco_wholebody/'
    elif augment_method == 'noise':
        data_path = '../JPL_Augmented_Posefeatures/gaussian/coco_wholebody/'
    elif augment_method in ['mixed']:
        data_path = '../JPL_Augmented_Posefeatures/mixed/coco_wholebody/'
    return data_path


def get_tra_test_files(randnum=None):
    tra_files = [i for i in os.listdir('../JPL_Augmented_Posefeatures/mixed/coco_wholebody/train/') if 'json' in i]
    val_files = [i for i in os.listdir('../JPL_Augmented_Posefeatures/mixed/coco_wholebody/validation/') if 'json' in i]
    test_files = [i for i in os.listdir('../JPL_Augmented_Posefeatures/mixed/coco_wholebody/test/') if 'json' in i]
    # tra_files = [i for i in tra_files if 'ori_' in i]

    if randnum:
        random.seed(randnum)
        random.shuffle(tra_files)
    return tra_files, val_files, test_files


def get_body_part(feature, body_part):
    """
    :param body_part: list, index0 for body, index1 for face, index2 for hands
    :return:
    """
    point_nums = [coco_body_point_num, coco_body_point_num + head_point_num]
    new_features = []
    if body_part[0]:
        new_features += feature[:point_nums[0]].tolist()
    if body_part[1]:
        new_features += feature[point_nums[0]:point_nums[1]].tolist()
    if body_part[2]:
        new_features += feature[point_nums[1]:].tolist()
    return np.array(new_features)


def get_inputs_size(body_part):
    input_size = 0
    if body_part[0]:
        input_size += coco_body_point_num
    if body_part[1]:
        input_size += head_point_num
    if body_part[2]:
        input_size += hands_point_num
    return 3 * input_size


def get_l_pair(body_part):
    l_pair = []
    if body_part[0]:
        l_pair += coco_body_l_pair
    if body_part[1]:
        l_pair += head_l_pair
    if body_part[2]:
        l_pair += hand_l_pair
    return l_pair


def get_first_id(feature_json, frame_sample_hop, hop):
    for frame in feature_json['frames']:
        if frame['frame_id'] % frame_sample_hop == hop:
            return frame['frame_id']
    return -1


class JPL_Dataset(Dataset):
    def __init__(self, data_files, augment_method, body_part, model, frame_sample_hop, subset, only_visible_point=False,
                 sequence_length=99999):
        super(Dataset, self).__init__()
        self.files = data_files
        self.data_path = get_data_path(augment_method=augment_method)
        if subset == 'train':
            self.data_path += 'train/'
        elif subset == 'validation':
            self.data_path += 'validation/'
        elif subset == 'test':
            self.data_path += 'test/'
        self.subset = subset
        self.body_part = body_part
        self.model = model
        self.frame_sample_hop = frame_sample_hop
        self.only_visible_point = only_visible_point
        self.sequence_length = sequence_length
        self.features, self.labels = [], []
        self.out_files = []
        index = 0
        for file in self.files:
            for hop in range(self.frame_sample_hop):
                if self.model in ['stgcn', 'msgcn', 'dgstgcn']:
                    x, label = self.get_stgraph_data_from_file(file, hop)
                    self.features.append(x)
                elif 'gcn_' in self.model:
                    x_list, label = self.get_graph_data_from_file(file, hop)
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
                        elif self.model in ['lstm', 'conv1d', 'tran']:
                            self.features = [feature]
                    else:
                        if self.model in ['avg', 'perframe']:
                            self.features = np.append(self.features, feature, axis=0)
                        elif self.model in ['lstm', 'conv1d', 'tran']:
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
                        box_x, box_y, box_width, box_height = frame['box'][0], frame['box'][1], frame['box'][2], \
                            frame['box'][3]
                        frame_feature = np.array(frame['keypoints'])
                        frame_feature = get_body_part(frame_feature, self.body_part)
                        # frame_feature[:, 0] = frame_feature[:, 0] / frame_width-0.5
                        # frame_feature[:, 1] = frame_feature[:, 1] / frame_height-0.5
                        frame_feature[:, 0] = (frame_feature[:, 0] - box_x) / box_width
                        frame_feature[:, 1] = (frame_feature[:, 1] - box_y) / box_height
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
        elif self.model in ['lstm', 'conv1d', 'tran']:
            features = torch.from_numpy(features)
        return features, label

    def get_graph_data_from_file(self, file, hop):
        with open(self.data_path + file, 'r') as f:
            feature_json = json.load(f)
            f.close()
        x_list = [0, 0, 0]
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        # first_id = feature_json['frames'][hop]['frame_id']
        first_id = get_first_id(feature_json, self.frame_sample_hop, hop)
        if first_id == -1:
            return 0, 0
        for index_body, body in enumerate(self.body_part):
            if body:
                index = 0
                b_p = [False, False, False]
                b_p[index_body] = True
                input_size = get_inputs_size(b_p)
                x_tensor = torch.zeros((int(self.sequence_length / self.frame_sample_hop), int(input_size / 3), 3))
                frame_num = 0
                while frame_num < int(self.sequence_length / self.frame_sample_hop):
                    if index == video_frame_num:
                        if self.only_visible_point:
                            x_tensor[frame_num] = torch.zeros((int(input_size / 3), 3))
                        else:
                            x_tensor[frame_num] = x
                        frame_num += 1
                    else:
                        frame = feature_json['frames'][index]
                        if frame['frame_id'] % self.frame_sample_hop == hop:
                            if frame['frame_id'] > first_id and frame['frame_id'] > frame_num * self.frame_sample_hop:
                                if self.only_visible_point:
                                    x_tensor[frame_num] = torch.zeros((int(input_size / 3), 3))
                                else:
                                    x_tensor[frame_num] = x
                                frame_num += 1
                            else:
                                index += 1
                                if frame['frame_id'] - first_id > self.sequence_length:
                                    break
                                else:
                                    frame_feature = np.array(frame['keypoints'])
                                    frame_feature = get_body_part(frame_feature, b_p)
                                    frame_feature[:, 0] = 2 * (frame_feature[:, 0] / frame_width - 0.5)
                                    frame_feature[:, 1] = 2 * (frame_feature[:, 1] / frame_height - 0.5)
                                    # frame_feature[:, 0] = (frame_feature[:, 0] - box_x) / box_width
                                    # frame_feature[:, 1] = (frame_feature[:, 1] - box_y) / box_height
                                    if self.only_visible_point:
                                        binary_result = (frame_feature[:, 2] > visible_threshold_score).astype(int)
                                        frame_feature[:, 2] = binary_result
                                        frame_feature[:, 0] = binary_result * frame_feature[:, 0]
                                        frame_feature[:, 1] = binary_result * frame_feature[:, 1]
                                    x = torch.tensor(frame_feature)
                                    x_tensor[frame_num] = x
                                    frame_num += 1
                        else:
                            index += 1
                            frame_num += 1
                if frame_num == 0:
                    return 0, 0
                x_list[index_body] = x_tensor
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        self.out_files.append(file)
        return x_list, label

    def get_stgraph_data_from_file(self, file, hop):
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
                x_l = np.zeros((3, int(self.sequence_length / self.frame_sample_hop), int(get_inputs_size(bp) / 3), 1))
                first_id = get_first_id(feature_json, self.frame_sample_hop, hop)
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
                            elif frame['frame_id'] % self.frame_sample_hop == hop:
                                box_x, box_y, box_width, box_height = frame['box'][0], frame['box'][1], frame['box'][2], \
                                    frame['box'][3]
                                frame_feature = np.array(frame['keypoints'])
                                frame_feature = get_body_part(frame_feature, bp)
                                frame_feature[:, 0] = 2 * (frame_feature[:, 0] / frame_width - 0.5)
                                frame_feature[:, 1] = 2 * (frame_feature[:, 1] / frame_height - 0.5)
                                # frame_feature[:, 0] = (frame_feature[:, 0] - box_x) / box_width
                                # frame_feature[:, 1] = (frame_feature[:, 1] - box_y) / box_height
                                if self.only_visible_point:
                                    binary_result = (frame_feature[:, 2] > visible_threshold_score).astype(int)
                                    frame_feature[:, 2] = binary_result
                                    frame_feature[:, 0] = binary_result * frame_feature[:, 0]
                                    frame_feature[:, 1] = binary_result * frame_feature[:, 1]
                                x_l[:, frame_num, :, 0] = frame_feature.T
                                frame_num += 1
                x_list[index_body] = x_l
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        return x_list, label

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        if self.model in ['avg', 'perframe']:
            return self.features.shape[0]
        elif self.model in ['lstm', 'gru', 'conv1d', 'tran'] or 'gcn' in self.model:
            return len(self.features)


def get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length, augment_method='mixed', subset='all',
                    randnum=None, fixed_files=None):
    print('Loading data from JPL %s dataset' % augment_method)
    subset_list = []
    result_str = ''
    if fixed_files:
        trainset = JPL_Dataset(data_files=fixed_files, augment_method=augment_method, body_part=body_part,
                               model=model, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length,
                               subset='train')
        subset_list.append(trainset)
        result_str += 'Train_set_size: %d, ' % len(trainset)
    elif model != 'r3d':
        tra_files, val_files, test_files = get_tra_test_files(randnum)
        if subset != 'test':
            trainset = JPL_Dataset(data_files=tra_files, augment_method=augment_method, body_part=body_part,
                                   model=model, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length,
                                   subset='train')
            subset_list.append(trainset)
            result_str += 'Train_set_size: %d, ' % len(trainset)
        if subset == 'all':
            valset = JPL_Dataset(data_files=val_files, augment_method=augment_method, body_part=body_part, model=model,
                                 frame_sample_hop=frame_sample_hop, sequence_length=sequence_length,
                                 subset='validation')
            subset_list.append(valset)
            result_str += 'Validation_set_size: %d, ' % len(valset)
        if subset != 'train':
            test_set = JPL_Dataset(data_files=test_files, augment_method=augment_method, body_part=body_part,
                                   model=model, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length,
                                   subset='test')
            subset_list.append(test_set)
            result_str += 'Test_set_size: %d.' % len(test_set)

    else:
        tra_files, val_files, test_files = get_tra_test_files(randnum)
        tra_files = [i for i in tra_files if 'noise' not in i]
        if subset != 'test':
            trainset = ImagesDataset(data_files=tra_files, frame_sample_hop=frame_sample_hop,
                                     sequence_length=sequence_length, subset='train')
            subset_list.append(trainset)
            result_str += 'Train_set_size: %d, ' % len(trainset)
        if subset == 'all':
            valset = ImagesDataset(data_files=val_files, frame_sample_hop=frame_sample_hop,
                                   sequence_length=sequence_length,
                                   subset='validation')
            subset_list.append(valset)
            result_str += 'Validation_set_size: %d, ' % len(valset)
        if subset != 'train':
            testset = ImagesDataset(data_files=test_files, frame_sample_hop=frame_sample_hop,
                                    sequence_length=sequence_length, subset='test')
            subset_list.append(testset)
            result_str += 'Test_set_size: %d, ' % len(testset)
    print(result_str)
    return tuple(subset_list) if len(subset_list) > 1 else subset_list[0]


class ImagesDataset(Dataset):
    def __init__(self, data_files, frame_sample_hop, sequence_length, subset):
        self.frame_sample_hop = frame_sample_hop
        self.json_files = data_files
        self.sequence_length = sequence_length
        self.r3d_image_size = 112
        self.json_data_path = get_data_path(augment_method='mixed')
        if subset == 'train':
            self.json_data_path += 'train/'
        elif subset == 'validation':
            self.json_data_path += 'validation/'
        elif subset == 'test':
            self.json_data_path += 'test/'
        self.video_files, self.bboxes, self.labels, self.null_files = [], [], [], []
        self.get_bboxes_labels_from_file()
        self.get_images_from_file()
        self.json_files = [item for item in self.json_files if item not in self.null_files]

    def get_bboxes_labels_from_file(self):
        for file in self.json_files:
            bboxes = {}
            with open(self.json_data_path + file, 'r') as f:
                feature_json = json.load(f)
                f.close()
            for frame in feature_json['frames']:
                if frame['frame_id'] <= self.sequence_length:
                    bboxes[frame['frame_id']] = frame['box']
            if len(bboxes.keys()) == 0:
                self.null_files.append(file)
                continue
            self.video_files.append(feature_json['video_name'])
            self.bboxes.append(bboxes)
            self.labels.append((feature_json['intention_class'], feature_json['attitude_class'],
                                feature_json['action_class']))

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, item):
        return self.videos[item], (self.labels[item][0], self.labels[item][1], self.labels[item][2])

    def get_images_from_file(self):
        self.videos = torch.zeros(
            (len(self.json_files), 3, self.sequence_length, self.r3d_image_size, self.r3d_image_size))
        for index, file in enumerate(self.video_files):
            cap = cv2.VideoCapture(video_path + file)
            bboxes = self.bboxes[index]
            images = torch.zeros((3, self.sequence_length, self.r3d_image_size, self.r3d_image_size))
            frame_id = -1
            for i in bboxes.keys():
                if i >= self.sequence_length:
                    break
                while frame_id < i:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_id += 1
                x, y, w, h = bboxes[i]
                cropped_frame = frame[int(y):int(y + h), int(x):int(x + w)]
                if cropped_frame is None or cropped_frame.size == 0:
                    continue
                cropped_frame = cv2.resize(cropped_frame, (self.r3d_image_size, self.r3d_image_size),
                                           interpolation=cv2.INTER_CUBIC)
                cropped_frame = torch.Tensor(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                cropped_frame = cropped_frame.permute(2, 0, 1)
                images[:, i, :, :] = cropped_frame

            self.videos[index] = images


class HARPER_Dataset(Dataset):
    def __init__(self, data_path, files, sequence_length, frames_before_event, multi_angle=False, train=False,
                 augment_method='original'):
        self.data_path = data_path
        self.files = files
        self.sequence_length = sequence_length
        self.frames_before_event = frames_before_event
        self.multi_angle = multi_angle
        self.train = train
        self.augment_method = augment_method
        self.features = []
        self.distances = []
        self.labels = []
        self.not_interact_downsample_rate = 0.1
        self.get_pose_sequences()

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def __len__(self):
        return len(self.features)

    def check_future_attack(self, attack_label, first_id, frames):
        check_duration = 5
        if attack_label == 3:
            for i in range(first_id + self.sequence_length,
                           first_id + self.frames_before_event + self.sequence_length - check_duration):
                if str(i) in frames.keys():
                    if frames[str(i)]['min_distance'] < contact_min_distance:
                        for ii in range(i, i + check_duration):
                            if str(ii) in frames.keys():
                                if frames[str(ii)]['min_distance'] < contact_min_distance:
                                    if ii == i + check_duration - 1:
                                        return 2
        return attack_label

    def check_current_attack(self, attack_label, first_id, frames):
        check_duration = 5
        if attack_label == 3:
            for i in range(first_id, first_id + self.sequence_length - check_duration):
                if str(i) in frames.keys():
                    if frames[str(i)]['min_distance'] < contact_min_distance:
                        for ii in range(i, i + check_duration):
                            if str(ii) in frames.keys():
                                if frames[str(ii)]['min_distance'] < contact_min_distance:
                                    if ii == i + check_duration - 1:
                                        return 2
        return attack_label

    def get_pose_sequences(self):
        for file in self.files:
            if 'json' in file:
                with open(self.data_path + file, 'r') as f:
                    feature_json = json.load(f)
                    f.close()
                interact_start_frame = feature_json['interact_start_frame'] if feature_json['interact_start_frame'] else \
                    feature_json['video_frames_number']
                interact_end_frame = feature_json['interact_end_frame'] if feature_json['interact_end_frame'] else \
                    feature_json['video_frames_number']
                if not self.multi_angle:
                    for camera in camera_name_list:
                        if feature_json['cameras'][camera]['frames']:
                            frame_width, frame_height = feature_json['cameras'][camera]['frame_size'][0], \
                                feature_json['cameras'][camera]['frame_size'][1]
                            frames = feature_json['cameras'][camera]['frames']
                            down_sample_count = 0
                            for frame_index in range(0, int(
                                    list(frames.keys())[-1]) - self.sequence_length - self.frames_before_event):
                                frame_num = 0
                                useful_num = 0
                                x_tensor = torch.zeros((self.sequence_length, harper_body_point_num, 3))
                                distance = [-1 for _ in range(self.sequence_length)]
                                while frame_num < self.sequence_length:
                                    if str(frame_index + frame_num) in frames.keys():
                                        frame_feature = np.array(frames[str(frame_index + frame_num)]['keypoints'])
                                        frame_feature[:, 0] = 2 * (frame_feature[:, 0] / frame_width - 0.5)
                                        frame_feature[:, 1] = 2 * (frame_feature[:, 1] / frame_height - 0.5)
                                        x = torch.tensor(frame_feature)
                                        x_tensor[frame_num] = x
                                        distance[frame_num] = frames[str(frame_index + frame_num)]['min_distance']
                                        useful_num += 1
                                    elif useful_num > 0:
                                        x_tensor[frame_num] = x_tensor[frame_num - 1]
                                        distance[frame_num] = distance[frame_num - 1]
                                    frame_num += 1
                                if useful_num == 0:
                                    continue
                                # contact_label = feature_json[
                                #     'contact_class'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 0
                                # intent_label = feature_json[
                                #     'intent_class'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 0
                                # attitude_label = feature_json[
                                #     'attitude_class'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 2
                                attack_label = feature_json['attack_class']
                                attack_current_label = attack_label if frame_index + self.sequence_length > interact_start_frame and frame_index < interact_end_frame else 3
                                attack_current_label = self.check_current_attack(attack_current_label, frame_index,
                                                                                 frames)
                                attack_future_label = attack_label if frame_index + self.sequence_length + self.frames_before_event > interact_start_frame and frame_index + self.sequence_length < interact_end_frame else 3
                                attack_future_label = self.check_future_attack(attack_future_label, frame_index,
                                                                               frames)
                                # action_label = feature_json[
                                #     'action_class'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 10
                                distance = torch.tensor(distance)
                                down_sample_count += 1
                                if attack_current_label == 3 and attack_future_label == 3 and down_sample_count % int(
                                        1 / self.not_interact_downsample_rate) != 0:
                                    continue
                                self.features.append([x_tensor])
                                self.distances.append(distance)
                                self.labels.append((attack_current_label, attack_future_label))
                                if self.train:
                                    if 'noise' in self.augment_method:
                                        self.add_gaussian_noise(x_tensor, distance, attack_current_label,
                                                                attack_future_label)
                                    if 'move' in self.augment_method:
                                        self.random_move(x_tensor, distance, attack_current_label,
                                                         attack_future_label)
                                    x_tensor[:, :, :2] = -x_tensor[:, :, :2]
                                    self.features.append([x_tensor])
                                    self.distances.append(distance)
                                    self.labels.append((attack_current_label, attack_future_label))

                else:
                    down_sample_count = 0
                    for frame_index in range(interact_start_frame - self.sequence_length - self.frames_before_event,
                                             interact_start_frame - self.sequence_length):
                        x_tensor = torch.zeros((6, self.sequence_length, harper_body_point_num, 3))
                        for camera in camera_name_list:
                            frame_width, frame_height = feature_json['cameras'][camera]['frame_size'][0], \
                                feature_json['cameras'][camera]['frame_size'][1]
                            frames = feature_json['cameras'][camera]['frames']
                            if frames:
                                distance = [-1 for _ in range(self.sequence_length)]
                                frame_num = 0
                                useful_num = 0
                                while frame_num < self.sequence_length:
                                    if frame_index + frame_num == frames[frame_index + useful_num]['frame_id']:
                                        frame_feature = np.array(frames[frame_index + useful_num]['keypoints'])
                                        frame_feature[:, 0] = 2 * (frame_feature[:, 0] / frame_width - 0.5)
                                        frame_feature[:, 1] = 2 * (frame_feature[:, 1] / frame_height - 0.5)
                                        x = torch.tensor(frame_feature)
                                        x_tensor[camera_name_list.index(camera), frame_num, :, :] = x
                                        distance.append(feature_json['min_distance'][frame_index + frame_num])
                                        useful_num += 1
                                    frame_num += 1
                                contact_label = feature_json[
                                    'contact_label'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 0
                                intent_label = feature_json[
                                    'intent_label'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 0
                                attitude_label = feature_json[
                                    'attitude_label'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 2
                                attack_label = feature_json[
                                    'attack_label'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 3
                                attack_label = self.check_attack(attack_label, frame_index, frames)
                                action_label = feature_json[
                                    'action_label'] if frame_index + self.sequence_length + self.frames_before_event < interact_start_frame else 10
                                distance = torch.tensor(distance)
                                self.features.append(x_tensor)
                                self.distances.append(distance)
                                self.labels.append(
                                    (contact_label, intent_label, attitude_label, attack_label, action_label))
                                down_sample_count += self.down_sample_rate

    def add_gaussian_noise(self, x_tensor, distance, attack_current_label, attack_future_label):
        sigma_list = [0.005, 0.01, 0.02]
        augment_times = 3
        for sigma_index, sigma in enumerate(sigma_list):
            for i in range(augment_times):
                x_gaussion_noise = np.random.normal(0, sigma, size=(self.sequence_length, harper_body_point_num, 1))
                y_gaussion_noise = np.random.normal(0, sigma, size=(self.sequence_length, harper_body_point_num, 1))
                score_gaussion_noise = np.zeros((self.sequence_length, harper_body_point_num, 1))
                gaussion_noise = np.hstack((x_gaussion_noise, y_gaussion_noise, score_gaussion_noise))
                keypoints = torch.Tensor((np.array(x_tensor) + gaussion_noise))
                self.features.append([keypoints])
                self.distances.append(distance)
                self.labels.append((attack_current_label, attack_future_label))
                keypoints[:, :, :2] = -keypoints[:, :, :2]
                self.features.append([keypoints])
                self.distances.append(distance)
                self.labels.append((attack_current_label, attack_future_label))

    def random_move(self, x_tensor, distance, attack_current_label, attack_future_label):
        augment_times = 10
        for _ in range(augment_times):
            x_move = torch.full((self.sequence_length, harper_body_point_num), (random.random() - 0.5) * 2)
            y_move = torch.full((self.sequence_length, harper_body_point_num), (random.random() - 0.5) * 2)
            keypoints = x_tensor.clone()
            keypoints[:, :, 0] = keypoints[:, :, 0] + x_move
            keypoints[:, :, 1] = keypoints[:, :, 1] + y_move
            self.features.append([keypoints])
            self.distances.append(distance)
            self.labels.append((attack_current_label, attack_future_label))
            keypoints[:, :, :2] = -keypoints[:, :, :2]
            self.features.append([keypoints])
            self.distances.append(distance)
            self.labels.append((attack_current_label, attack_future_label))


def get_harper_dataset(sequence_length, frames_before_event, augment_method, multi_angle=False):
    print('Loading data from HARPER dataset')
    data_path = '../HARPER/'
    train_files = os.listdir(data_path + 'train/pose_sequences/')
    val_files = os.listdir(data_path + 'validation/pose_sequences/')
    test_files = os.listdir(data_path + 'test/pose_sequences/')
    trainset = HARPER_Dataset(data_path=data_path + 'train/pose_sequences/', files=train_files,
                              sequence_length=sequence_length, frames_before_event=frames_before_event,
                              multi_angle=multi_angle, train=True, augment_method=augment_method)
    valset = HARPER_Dataset(data_path=data_path + 'validation/pose_sequences/', files=val_files,
                            sequence_length=sequence_length, frames_before_event=frames_before_event,
                            multi_angle=multi_angle)
    testset = HARPER_Dataset(data_path=data_path + 'test/pose_sequences/', files=test_files,
                             sequence_length=sequence_length, frames_before_event=frames_before_event,
                             multi_angle=multi_angle)
    print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (len(trainset), len(valset), len(testset)))
    return trainset, valset, testset


if __name__ == '__main__':
    trainset, valset, testset = get_harper_dataset(10, 10)
    attack_current_dict = {}
    attack_future_dict = {}
    for attack in attack_class:
        attack_current_dict[attack] = 0
        attack_future_dict[attack] = 0
    for dataset in [testset]:
        for data in dataset:
            attack_current_dict[attack_class[data[1][0]]] += 1
            attack_future_dict[attack_class[data[1][1]]] += 1
    print('current', attack_current_dict)
    print('future', attack_future_dict)
