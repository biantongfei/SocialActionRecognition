import os
import json
import random

import numpy as np
from torch.utils.data import Dataset

testset_rate = 0.5
coco_point_num = 133
halpe_point_num = 136
fps = 30


def get_data_path(is_crop, is_coco):
    if is_crop:
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/crop/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/crop/halpe136/'
    else:
        if is_coco:
            data_path = '../JPL_Augmented_Posefeatures/gaussian/coco_wholebody/'
        else:
            data_path = '../JPL_Augmented_Posefeatures/gaussian/halpe136/'
    return data_path


def get_tra_test_files(is_crop, is_coco, not_add_class, ori_videos=False):
    data_path = get_data_path(is_crop, is_coco)
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
            if ori_videos and '-ori_' not in file:
                continue
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


class Dataset(Dataset):
    def __init__(self, data_files, action_recognition, is_crop, is_coco, body_part, video_len=99999, avg=False):
        super(Dataset, self).__init__()
        self.files = data_files
        self.data_path = get_data_path(is_crop=is_crop, is_coco=is_coco)
        self.action_recognition = action_recognition  # 0 for origin 7 classes; 1 for add not interested and interested; False for attitude recognition
        self.is_crop = is_crop
        self.is_coco = is_coco
        self.body_part = body_part  # 1 for only body, 2 for head and body, 3 for hands and body, 4 for head, hands and body
        self.video_len = video_len
        self.avg = avg

    def __getitem__(self, idx):
        with open(self.data_path + self.files[idx], 'r') as f:
            feature_json = json.load(f)
        features = []
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        frame_num = len(feature_json['frames'])
        last_frame_id = feature_json['frames'][0]['frame_id'] - 1
        index = 0
        while len(features) < int(self.video_len * fps):
            if index == frame_num:
                break
            frame = feature_json['frames'][index]
            if last_frame_id + 1 != frame['frame_id']:
                features.append(np.full((2 * len(frame['keypoints']) + 4), np.nan))
                last_frame_id += 1
            else:
                box_x, box_y, box_width, box_height = frame['box'][0], frame['box'][1], frame['box'][2], frame['box'][3]
                frame_feature = np.array(frame['keypoints'])[:, :2]
                frame_feature[:, 0] = (frame_feature[:, 0] - box_x) / box_width
                frame_feature[:, 1] = (frame_feature[:, 1] - box_y) / box_height
                # frame_feature[:, 0] = frame_feature[:, 0] / frame_width - 0.5
                # frame_feature[:, 1] = frame_feature[:, 1] / frame_height - 0.5
                # frame_feature = get_body_part(frame_feature, self.is_coco, self.body_part)
                frame_feature = np.append(frame_feature, [
                    [(box_x - (frame_width / 2)) / frame_width, (box_y - (frame_height / 2)) / frame_height],
                    [box_width / frame_width, box_height / frame_height]], axis=0)
                frame_feature = frame_feature.reshape(1, frame_feature.size)[0]
                features.append(frame_feature)
                index += 1
        features = np.array(features)
        if self.action_recognition:
            label = feature_json['action_class']
        else:
            if feature_json['action_class'] == 7:
                label = 1
            elif feature_json['action_class'] == 8:
                label = 2
            else:
                label = 0
        if self.avg:
            features = np.nanmean(features, axis=0)
        return features, label

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    is_crop = True
    is_coco = True
    tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco, not_add_class=False)
    dataset = Dataset(data_files=tra_files, action_recognition=False, is_crop=is_crop, is_coco=is_coco,
                      body_part=[True, True, True], avg=True, video_len=2)
    features, labels = dataset.__getitem__(9)
    print(features.shape, labels)