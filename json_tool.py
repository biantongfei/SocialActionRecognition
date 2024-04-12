import json
import os
import cv2
import numpy as np


def summarize_features(feature_path):
    intention_class_list = ['interacting', 'interested', 'not_interested']
    attitude_class_list = ['positive', 'negative', 'no_interacting']
    action_class_list = ['hand_shake', 'hug', 'pet', 'wave', 'punch', 'throw', 'point', 'gaze', 'leave', 'no_response']
    intention_dict = {}
    attitude_dict = {}
    action_dict = {}
    for i in range(len(intention_class_list)):
        intention_dict[i] = {'count': 0, 'total_frames': 0}
    for i in range(len(attitude_class_list)):
        attitude_dict[i] = {'count': 0, 'total_frames': 0}
    for i in range(len(action_class_list)):
        action_dict[i] = {'count': 0, 'total_frames': 0}
    feature_list = os.listdir(feature_path)
    feature_list.sort()
    for feature in feature_list:
        # print(feature)
        if 'json' not in feature:
            continue
        with open(feature_path + feature, "r") as f:
            feature_json = json.load(f)
            f.close()
        # if feature_json['action_class']==0:
        #     print(feature)
        intention_dict[feature_json['intention_class']]['count'] += 1
        intention_dict[feature_json['intention_class']]['total_frames'] += feature_json['detected_frames_number']
        attitude_dict[feature_json['attitude_class']]['count'] += 1
        attitude_dict[feature_json['attitude_class']]['total_frames'] += feature_json['detected_frames_number']
        action_dict[feature_json['action_class']]['count'] += 1
        action_dict[feature_json['action_class']]['total_frames'] += feature_json['detected_frames_number']

    for c in intention_dict.keys():
        print('intention: %s: {count:%d, avg_frames:%s}' % (
            intention_class_list[c], intention_dict[c]['count'], '{:.2f}'.format(
                intention_dict[c]['total_frames'] / intention_dict[c]['count'])))
        # print('intention: %s: {count:%d}' % (intention_class_list[c], intention_dict[c]['count']))
    for c in attitude_dict.keys():
        print('attitude: %s: {count:%d, avg_frames:%s}' % (
            attitude_class_list[c], attitude_dict[c]['count'], '{:.2f}'.format(
                attitude_dict[c]['total_frames'] / attitude_dict[c]['count'])))
        # print('attitude: %s: {count:%d}' % (attitude_class_list[c], attitude_dict[c]['count']))
    for c in action_dict.keys():
        print('action: %s: {count:%d, avg_frames:%s}' % (action_class_list[c], action_dict[c]['count'], '{:.2f}'.format(
            action_dict[c]['total_frames'] / action_dict[c]['count'])))
        # print('action: %s: {count:%d}' % (action_class_list[c], action_dict[c]['count']))


def refactor_jsons():
    video_path = '../JPL_Augmented_Posefeatures/more_video/'
    json_path = '../JPL_Augmented_Posefeatures/more_feature/crop/coco_wholebody/'
    jsons = os.listdir(json_path)
    jsons.sort()
    for file in jsons:
        print(file, 'coco')
        if file == '.DS_Store':
            continue
        with open(json_path + file, "r") as f:
            ori_json = json.load(f)
        video_name = ori_json['video_name']
        cap = cv2.VideoCapture(video_path + file.split('.')[0] + '.avi')
        frame_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for person_id in ori_json['persons'].keys():
            if len(ori_json['persons'][person_id]['frames']) < 10 and len(
                    ori_json['persons'][person_id]['frames']) < frames_number / 10:
                continue
            new_json = {'video_name': video_name, 'frame_size': frame_size, 'video_frames_number': frames_number,
                        'detected_frames_number': len(ori_json['persons'][person_id]['frames']),
                        'person_id': int(person_id),
                        'intention_class': ori_json['persons'][person_id]['intention_class'],
                        'attitude_class': ori_json['persons'][person_id]['attitude_class'],
                        'action_class': ori_json['persons'][person_id]['action_class'],
                        'frames': []}
            for frame in ori_json['persons'][person_id]['frames']:
                keypoints = []
                for index, keypoint in enumerate(frame['keypoints']):
                    if index % 3 == 0:
                        keypoints.append(frame['keypoints'][index:index + 3])
                    else:
                        continue
                new_json['frames'].append({'frame_id': int(frame['image_id'].split('.')[0]), 'keypoints': keypoints,
                                           'score': frame['score'], 'box': frame['box']})
            with open('%s_p%s.json' % (json_path + file.split('.')[0], person_id), "w") as outfile:
                json.dump(new_json, outfile)
        os.system('rm -rf %s' % (json_path + file))

    json_path = '../JPL_Augmented_Posefeatures/more_feature/crop/halpe136/'
    jsons = os.listdir(json_path)
    jsons.sort()
    for file in jsons:
        print(file, 'halpe')
        if file == '.DS_Store':
            continue
        with open(json_path + file, "r") as f:
            ori_json = json.load(f)
        video_name = ori_json['video_name']
        cap = cv2.VideoCapture(video_path + file.split('.')[0] + '.avi')
        frame_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for person_id in ori_json['persons'].keys():
            if len(ori_json['persons'][person_id]['frames']) < 10 and len(
                    ori_json['persons'][person_id]['frames']) < frames_number / 10:
                print(video_name, person_id, len(ori_json['persons'][person_id]['frames']),
                      ori_json['persons'][person_id]['action_class'])
                continue
            new_json = {'video_name': video_name, 'frame_size': frame_size, 'video_frames_number': frames_number,
                        'detected_frames_number': len(ori_json['persons'][person_id]['frames']),
                        'person_id': int(person_id),
                        'intention_class': ori_json['persons'][person_id]['intention_class'],
                        'attitude_class': ori_json['persons'][person_id]['attitude_class'],
                        'action_class': ori_json['persons'][person_id]['action_class'],
                        'frames': []}
            for frame in ori_json['persons'][person_id]['frames']:
                keypoints = []
                for index, keypoint in enumerate(frame['keypoints']):
                    if index % 3 == 0:
                        keypoints.append(frame['keypoints'][index:index + 3])
                    else:
                        continue
                new_json['frames'].append({'frame_id': int(frame['image_id'].split('.')[0]), 'keypoints': keypoints,
                                           'score': frame['score'], 'box': frame['box']})
            with open('%s_p%s.json' % (json_path + file.split('.')[0], person_id), "w") as outfile:
                json.dump(new_json, outfile)
        os.system('rm -rf %s' % (json_path + file))


def flip_feature(ori_json):
    width, height = ori_json['frame_size'][0], ori_json['frame_size'][1]
    new_json = {'video_name': ori_json['video_name'], 'frame_size': ori_json['frame_size'],
                'video_frames_number': ori_json['video_frames_number'],
                'detected_frames_number': ori_json['detected_frames_number'], 'person_id': ori_json['person_id'],
                'intention_class': ori_json['intention_class'], 'attitude_class': ori_json['attitude_class'],
                'action_class': ori_json['action_class'], 'frames': []}
    for frame in ori_json['frames']:
        keypoints = []
        for keypoint in frame['keypoints']:
            keypoints.append([width - keypoint[0], keypoint[1], keypoint[2]])
        new_json['frames'].append({'frame_id': frame['frame_id'], 'keypoints': keypoints, 'score': frame['score'],
                                   'box': [width - frame['box'][0] - frame['box'][2], frame['box'][1],
                                           frame['box'][2], frame['box'][3]]})
    return new_json


def gaussion_augment():
    sigma_list = [0.01, 0.005, 0.001]
    sigma_str = ['10', '05', '01']
    augment_times = 3

    json_path = '../JPL_Augmented_Posefeatures/more_feature/crop/coco_wholebody/'
    out_path = '../JPL_Augmented_Posefeatures/more_feature/gaussian/coco_wholebody/'
    files = os.listdir(json_path)
    files.sort()
    for file in files:
        if '-ori_' in file:
            print(file, 'coco')
            with open(json_path + file, "r") as f:
                ori_json = json.load(f)
            with open(out_path + file, "w") as outfile:
                json.dump(ori_json, outfile)
            new_json = flip_feature(ori_json)
            with open(out_path + '%s-flip_p%s' % (file.split('_p')[0], file.split('_p')[-1]), "w") as outfile:
                json.dump(new_json, outfile)
            for sigma_index, sigma in enumerate(sigma_list):
                for i in range(augment_times):
                    new_json = {'video_name': ori_json['video_name'], 'frame_size': ori_json['frame_size'],
                                'video_frames_number': ori_json['video_frames_number'],
                                'detected_frames_number': ori_json['detected_frames_number'],
                                'person_id': ori_json['person_id'], 'intention_class': ori_json['intention_class'],
                                'attitude_class': ori_json['attitude_class'], 'action_class': ori_json['action_class'],
                                'frames': []}
                    for frame in ori_json['frames']:
                        box_size = frame['box'][2:]
                    keypoints = frame['keypoints']
                    x_gaussion_noise = np.random.normal(0, box_size[0] * sigma, size=(len(keypoints), 1))
                    y_gaussion_noise = np.random.normal(0, box_size[1] * sigma, size=(len(keypoints), 1))
                    score_gaussion_noise = np.zeros((len(keypoints), 1))
                    gaussion_noise = np.hstack((x_gaussion_noise, y_gaussion_noise, score_gaussion_noise))
                    keypoints = (np.array(keypoints) + gaussion_noise).tolist()
                    new_json['frames'].append(
                        {'frame_id': frame['frame_id'], 'keypoints': keypoints, 'score': frame['score'],
                         'box': frame['box']})
                    with open(out_path + '%s-noise%s-%d_p%s' % (
                            file.split('-')[0], sigma_str[sigma_index], i, file.split('_p')[-1]),
                              "w") as outfile:
                        json.dump(new_json, outfile)
                    new_json = flip_feature(new_json)
                    with open(out_path + '%s-noise%s-%d-flip_p%s' % (
                            file.split('-')[0], sigma_str[sigma_index], i, file.split('_p')[-1]),
                              "w") as outfile:
                        json.dump(new_json, outfile)

    json_path = '../JPL_Augmented_Posefeatures/more_feature/crop/halpe136/'
    out_path = '../JPL_Augmented_Posefeatures/more_feature/gaussian/halpe136/'
    files = os.listdir(json_path)
    files.sort()
    for file in files:
        if'-ori_' in file:
            print(file, 'halpe')
            with open(json_path + file, "r") as f:
                ori_json = json.load(f)
            with open(out_path + file, "w") as outfile:
                json.dump(ori_json, outfile)
            new_json = flip_feature(ori_json)
            with open(out_path + '%s-flip_p%s' % (file.split('_p')[0], file.split('_p')[-1]), "w") as outfile:
                json.dump(new_json, outfile)
            for sigma_index, sigma in enumerate(sigma_list):
                for i in range(augment_times):
                    new_json = {'video_name': ori_json['video_name'], 'frame_size': ori_json['frame_size'],
                                'video_frames_number': ori_json['video_frames_number'],
                                'detected_frames_number': ori_json['detected_frames_number'],
                                'person_id': ori_json['person_id'], 'intention_class': ori_json['intention_class'],
                                'attitude_class': ori_json['attitude_class'], 'action_class': ori_json['action_class'],
                                'frames': []}
                    for frame in ori_json['frames']:
                        box_size = frame['box'][2:]
                    keypoints = frame['keypoints']
                    x_gaussion_noise = np.random.normal(0, box_size[0] * sigma, size=(len(keypoints), 1))
                    y_gaussion_noise = np.random.normal(0, box_size[1] * sigma, size=(len(keypoints), 1))
                    score_gaussion_noise = np.zeros((len(keypoints), 1))
                    gaussion_noise = np.hstack((x_gaussion_noise, y_gaussion_noise, score_gaussion_noise))
                    keypoints = (np.array(keypoints) + gaussion_noise).tolist()
                    new_json['frames'].append(
                        {'frame_id': frame['frame_id'], 'keypoints': keypoints, 'score': frame['score'],
                         'box': frame['box']})
                    with open(out_path + '%s-noise%s-%d_p%s' % (
                            file.split('-')[0], sigma_str[sigma_index], i, file.split('_p')[-1]),
                              "w") as outfile:
                        json.dump(new_json, outfile)
                    new_json = flip_feature(new_json)
                    with open(out_path + '%s-noise%s-%d-flip_p%s' % (
                            file.split('-')[0], sigma_str[sigma_index], i, file.split('_p')[-1]),
                              "w") as outfile:
                        json.dump(new_json, outfile)


def mixed_augment():
    sigma_list = [0.01, 0.005]
    sigma_str = ['10', '05']
    augment_times = 2

    crop_path = '../JPL_Augmented_Posefeatures/more_feature/crop/coco_wholebody/'
    out_path = '../JPL_Augmented_Posefeatures/more_feature/mixed/coco_wholebody/'
    files = os.listdir(crop_path)
    files.sort()
    for file in files:
        print(file, 'coco')
        if 'json' not in file:
            continue
        with open(crop_path + file, "r") as f:
            ori_json = json.load(f)
            f.close()
        with open(out_path + file, "w") as outfile:
            json.dump(ori_json, outfile)
            outfile.close()
        for sigma_index, sigma in enumerate(sigma_list):
            for i in range(augment_times):
                new_json = {'video_name': ori_json['video_name'], 'frame_size': ori_json['frame_size'],
                                'video_frames_number': ori_json['video_frames_number'],
                                'detected_frames_number': ori_json['detected_frames_number'],
                                'person_id': ori_json['person_id'], 'intention_class': ori_json['intention_class'],
                                'attitude_class': ori_json['attitude_class'], 'action_class': ori_json['action_class'],
                                'frames': []}
                for frame in ori_json['frames']:
                    box_size = frame['box'][2:]
                    keypoints = frame['keypoints']
                    x_gaussion_noise = np.random.normal(0, box_size[0] * sigma, size=(len(keypoints), 1))
                    y_gaussion_noise = np.random.normal(0, box_size[1] * sigma, size=(len(keypoints), 1))
                    score_gaussion_noise = np.zeros((len(keypoints), 1))
                    gaussion_noise = np.hstack((x_gaussion_noise, y_gaussion_noise, score_gaussion_noise))
                    keypoints = (np.array(keypoints) + gaussion_noise).tolist()
                    new_json['frames'].append(
                        {'frame_id': frame['frame_id'], 'keypoints': keypoints, 'score': frame['score'],
                         'box': frame['box']})
                with open(out_path + '%s-noise%s-%d_p%s' % (
                        file.split('_p')[0], sigma_str[sigma_index], i, file.split('_p')[-1]),
                          "w") as outfile:
                    json.dump(new_json, outfile)
                    outfile.close()

    crop_path = '../JPL_Augmented_Posefeatures/more_feature/crop/halpe136/'
    out_path = '../JPL_Augmented_Posefeatures/more_feature/mixed/halpe136/'
    files = os.listdir(crop_path)
    files.sort()
    for file in files:
        if 'json' not in file:
            continue
        print(file, 'halpe')
        with open(crop_path + file, "r") as f:
            ori_json = json.load(f)
            f.close()
        with open(out_path + file, "w") as outfile:
            json.dump(ori_json, outfile)
            outfile.close()
        for sigma_index, sigma in enumerate(sigma_list):
            for i in range(augment_times):
                new_json = {'video_name': ori_json['video_name'], 'frame_size': ori_json['frame_size'],
                            'video_frames_number': ori_json['video_frames_number'],
                            'detected_frames_number': ori_json['detected_frames_number'],
                            'person_id': ori_json['person_id'], 'intention_class': ori_json['intention_class'],
                            'attitude_class': ori_json['attitude_class'], 'action_class': ori_json['action_class'],
                            'frames': []}
                for frame in ori_json['frames']:
                    box_size = frame['box'][2:]
                    keypoints = frame['keypoints']
                    x_gaussion_noise = np.random.normal(0, box_size[0] * sigma, size=(len(keypoints), 1))
                    y_gaussion_noise = np.random.normal(0, box_size[1] * sigma, size=(len(keypoints), 1))
                    score_gaussion_noise = np.zeros((len(keypoints), 1))
                    gaussion_noise = np.hstack((x_gaussion_noise, y_gaussion_noise, score_gaussion_noise))
                    keypoints = (np.array(keypoints) + gaussion_noise).tolist()
                    new_json['frames'].append(
                        {'frame_id': frame['frame_id'], 'keypoints': keypoints, 'score': frame['score'],
                         'box': frame['box']})
                with open(out_path + '%s-noise%s-%d_p%s' % (
                        file.split('_p')[0], sigma_str[sigma_index], i, file.split('_p')[-1]),
                          "w") as outfile:
                    json.dump(new_json, outfile)
                    outfile.close()


def add_attitude_class():
    json_path = '../JPL_Augmented_Posefeatures/more_feature/crop/coco_wholebody/'
    files = os.listdir(json_path)
    for file in files:
        if 'json' not in file:
            continue
        with open(json_path + file, "r") as f:
            old_json = json.load(f)
            f.close()
        if old_json['action_class'] in [4, 5]:
            new_json = {'video_name': old_json['video_name'], 'frame_size': old_json['frame_size'],
                        'video_frames_number': old_json['video_frames_number'],
                        'detected_frames_number': old_json['detected_frames_number'],
                        'person_id': old_json['person_id'], 'intention_class': 0,
                        'attitude_class': 1, 'action_class': old_json['action_class'],
                        'frames': old_json['frames']}
            with open(json_path + file, 'w') as f:
                json.dump(new_json, f)
                f.close()
    json_path = '../JPL_Augmented_Posefeatures/more_feature/crop/halpe136/'
    files = os.listdir(json_path)
    for file in files:
        if 'json' not in file:
            continue
        with open(json_path + file, "r") as f:
            old_json = json.load(f)
            f.close()
        if old_json['action_class'] in [4, 5]:
            new_json = {'video_name': old_json['video_name'], 'frame_size': old_json['frame_size'],
                        'video_frames_number': old_json['video_frames_number'],
                        'detected_frames_number': old_json['detected_frames_number'],
                        'person_id': old_json['person_id'], 'intention_class': 0,
                        'attitude_class': 1, 'action_class': old_json['action_class'],
                        'frames': old_json['frames']}
            with open(json_path + file, 'w') as f:
                json.dump(new_json, f)
                f.close()


if __name__ == '__main__':
    # add_attitude_class()
    # refactor_jsons()
    # feature_path = '../JPL_Augmented_Posefeatures/crop/coco_wholebody/'
    # summarize_features(feature_path)
    # gaussion_augment()
    # files = os.listdir(feature_path)
    # files.sort()
    # pre_video = ''
    # for file in files:
    #     if file.split('p')[0] == pre_video:
    #         with open(feature_path + file, 'r') as f:
    #             feature_json = json.load(f)
    #         if feature_json['action_class'] == 0:
    #             print(file)
    #     pre_video = file.split('p')[0]
    mixed_augment()
    # add_attitude_class()
