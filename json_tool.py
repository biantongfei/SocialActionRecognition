import json
import os
import cv2
import numpy as np


def summarize_features(feature_path):
    action_class_list = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw', 'not_interested',
                         'interested', 'total']
    summaize_dict = {}
    for i in range(len(action_class_list)):
        summaize_dict[i] = {'count': 0, 'total_frames': 0}
    feature_list = os.listdir(feature_path)
    feature_list.sort()
    for feature in feature_list:
        print(feature)
        if 'json' not in feature or 'ori_' not in feature:
            continue
        with open(feature_path + feature, "r") as f:
            feature_json = json.load(f)
        summaize_dict[feature_json['action_class']]['count'] += 1
        summaize_dict[feature_json['action_class']]['total_frames'] += feature_json['frames_number']
        summaize_dict[len(action_class_list) - 1]['count'] += 1
        summaize_dict[len(action_class_list) - 1]['total_frames'] += feature_json['frames_number']

    for action_class in summaize_dict.keys():
        print('%s:{count:%d, avg_frames:%s}' % (action_class_list[action_class], summaize_dict[action_class]['count'],
                                                '{:.2f}'.format(summaize_dict[action_class]['total_frames'] /
                                                                summaize_dict[action_class]['count'])))


def refactor_jsons():
    video_path = 'videos/'
    json_path = 'features/crop/coco_wholebody/'
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
            if ori_json['persons'][person_id]['action_class'] not in [7, 8] and len(
                    ori_json['persons'][person_id]['frames']) < 10 and len(
                ori_json['persons'][person_id]['frames']) < frames_number / 10:
                continue
            new_json = {'video_name': video_name, 'frame_size': frame_size, 'frames_number': frames_number,
                        'person_id': int(person_id), 'action_class': ori_json['persons'][person_id]['action_class'],
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

    json_path = 'features/crop/halpe136/'
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
            if ori_json['persons'][person_id]['action_class'] not in [7, 8] and len(
                    ori_json['persons'][person_id]['frames']) < 10 and len(
                ori_json['persons'][person_id]['frames']) < frames_number / 10:
                print(video_name, person_id, len(ori_json['persons'][person_id]['frames']),
                      ori_json['persons'][person_id]['action_class'])
                continue
            new_json = {'video_name': video_name, 'frame_size': frame_size, 'frames_number': frames_number,
                        'person_id': int(person_id), 'action_class': ori_json['persons'][person_id]['action_class'],
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
                'frames_number': ori_json['frames_number'], 'person_id': ori_json['person_id'],
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
    sigma_list = [0.05, 0.01, 0.005]
    sigma_str = 'big'
    # sigma_list = [0.005, 0.001, 0.0005]
    augment_times = 3

    json_path = 'features/crop/coco_wholebody/'
    out_path = 'features/gaussian/big/coco_wholebody/'
    files = os.listdir(json_path)
    files.sort()
    for file in files:
        if 'ori_' in file:
            print(file, 'coco')
            with open(json_path + file, "r") as f:
                ori_json = json.load(f)
            with open(out_path + file, "w") as outfile:
                json.dump(ori_json, outfile)
            new_json = flip_feature(ori_json)
            with open(out_path + file.split('.')[0].replace('-ori', '') + '-flip.json', "w") as outfile:
                json.dump(new_json, outfile)
            for sigma in sigma_list:
                for i in range(augment_times):
                    new_json = {'video_name': ori_json['video_name'], 'frame_size': ori_json['frame_size'],
                                'frames_number': ori_json['frames_number'], 'person_id': ori_json['person_id'],
                                'action_class': ori_json['action_class'], 'frames': []}
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
                    with open(out_path + '%s-noise%s-%d.json' % (
                            file.split('.')[0].replace('-ori', ''), str(sigma_list.index(sigma)), i),
                              "w") as outfile:
                        json.dump(new_json, outfile)
                    new_json = flip_feature(new_json)
                    with open(out_path + '%s-noise%s-%d-flip.json' % (
                            file.split('.')[0].replace('-ori', ''), str(sigma_list.index(sigma)), i),
                              "w") as outfile:
                        json.dump(new_json, outfile)

    json_path = 'features/crop/halpe136/'
    out_path = 'features/gaussian/big/halpe136/'
    files = os.listdir(json_path)
    files.sort()
    for file in files:
        if 'ori_' in file:
            print(file, 'halpe')
            with open(json_path + file, "r") as f:
                ori_json = json.load(f)
            with open(out_path + file, "w") as outfile:
                json.dump(ori_json, outfile)
            new_json = flip_feature(ori_json)
            with open(out_path + file.split('.')[0].replace('-ori', '') + '-flip.json', "w") as outfile:
                json.dump(new_json, outfile)
            for sigma in sigma_list:
                for i in range(augment_times):
                    new_json = {'video_name': ori_json['video_name'], 'frame_size': ori_json['frame_size'],
                                'frames_number': ori_json['frames_number'], 'person_id': ori_json['person_id'],
                                'action_class': ori_json['action_class'], 'frames': []}
                    for frame in ori_json['frames']:
                        box = frame['box'][2:]
                        keypoints = frame['keypoints']
                        x_gaussion_noise = np.random.normal(0, box[0] * sigma, size=(len(keypoints), 1))
                        y_gaussion_noise = np.random.normal(0, box[1] * sigma, size=(len(keypoints), 1))
                        score_gaussion_noise = np.zeros((len(keypoints), 1))
                        gaussion_noise = np.hstack((x_gaussion_noise, y_gaussion_noise, score_gaussion_noise))
                        keypoints = (np.array(keypoints) + gaussion_noise).tolist()
                        new_json['frames'].append(
                            {'frame_id': frame['frame_id'], 'keypoints': keypoints, 'score': frame['score'],
                             'box': frame['box']})
                    with open(out_path + '%s-noise%s-%d.json' % (
                            file.split('.')[0].replace('-ori', ''), sigma_str[sigma_list.index(sigma)], i),
                              "w") as outfile:
                        json.dump(new_json, outfile)
                    new_json = flip_feature(new_json)
                    with open(out_path + '%s-noise%s-%d-flip.json' % (
                            file.split('.')[0].replace('-ori', ''), sigma_str[sigma_list.index(sigma)], i),
                              "w") as outfile:
                        json.dump(new_json, outfile)


def adjust_box():
    data_path = '../jpl_augmented/features/'
    format_list = ['coco_wholebody/', 'halpe136/']
    sigma_list = ['small/', 'medium/', 'big/']
    for format in format_list:
        files = os.listdir(data_path + 'crop/' + format)
        for file in files:
            if 'json' not in file:
                continue
            print('crop/' + format + file)
            with open(data_path + 'crop/' + format + file, 'r') as f:
                feature_json = json.load(f)
            for frame in feature_json['frames']:
                frame['box'][0] += frame['box'][2] / 2
                frame['box'][1] += frame['box'][3] / 2
            with open(data_path + 'crop/' + format + file, 'w') as f:
                json.dump(feature_json, f)
    for sigma in sigma_list:
        for format in format_list:
            if 'json' not in file:
                continue
            files = os.listdir(data_path + 'gaussian/' + sigma + format)
            print('gaussian/' + sigma + format + file)
            for file in files:
                with open(data_path + 'gaussian/' + sigma + format + file, 'r') as f:
                    feature_json = json.load(f)
                for frame in feature_json['frames']:
                    frame['box'][0] += frame['box'][2] / 2
                    frame['box'][1] += frame['box'][3] / 2
                with open(data_path + 'gaussian/' + sigma + format + file, 'w') as f:
                    json.dump(feature_json, f)


if __name__ == '__main__':
    refactor_jsons()
    # feature_path = '../jpl_augmented/features/crop/coco_wholebody/'
    # feature_path = '../jpl_augmented/features/crop/halpe136/'
    # summarize_features(feature_path)
    # gaussion_augment()
    # adjust_box()
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