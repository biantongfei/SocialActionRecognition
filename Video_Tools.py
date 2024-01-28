import os
import json
import random

import cv2
import xlrd


def read_jpl_labels():
    xlsx_path = "jpl_interaction_labels.xls"

    # Open the Workbook
    workbook = xlrd.open_workbook(xlsx_path)

    # Open the worksheet
    worksheet = workbook.sheet_by_index(0)
    labels = {}
    for row in worksheet:
        if row[0].value == "file_name":
            continue
        else:
            labels[row[0].value] = row[2].value
    return labels


def video_augmentation(video_dir, out_path):
    crop_rate_list = [0.95, 0.85, 0.75]
    crop_times_per_rate = 3
    videos = os.listdir(video_dir)
    videos.sort()

    for video_name in videos:
        if '.avi' not in video_name:
            continue
        print(video_name)
        # read video
        cap = cv2.VideoCapture(video_dir + video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        success, frame = cap.read()
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # randomly select crop positions
        new_frames = []
        new_positions = []
        for crop_rate in crop_rate_list:
            for i in range(crop_times_per_rate):
                x = random.randint(0, int((1 - crop_rate) * width))
                y = random.randint(0, int((1 - crop_rate) * height))
                new_positions.append([x, y])

        while success:
            augmented_frames = []
            for i, position in enumerate(new_positions):
                crop_rate = crop_rate_list[int(i / crop_times_per_rate)]
                new_frame = frame[position[1]:position[1] + int(crop_rate * height),
                            position[0]:position[0] + int(crop_rate * width)]

                # add cropped frame
                augmented_frames.append(new_frame)

                # add flipped frame
                new_frame = cv2.flip(new_frame, 1)
                augmented_frames.append(new_frame)

            if new_frames:
                for index in range(len(augmented_frames)):
                    new_frames[index].append(augmented_frames[index])
                new_frames[-2].append(frame)
                frame = cv2.flip(frame, 1)
                new_frames[-1].append(frame)
            else:
                for index in range(len(augmented_frames)):
                    new_frames.append([augmented_frames[index]])
                new_frames.append([frame])
                frame = cv2.flip(frame, 1)
                new_frames.append([frame])
            success, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()

        # write videos
        for index, frames in enumerate(new_frames[:-2]):
            crop_rate = crop_rate_list[int(index / (crop_times_per_rate * 2))]
            count = int((index % (crop_times_per_rate * 2)) / 2)
            is_flip = index % 2
            write_video(frames, video_name, out_path, fps=fps, width=int(width * crop_rate),
                        height=int(height * crop_rate), is_original=False, crop_rate=crop_rate, count=count,
                        is_flip=is_flip)
        write_video(new_frames[-1], video_name, out_path, fps=fps, width=width, height=height, is_original=True)
        write_video(new_frames[-2], video_name, out_path, fps=fps, width=width, height=height, is_original=True,
                    is_flip=True)


def write_video(frames, video_name, out_path, fps, width, height, is_original=True, crop_rate=None, count=None,
                is_flip=None):
    if is_original:
        if is_flip:
            name = video_name.split('.')[0] + '-ori-flip.avi'
        else:
            name = video_name.split('.')[0] + '-ori.avi'
    else:
        name = video_name.split('.')[0] + '-resize%d-%d%s.avi' % (crop_rate * 100, count, '-flip' if is_flip else '')
    print(name)
    output = cv2.VideoWriter(out_path + name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                             (width, height))
    for frame in frames:
        output.write(frame)
    output.release()
    cv2.destroyAllWindows()


def label_person_id(now_video, coco, fps):
    video_dir = '../JPL_Augmented_Posefeatures/more_video_augment/'
    videos = os.listdir(video_dir)
    videos.sort()
    video_length = len(videos)

    for index, video in enumerate(videos):
        if video == now_video:
            videos = videos[index:]
            break

    for video in videos:
        if video == '.DS_Store':
            continue
        if coco:
            print('----------------------------------------------')
            print(video, "{:.2f}%".format(100 * 2 * index / (2 * video_length)), 'coco')
            label_video(video, coco=True, fps=fps)
        print('----------------------------------------------')
        print(video, "{:.2f}%".format(100 * (2 * index + 1) / (2 * video_length)), 'halpe')
        label_video(video, coco=False, fps=fps)
        index += 1
        coco = True


def label_video(video_name, coco, fps):
    if coco:
        outpath = '../JPL_Augmented_Posefeatures/more_video_features/crop/coco_wholebody/'
        ori_json_path = '../JPL_Augmented_Posefeatures/more_video_features/crop/Alphapose_coco_wholebody/' + \
                        video_name.split('.')[0] + '.json'
    else:
        outpath = '../JPL_Augmented_Posefeatures/more_video_features/crop/halpe136/'
        ori_json_path = '../JPL_Augmented_Posefeatures/more_video_features/crop/Alphapose_halpe136/' + \
                        video_name.split('.')[0] + '.json'

    # rebuild json structure
    with open(ori_json_path, "r") as f:
        ori_json = json.load(f)
    persons = {}
    for pose in ori_json:
        if pose['idx'] in persons.keys():
            persons[int(pose['idx'])].append(
                {'image_id': pose['image_id'], 'keypoints': pose['keypoints'], 'score': pose['score'],
                 'box': pose['box']})
        else:
            persons[int(pose['idx'])] = [
                {'image_id': pose['image_id'], 'keypoints': pose['keypoints'], 'score': pose['score'],
                 'box': pose['box']}]

    # label for person
    # labels = read_jpl_labels()
    labels = 8
    new_json = {'video_name': video_name, 'persons': {}}
    now_id = 1
    for person_id in persons.keys():
        boxes = {}
        for frame in persons[person_id]:
            boxes[int(frame['image_id'].split('.')[0])] = frame['box']
        while True:
            play_video_box(video_name, boxes=boxes, fps=fps)
            answer = input('please type your answer for person_id %d: ' % (person_id))

            # again for replay
            if answer == 'a':
                continue

            # no for not a person
            elif answer == 'n':
                print('not a person!!!')
                break
            elif len(answer) == 2:
                # yes for a new person
                if answer[0] == 'y':
                    # 1 for ori class, 2 for not interested, 3 for interested
                    if int(answer[1]) == 1:
                        # action_class = int(labels[video_name.split('-')[0]])
                        action_class = 8
                    elif int(answer[1]) == 2:
                        action_class = 7
                    elif int(answer[1]) == 3:
                        action_class = 8
                    else:
                        print('answer again')
                        continue
                    new_json['persons'][now_id] = {'action_class': action_class, 'frames': persons[person_id]}
                    print('action_class: %s, frames_num: %s' % (action_class, len(persons[person_id])))
                    now_id += 1
                    break

                # merge for merge frames for same person
                elif answer[0] == 'm':
                    index1 = 0  # index for existed one
                    index2 = 0  # index for new one
                    frames = []
                    while index1 < len(new_json['persons'][int(answer[1])]['frames']) or index2 < len(
                            persons[person_id]):
                        # print(index1, new_json['persons'][int(answer[1])]['frames'][index1]['image_id'], index2,
                        #       persons[person_id][index2]['image_id'])
                        if index1 >= len(new_json['persons'][int(answer[1])]['frames']):
                            frames.append(persons[person_id][index2])
                            index2 += 1
                        elif index2 >= len(persons[person_id]):
                            frames.append(new_json['persons'][int(answer[1])]['frames'][index1])
                            index1 += 1
                        elif int(new_json['persons'][int(answer[1])]['frames'][index1]['image_id'].split('.')[0]) < int(
                                persons[person_id][index2]['image_id'].split('.')[0]):
                            frames.append(new_json['persons'][int(answer[1])]['frames'][index1])
                            index1 += 1
                        elif int(new_json['persons'][int(answer[1])]['frames'][index1]['image_id'].split('.')[0]) > int(
                                persons[person_id][index2]['image_id'].split('.')[0]):
                            frames.append(persons[person_id][index2])
                            index2 += 1
                        else:
                            if new_json['persons'][int(answer[1])]['frames'][index1]['score'] > \
                                    persons[person_id][index2]['score']:
                                frames.append(new_json['persons'][int(answer[1])]['frames'][index1])
                            else:
                                frames.append(persons[person_id][index2])
                            index1 += 1
                            index2 += 1
                    print(len(frames))
                    new_json['persons'][int(answer[1])]['frames'] = frames
                    break
                else:
                    print('answer again')
            else:
                print('answer again')

    # write json
    with open(outpath + video_name.split('.')[0] + '.json', "w") as outfile:
        json.dump(new_json, outfile)


def play_video_box(video_name, boxes, fps):
    video_dir = '../JPL_Augmented_Posefeatures/more_video_augment/'
    cap = cv2.VideoCapture(video_dir + video_name)
    success, frame = cap.read()
    index = 0
    while success:
        frame = cv2.resize(frame,
                           (2 * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 2 * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if index in boxes.keys():
            bbox = boxes[index]
            cv2.rectangle(frame, (2 * int(bbox[0]), 2 * int(bbox[1])),
                          (2 * int(bbox[0] + bbox[2]), 2 * int(bbox[1] + bbox[3])), (0, 225, 0), 6)
        cv2.imshow(video_name, frame)
        cv2.waitKey(fps)
        success, frame = cap.read()
        index += 1
    cap.release()
    cv2.destroyWindow(video_name)


def refactor_jsons():
    coco_path = '../JPL_Augmented_Posefeatures/more_video_features/crop/Alphapose_coco_wholebody/'
    files = os.listdir(coco_path)
    for file in files:
        print(file)
        if len(file.split('.')) < 2:
            # if file.split('.')[-1]=='avi':
            print(file, 'coco')
            os.system('mv %s %s' % (coco_path + file + '/alphapose-results.json', coco_path + file + '.json'))
            os.system('rm -rf %s' % (coco_path + file))
        else:
            continue

    halpe_path = '../JPL_Augmented_Posefeatures/more_video_features/crop/Alphapose_halpe136/'
    files = os.listdir(halpe_path)
    for file in files:
        print(file)
        if len(file.split('.')) < 2:
            # if file.split('.')[-1] == 'avi':
            print(file, 'halpe')
            os.system('mv %s %s' % (halpe_path + file + '/alphapose-results.json', halpe_path + file + '.json'))
            os.system('rm -rf %s' % (halpe_path + file))
        else:
            continue


if __name__ == "__main__":
    # for i in range(178, 224):
    #     print(i)
    #     del_json_content('11_5', 1, i, False)
    #     del_json_content('11_5', 2, i, False)
    # del_json_content('12_2', 1, 245, False)
    # select_user_from_jpl()
    # video_augmentation('../JPL_Augmented_Posefeatures/more_video/', '../JPL_Augmented_Posefeatures/more_video_augment/')
    label_person_id('c5-resize95-1.avi', True, 1)
    # refactor_jsons()

    # video_files = os.listdir('jpl_augmented/videos/')
    # video_files.sort()
    # coco_files = os.listdir('Alphapose_coco_wholebody/')
    # coco_files = os.listdir('jpl_augmented/features/crop/coco_wholebody/')
    # coco_files.sort()
    # halpe_files = os.listdir('Alphapose_halpe136/')
    # halpe_files = os.listdir('jpl_augmented/features/crop/halpe136/')
    # halpe_files.sort()
    # flag = False
    # coco_files = coco_files[1:]
    # halpe_files = halpe_files[1:]
    # for index in range(len(video_files)):
    #     print(index)
    #     if video_files[index].split('.')[0] != coco_files[index].split('.')[0]:
    #         print(video_files[index].split('.')[0], coco_files[index].split('.')[0], 'coco')
    #         flag = True
    #     if video_files[index].split('.')[0] != halpe_files[index].split('.')[0]:
    #         print(video_files[index].split('.')[0], halpe_files[index].split('.')[0], 'halpe')
    #         flag = True
    #     if flag:
    #         break
