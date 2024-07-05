import os
import json
import random

import cv2
import xlrd
import matplotlib.pyplot as plt


def read_jpl_labels():
    xls_path = '../JPL_Augmented_Posefeatures/robot_interaction_labels.xls'
    workbook = xlrd.open_workbook(xls_path)

    # Open the worksheet
    worksheet = workbook.sheet_by_index(0)
    labels = {}
    count = 0
    for row in worksheet:
        i = 0
        while i < 5 and row[i * 3 + 2].ctype == 2:
            if '%stest%d' % (row[0].value, int(row[1].value)) not in labels.keys():
                labels['%stest%d' % (row[0].value, int(row[1].value))] = [
                    [int(row[2].value), int(row[3].value), int(row[4].value)]]
            else:
                labels['%stest%d' % (row[0].value, int(row[1].value))].append(
                    [int(row[i * 3 + 2].value), int(row[i * 3 + 3].value), int(row[i * 3 + 4].value)])
            i += 1
            count += 1
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


def label_person_id(now_video, fps):
    video_dir = '../JPL_Augmented_Posefeatures/more_video/'
    videos = os.listdir(video_dir)
    videos.sort()
    video_length = len(videos)

    for index, video in enumerate(videos):
        if video == now_video:
            videos = videos[index:]
            break

    for video in videos:
        print(video)
        if video == '.DS_Store':
            continue
        print('----------------------------------------------')
        print(video, "{:.2f}%".format(100 * (2 * index + 1) / (2 * video_length)))
        label_video(video, fps=fps)
        index += 1


def label_video(video_name, fps):
    coco_outpath = '../JPL_Augmented_Posefeatures/more_feature/crop/coco_wholebody/'
    coco_ori_json_path = '../JPL_Augmented_Posefeatures/more_feature/alphapose/coco_wholebody/' + video_name.split('.')[
        0] + '.json'
    halpe_outpath = '../JPL_Augmented_Posefeatures/more_feature/crop/halpe136/'
    halpe_ori_json_path = '../JPL_Augmented_Posefeatures/more_feature/alphapose/halpe136/' + video_name.split('.')[
        0] + '.json'

    # rebuild json structure
    with open(coco_ori_json_path, "r") as f:
        coco_ori_json = json.load(f)
        f.close()
    with open(halpe_ori_json_path, "r") as f:
        halpe_ori_json = json.load(f)
        f.close()
    coco_persons = {}
    halpe_person = {}
    for pose in coco_ori_json:
        if pose['idx'] in coco_persons.keys():
            coco_persons[int(pose['idx'])].append(
                {'image_id': pose['image_id'], 'keypoints': pose['keypoints'], 'score': pose['score'],
                 'box': pose['box']})
        else:
            coco_persons[int(pose['idx'])] = [
                {'image_id': pose['image_id'], 'keypoints': pose['keypoints'], 'score': pose['score'],
                 'box': pose['box']}]
    for pose in halpe_ori_json:
        if pose['idx'] in halpe_person.keys():
            halpe_person[int(pose['idx'])].append(
                {'image_id': pose['image_id'], 'keypoints': pose['keypoints'], 'score': pose['score'],
                 'box': pose['box']})
        else:
            halpe_person[int(pose['idx'])] = [
                {'image_id': pose['image_id'], 'keypoints': pose['keypoints'], 'score': pose['score'],
                 'box': pose['box']}]

    # label for person
    coco_new_json = {'video_name': video_name, 'persons': {}}
    halpe_new_json = {'video_name': video_name, 'persons': {}}
    now_id = 1
    for person_id in coco_persons.keys():
        boxes = {}
        for frame in coco_persons[person_id]:
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
                    # 1 for ori class, 2 for interested, 3 for not interested
                    if int(answer[1]) in [0, 1, 2, 3]:
                        # action_class = int(labels[video_name.split('-')[0]])
                        action_class = int(answer[1])
                        intention_class = 0
                        attitude_class = 0
                    elif int(answer[1]) in [4, 5]:
                        action_class = int(answer[1])
                        intention_class = 0
                        attitude_class = 1
                    elif int(answer[1]) in [6, 7]:
                        action_class = int(answer[1])
                        attitude_class = 2
                        intention_class = 1
                    elif int(answer[1]) in [8, 9]:
                        action_class = int(answer[1])
                        attitude_class = 2
                        intention_class = 2
                    else:
                        print('answer again')
                        continue
                    coco_new_json['persons'][now_id] = {'intention_class': intention_class,
                                                        'attitude_class': attitude_class,
                                                        'action_class': action_class, 'frames': coco_persons[person_id]}
                    halpe_new_json['persons'][now_id] = {'intention_class': intention_class,
                                                         'attitude_class': attitude_class,
                                                         'action_class': action_class,
                                                         'frames': halpe_person[person_id]}
                    print('intention_class: %s, attitude_class: %s, action_class: %s, frames_num: %s' % (
                        intention_class, attitude_class, action_class, len(coco_persons[person_id])))
                    now_id += 1
                    break

                # merge for merge frames for same person
                elif answer[0] == 'm':
                    index1 = 0  # index for existed one
                    index2 = 0  # index for new one
                    coco_frames = []
                    halpe_frames = []
                    while index1 < len(coco_new_json['persons'][int(answer[1])]['frames']) or index2 < len(
                            coco_persons[person_id]):
                        # print(index1, new_json['persons'][int(answer[1])]['frames'][index1]['image_id'], index2,
                        #       persons[person_id][index2]['image_id'])
                        if index1 >= len(coco_new_json['persons'][int(answer[1])]['frames']):
                            coco_frames.append(coco_persons[person_id][index2])
                            halpe_frames.append(halpe_person[person_id][index2])
                            index2 += 1
                        elif index2 >= len(coco_persons[person_id]):
                            coco_frames.append(coco_new_json['persons'][int(answer[1])]['frames'][index1])
                            halpe_frames.append(halpe_new_json['persons'][int(answer[1])]['frames'][index1])
                            index1 += 1
                        elif int(coco_new_json['persons'][int(answer[1])]['frames'][index1]['image_id'].split('.')[
                                     0]) < int(coco_persons[person_id][index2]['image_id'].split('.')[0]):
                            coco_frames.append(coco_new_json['persons'][int(answer[1])]['frames'][index1])
                            halpe_frames.append(halpe_new_json['persons'][int(answer[1])]['frames'][index1])
                            index1 += 1
                        elif int(coco_new_json['persons'][int(answer[1])]['frames'][index1]['image_id'].split('.')[
                                     0]) > int(
                            coco_persons[person_id][index2]['image_id'].split('.')[0]):
                            coco_frames.append(coco_persons[person_id][index2])
                            halpe_frames.append(halpe_person[person_id][index2])
                            index2 += 1
                        else:
                            if coco_new_json['persons'][int(answer[1])]['frames'][index1]['score'] > \
                                    coco_persons[person_id][index2]['score']:
                                coco_frames.append(coco_new_json['persons'][int(answer[1])]['frames'][index1])
                                halpe_frames.append(halpe_new_json['persons'][int(answer[1])]['frames'][index1])
                            else:
                                coco_frames.append(coco_persons[person_id][index2])
                                halpe_frames.append(halpe_person[person_id][index2])
                            index1 += 1
                            index2 += 1
                    print(len(coco_frames))
                    coco_new_json['persons'][int(answer[1])]['frames'] = coco_frames
                    halpe_new_json['persons'][int(answer[1])]['frames'] = halpe_frames
                    break
                else:
                    print('answer again')
            else:
                print('answer again')

    # write json
    with open(coco_outpath + video_name.split('.')[0] + '.json', "w") as outfile:
        json.dump(coco_new_json, outfile)
    with open(halpe_outpath + video_name.split('.')[0] + '.json', "w") as outfile:
        json.dump(halpe_new_json, outfile)


def play_video_box(video_name, boxes, fps):
    video_dir = '../JPL_Augmented_Posefeatures/more_video/'
    cap = cv2.VideoCapture(video_dir + video_name)
    success, frame = cap.read()
    index = 0
    while success:
        frame = cv2.resize(frame,
                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if index in boxes.keys():
            bbox = boxes[index]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 225, 0), 6)
        cv2.imshow(video_name, frame)
        cv2.waitKey(fps)
        success, frame = cap.read()
        index += 1
    cap.release()
    cv2.destroyWindow(video_name)


def refactor_jsons():
    coco_path = '../JPL_Augmented_Posefeatures/more_feature/alphapose/coco_wholebody/'
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

    halpe_path = '../JPL_Augmented_Posefeatures/more_feature/alphapose/halpe136/'
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


def edit_videos():
    video_path = '../JPL_Augmented_Posefeatures/more_video_ori/'
    out_video_path = '../JPL_Augmented_Posefeatures/more_video_ori2/'
    labels = read_jpl_labels()
    print(labels)
    videos = os.listdir(video_path)
    for video in videos:
        for index, label in enumerate(labels[video.split('.')[0]]):
            if label[0] not in [3, 4, 6]:
                print(video, index, label[0])
                cap = cv2.VideoCapture(video_path + video)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter('%s_%d.avi' % (out_video_path + video.split('.')[0], index), fourcc, fps,
                                      (frame_width, frame_height))
                current_frame = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        if label[1] <= current_frame <= label[2]:
                            out.write(frame)
                        current_frame += 1
                    else:
                        break
                out.release()
                cap.release()
                cv2.destroyAllWindows()


def draw_keypoints():
    with open('../JPL_Augmented_Posefeatures/mixed/coco_wholebody/1_1-resize75-0_p1.json', 'r') as f:
        json_file = json.load(f)
        f.close()
    frame_width, frame_height = json_file['frame_size'][0], json_file['frame_size'][1]
    print(frame_width, frame_height)
    for frame in json_file['frames']:
        x, y = [], []
        for point in frame['keypoints']:
            x.append(point[0])
            y.append(point[1])
        bx, by, w, h = frame['box']
        plt.scatter(x, y, marker='.', color='green')
        plt.plot((bx, by), (bx + w, by), linewidth=1, color='black')
        plt.plot((bx, by), (bx, by + h), linewidth=1, color='black')
        plt.axis('equal')
        plt.axis('off')
        plt.show()
        break


if __name__ == "__main__":
    # for i in range(178, 224):
    #     print(i)
    #     del_json_content('11_5', 1, i, False)
    #     del_json_content('11_5', 2, i, False)
    # del_json_content('12_2', 1, 245, False)
    # select_user_from_jpl()
    # video_augmentation('../JPL_Augmented_Posefeatures/more_video_ori2/', '../JPL_Augmented_Posefeatures/more_video/')
    # label_person_id('Wantest9_2-ori-flip.avi', 1)
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
    # edit_videos()
    draw_keypoints()
