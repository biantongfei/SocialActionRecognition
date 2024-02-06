from Dataset import Dataset, get_tra_test_files
from Models import DNN, RNN, Cnn1D
from draw_utils import draw_training_process, plot_confusion_matrix

import torch
from DataLoader import JPLDataLoader
from torch.nn import functional
from torch import nn
import torch.nn.utils.rnn as rnn_utils

from sklearn.metrics import f1_score
import csv

avg_batch_size = 128
perframe_batch_size = 2048
rnn_batch_size = 32
conv1d_batch_size = 64
avg_train_epoch = 3
perframe_train_epoch = 1
rnn_train_epoch = 5
conv1d_train_epoch = 1
valset_rate = 0.2
dnn_learning_rate = 1e-3
rnn_learning_rate = 1e-3
conv1d_learning_rate = 1e-3
if torch.cuda.is_available():
    print('Using CUDA for training')
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    print('Using MPS for training')
    device = torch.device('mps')
else:
    print('Using CPU for training')
    device = torch.device('cpu')
dtype = torch.float
intent_class = ['interacting', 'interested', 'uninterested']
attitude_classes = ['positive', 'negative', 'others']
action_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw', 'uninterested', 'interested']


def draw_save(performance_model):
    with open('plots/performance.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        int_y_true = {}
        int_y_pred = {}
        att_y_true = {}
        att_y_pred = {}
        act_y_true = {}
        act_y_pred = {}
        for index, p_m in enumerate(performance_model):
            data = [index + 1]
            for key in p_m.keys():
                data.append(p_m[key]['intent_accuracy'])
                data.append(p_m[key]['attitude_accuracy'])
                data.append(p_m[key]['action_accuracy'])
                if key in att_y_true.keys():
                    int_y_true[key] = torch.cat((int_y_true[key], p_m[key]['intent_y_true']), dim=0)
                    int_y_pred[key] = torch.cat((int_y_pred[key], p_m[key]['intent_y_pred']), dim=0)
                    att_y_true[key] = torch.cat((att_y_true[key], p_m[key]['attitude_y_true']), dim=0)
                    att_y_pred[key] = torch.cat((att_y_pred[key], p_m[key]['attitude_y_pred']), dim=0)
                    act_y_true[key] = torch.cat((act_y_true[key], p_m[key]['action_y_true']), dim=0)
                    act_y_pred[key] = torch.cat((act_y_pred[key], p_m[key]['action_y_pred']), dim=0)
                else:
                    int_y_true[key] = p_m[key]['intent_y_true']
                    int_y_pred[key] = p_m[key]['intent_y_pred']
                    att_y_true[key] = p_m[key]['attitude_y_true']
                    att_y_pred[key] = p_m[key]['attitude_y_pred']
                    act_y_true[key] = p_m[key]['action_y_true']
                    act_y_pred[key] = p_m[key]['action_y_pred']
            spamwriter.writerow(data)
        csvfile.close()
    for key in att_y_true.keys():
        plot_confusion_matrix(int_y_true[key], int_y_pred[key], intent_class, sub_name="%s_intent" % key)
        plot_confusion_matrix(att_y_true[key], att_y_pred[key], attitude_classes, sub_name="%s_attitude" % key)
        plot_confusion_matrix(act_y_true[key], act_y_pred[key], action_classes, sub_name="%s_action" % key)


def transform_preframe_result(y_true, y_pred, frame_num_list):
    index_1 = 0
    index_2 = 0
    y, y_hat = [], []
    for frame_num in frame_num_list:
        index_2 += frame_num
        label = int(torch.mean(y_true[index_1:index_2]))
        predict = int(torch.mode(y_pred[index_1:index_2])[0])
        y.append(label)
        y_hat.append(predict)
        index_1 += frame_num
    return torch.Tensor(y), torch.Tensor(y_hat)


def filter_others_from_result(y_true, y_pred):
    i = 0
    while i < y_true.shape[0]:
        if y_true[i] == 2:
            if i == y_true.shape[0] - 1:
                y_true = y_true[:i]
                y_pred = y_pred[:i]
            else:
                y_true = torch.cat((y_true[:i], y_true[i + 1:]))
                y_pred = torch.cat((y_pred[:i], y_pred[i + 1:]))
        else:
            i += 1
    return y_true, y_pred


def train(model, body_part, framework, sample_fps, video_len=99999, ori_videos=False, empty_frame=False):
    """
    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    # train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}, 'mixed_same+coco': {},
    #               'mixed_same+halpe': {}, 'mixed_large+coco': {}, 'mixed_large+halpe': {}}
    # train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}, 'mixed_same+coco': {},
    #               'mixed_same+halpe': {}}
    # train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}}
    # train_dict = {'mixed_same+coco': {}}
    # train_dict = {'mixed_same+coco': {}, 'mixed_same+halpe': {}, 'mixed_large+coco': {}, 'mixed_large+halpe': {}}
    train_dict = {'mixed_large+halpe': {}}
    # train_dict = {'crop+coco': {}}
    trainging_process = {}
    performance_model = {}
    for key in train_dict.keys():
        trainging_process[key] = {'intent_accuracy': [], 'attitude_accuracy': [], 'action_accuracy': [], 'loss': []}
        performance_model[key] = {'intent_accuracy': None, 'intent_y_true': None, 'intent_y_pred': None,
                                  'attitude_accuracy': None, 'attitude_y_true': None, 'attitude_y_pred': None,
                                  'action_accuracy': None, 'action_y_true': None, 'action_y_pred': None}

    if model == 'avg':
        batch_size = avg_batch_size
        epoch_limit = avg_train_epoch
        learning_rate = dnn_learning_rate
    elif model == 'perframe':
        batch_size = perframe_batch_size
        epoch_limit = perframe_train_epoch
        learning_rate = dnn_learning_rate
    elif model in ['lstm', 'gru']:
        batch_size = rnn_batch_size
        epoch_limit = rnn_train_epoch
        learning_rate = rnn_learning_rate
    elif model == 'conv1d':
        batch_size = conv1d_batch_size
        epoch_limit = conv1d_train_epoch
        learning_rate = conv1d_learning_rate

    for hyperparameter_group in train_dict.keys():
        print('loading data for', hyperparameter_group)
        augment_method = hyperparameter_group.split('+')[0]
        is_coco = True if 'coco' in hyperparameter_group else False
        tra_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco,
                                                   ori_videos=ori_videos)
        trainset = Dataset(data_files=tra_files[int(len(tra_files) * valset_rate):], augment_method=augment_method,
                           is_coco=is_coco, body_part=body_part, model=model, sample_fps=sample_fps,
                           video_len=video_len, empty_frame=empty_frame)
        valset = Dataset(data_files=tra_files[:int(len(tra_files) * valset_rate)], augment_method=augment_method,
                         is_coco=is_coco, body_part=body_part, model=model, sample_fps=sample_fps, video_len=video_len,
                         empty_frame=empty_frame)
        testset = Dataset(data_files=test_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                          model=model, sample_fps=sample_fps, video_len=video_len, empty_frame=empty_frame)
        max_length = max(trainset.max_length, valset.max_length, testset.max_length)
        print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (
            len(trainset), len(valset), len(testset)))
        if model in ['avg', 'perframe']:
            net = DNN(is_coco=is_coco, body_part=body_part, framework=framework)
        elif model in ['lstm', 'gru']:
            net = RNN(is_coco=is_coco, body_part=body_part, framework=framework, bidirectional=True, gru=model == 'gru')
        elif model == 'conv1d':
            net = Cnn1D(is_coco=is_coco, body_part=body_part, framework=framework)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        train_dict[hyperparameter_group] = {'augment_method': augment_method, 'is_coco': is_coco, 'trainset': trainset,
                                            'valset': valset, 'testset': testset, 'net': net, 'optimizer': optimizer,
                                            'int_best_acc': -1, 'att_best_acc': -1, 'act_best_acc': -1,
                                            'unimproved_epoch': 0}
    print('Start Training!!!')
    epoch = 1
    continue_train = True
    while continue_train:
        continue_train = False
        for hyperparameter_group in train_dict.keys():
            if train_dict[hyperparameter_group]['unimproved_epoch'] < epoch_limit:
                continue_train = True
            else:
                continue
            train_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['trainset'],
                                         batch_size=batch_size, max_length=max_length, shuffle=True,
                                         empty_frame=empty_frame)
            val_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['valset'],
                                       max_length=max_length, batch_size=batch_size, empty_frame=empty_frame)
            net = train_dict[hyperparameter_group]['net']
            optimizer = train_dict[hyperparameter_group]['optimizer']
            for data in train_loader:
                if model in ['avg', 'perframe', 'conv1d']:
                    inputs, (int_labels, att_labels, act_labels) = data
                elif model in ['lstm', 'gru']:
                    (inputs, (int_labels, att_labels, act_labels)), data_length = data
                    inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs, int_labels, att_labels, act_labels = inputs.to(dtype).to(device), int_labels.to(
                    device), att_labels.to(device), act_labels.to(device)
                net.train()
                int_outputs, att_outputs, act_outputs = net(inputs)
                loss_1 = functional.cross_entropy(int_outputs, int_labels)
                loss_2 = functional.cross_entropy(att_outputs, att_labels)
                loss_3 = functional.cross_entropy(act_outputs, act_labels)
                optimizer.zero_grad()
                total_loss = loss_1 + loss_2 + loss_3
                total_loss.backward()
                optimizer.step()

            int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
            for data in val_loader:
                if model in ['avg', 'perframe', 'conv1d']:
                    inputs, (int_labels, att_labels, act_labels) = data
                elif model in ['lstm', 'gru']:
                    (inputs, (int_labels, att_labels, act_labels)), data_length = data
                    inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs, int_labels, att_labels, act_labels = inputs.to(dtype).to(device), int_labels.to(
                    device), att_labels.to(device), act_labels.to(device)
                net.eval()
                int_outputs, att_outputs, act_outputs = net(inputs)
                int_pred = int_outputs.argmax(dim=1)
                att_pred = att_outputs.argmax(dim=1)
                act_pred = act_outputs.argmax(dim=1)
                int_y_true += int_labels.tolist()
                att_y_true += att_labels.tolist()
                act_y_true += act_labels.tolist()
                int_y_pred += int_pred.tolist()
                att_y_pred += att_pred.tolist()
                act_y_pred += act_pred.tolist()
            int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = torch.Tensor(
                int_y_true), torch.Tensor(int_y_pred), torch.Tensor(att_y_true), torch.Tensor(att_y_pred), torch.Tensor(
                act_y_true), torch.Tensor(act_y_pred)
            if model == 'perframe':
                int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred,
                                                                   train_dict[hyperparameter_group][
                                                                       'valset'].frame_number_list)
                att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                                   train_dict[hyperparameter_group][
                                                                       'valset'].frame_number_list)
                act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred,
                                                                   train_dict[hyperparameter_group][
                                                                       'valset'].frame_number_list)
            att_y_true, att_y_pred = filter_others_from_result(att_y_true, att_y_pred)
            int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
            att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
            act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
            trainging_process[hyperparameter_group]['intent_accuracy'].append(att_acc)
            trainging_process[hyperparameter_group]['attitude_accuracy'].append(att_acc)
            trainging_process[hyperparameter_group]['action_accuracy'].append(act_acc)
            trainging_process[hyperparameter_group]['loss'].append(float(total_loss))
            if int_acc > train_dict[hyperparameter_group]['int_best_acc'] or att_acc > train_dict[hyperparameter_group][
                'att_best_acc'] or act_acc > train_dict[hyperparameter_group]['act_best_acc']:
                train_dict[hyperparameter_group]['int_best_acc'] = int_acc if int_acc > \
                                                                              train_dict[hyperparameter_group][
                                                                                  'int_best_acc'] else \
                    train_dict[hyperparameter_group]['int_best_acc']
                train_dict[hyperparameter_group]['att_best_acc'] = att_acc if att_acc > \
                                                                              train_dict[hyperparameter_group][
                                                                                  'att_best_acc'] else \
                    train_dict[hyperparameter_group]['att_best_acc']
                train_dict[hyperparameter_group]['act_best_acc'] = act_acc if act_acc > \
                                                                              train_dict[hyperparameter_group][
                                                                                  'act_best_acc'] else \
                    train_dict[hyperparameter_group]['act_best_acc']
                train_dict[hyperparameter_group]['unimproved_epoch'] = 0
            else:
                train_dict[hyperparameter_group]['unimproved_epoch'] += 1
            print('%s, epcoch: %d, unimproved_epoch: %d, int_acc: %s, att_acc: %s, act_acc: %s, loss: %s' % (
                hyperparameter_group, epoch, train_dict[hyperparameter_group]['unimproved_epoch'],
                "%.2f%%" % (int_acc * 100), "%.2f%%" % (att_acc * 100), "%.2f%%" % (act_acc * 100),
                "%.4f" % total_loss))
        epoch += 1
        print('------------------------------------------')
        # break

    for hyperparameter_group in train_dict:
        test_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['testset'],
                                    max_length=max_length, batch_size=batch_size, empty_frame=empty_frame)
        int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
        for data in test_loader:
            if model in ['avg', 'perframe', 'conv1d']:
                inputs, (int_labels, att_labels, act_labels) = data
            elif model in ['lstm', 'gru']:
                (inputs, (int_labels, att_labels, act_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
            inputs, int_labels, att_labels, act_labels = inputs.to(dtype).to(device), int_labels.to(
                device), att_labels.to(device), act_labels.to(device)
            net = train_dict[hyperparameter_group]['net'].to(device)
            net.eval()
            int_outputs, att_outputs, act_outputs = net(inputs)
            int_pred = int_outputs.argmax(dim=1)
            att_pred = att_outputs.argmax(dim=1)
            act_pred = act_outputs.argmax(dim=1)
            int_y_true += int_labels.tolist()
            int_y_pred += int_pred.tolist()
            att_y_true += att_labels.tolist()
            att_y_pred += att_pred.tolist()
            act_y_true += act_labels.tolist()
            act_y_pred += act_pred.tolist()
        int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = torch.Tensor(int_y_true), torch.Tensor(
            int_y_pred), torch.Tensor(att_y_true), torch.Tensor(att_y_pred), torch.Tensor(act_y_true), torch.Tensor(
            act_y_pred)
        if model == 'perframe':
            int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred,
                                                               train_dict[hyperparameter_group][
                                                                   'testset'].frame_number_list)
            att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                               train_dict[hyperparameter_group][
                                                                   'testset'].frame_number_list)
            act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred,
                                                               train_dict[hyperparameter_group][
                                                                   'testset'].frame_number_list)
        att_y_true, att_y_pred = filter_others_from_result(att_y_true, att_y_pred)
        int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
        att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        performance_model[hyperparameter_group]['intent_accuracy'] = int_acc
        performance_model[hyperparameter_group]['intent_y_true'] = int_y_true
        performance_model[hyperparameter_group]['intent_y_pred'] = int_y_pred
        performance_model[hyperparameter_group]['attitude_accuracy'] = att_acc
        performance_model[hyperparameter_group]['attitude_y_true'] = att_y_true
        performance_model[hyperparameter_group]['attitude_y_pred'] = att_y_pred
        performance_model[hyperparameter_group]['action_accuracy'] = act_acc
        performance_model[hyperparameter_group]['action_y_true'] = act_y_true
        performance_model[hyperparameter_group]['action_y_pred'] = act_y_pred
        print('%s: int_acc: %s, att_acc: %s, act_acc: %s' % (
            hyperparameter_group, "%.2f%%" % (int_acc * 100), "%.2f%%" % (att_acc * 100), "%.2f%%" % (act_acc * 100)))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # draw_training_process(trainging_process)
    return performance_model


if __name__ == '__main__':
    model = 'conv1d'
    body_part = [True, True, True]
    # framework = 'parallel'
    # framework = 'tree'
    framework = 'chain'
    ori_video = False
    sample_fps = 30
    video_len = False
    empty_frame = False
    performance_model = []
    i = 0
    while i < 10:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if video_len:
            p_m = train(model=model, body_part=body_part, framework=framework, sample_fps=sample_fps,
                        ori_videos=ori_video, video_len=video_len, empty_frame=empty_frame)
        else:
            p_m = train(model=model, body_part=body_part, framework=framework, sample_fps=sample_fps,
                        ori_videos=ori_video, empty_frame=empty_frame)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(performance_model)
    print('model: %s, body_part:' % model, body_part,
          ', sample_fps: %d, video_len: %s, empty_frame: %s' % (sample_fps, str(video_len), str(empty_frame)))
