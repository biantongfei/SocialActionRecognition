from Dataset import Dataset, get_tra_test_files
from Models import DNN, RNN, Cnn1D, GNN
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
conv1d_batch_size = 32
gnn_batch_size = 32
avg_train_epoch = 1
perframe_train_epoch = 1
rnn_train_epoch = 1
conv1d_train_epoch = 1
gnn_train_epoch = 1
valset_rate = 0.2
learning_rate = 1e-3
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


def draw_save(performance_model, framework):
    tasks = [framework] if framework in ['intent', 'attitude', 'action'] else ['intent', 'attitude', 'action']
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
            keys = p_m.keys()
            for key in p_m.keys():
                if 'intent' in tasks:
                    data.append(p_m[key]['intent_accuracy'])
                    data.append(p_m[key]['intent_f1'])
                    if key in int_y_true.keys():
                        int_y_true[key] = torch.cat((int_y_true[key], p_m[key]['intent_y_true']), dim=0)
                        int_y_pred[key] = torch.cat((int_y_pred[key], p_m[key]['intent_y_pred']), dim=0)
                    else:
                        int_y_true[key] = p_m[key]['intent_y_true']
                        int_y_pred[key] = p_m[key]['intent_y_pred']
                if 'attitude' in tasks:
                    data.append(p_m[key]['attitude_accuracy'])
                    data.append(p_m[key]['attitude_f1'])
                    if key in att_y_true.keys():
                        att_y_true[key] = torch.cat((att_y_true[key], p_m[key]['attitude_y_true']), dim=0)
                        att_y_pred[key] = torch.cat((att_y_pred[key], p_m[key]['attitude_y_pred']), dim=0)
                    else:
                        att_y_true[key] = p_m[key]['attitude_y_true']
                        att_y_pred[key] = p_m[key]['attitude_y_pred']
                if 'action' in tasks:
                    data.append(p_m[key]['action_accuracy'])
                    data.append(p_m[key]['action_f1'])
                    if key in act_y_true.keys():
                        act_y_true[key] = torch.cat((act_y_true[key], p_m[key]['action_y_true']), dim=0)
                        act_y_pred[key] = torch.cat((act_y_pred[key], p_m[key]['action_y_pred']), dim=0)
                    else:
                        act_y_true[key] = p_m[key]['action_y_true']
                        act_y_pred[key] = p_m[key]['action_y_pred']
            spamwriter.writerow(data)
        csvfile.close()
    # for key in keys:
    #     if 'intent' in tasks:
    #         plot_confusion_matrix(int_y_true[key], int_y_pred[key], intent_class, sub_name="%s_intent" % key)
    #     if 'attitude' in tasks:
    #         plot_confusion_matrix(att_y_true[key], att_y_pred[key], attitude_classes, sub_name="%s_attitude" % key)
    #     if 'action' in tasks:
    #         plot_confusion_matrix(act_y_true[key], act_y_pred[key], action_classes, sub_name="%s_action" % key)


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


def filter_others_from_result(y_true, y_pred, task):
    i = 0
    while i < y_true.shape[0]:
        if (task == 'attitude' and y_true[i] == 2) or (task == 'action' and y_true[i] in [4, 7, 8]):
            if i == y_true.shape[0] - 1:
                y_true = y_true[:i]
                y_pred = y_pred[:i]
            else:
                y_true = torch.cat((y_true[:i], y_true[i + 1:]))
                y_pred = torch.cat((y_pred[:i], y_pred[i + 1:]))
        else:
            i += 1
    return y_true, y_pred


def train(model, body_part, data_format, framework, sample_fps, video_len=99999, ori_videos=False, empty_frame=False):
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
    # train_dict = {'mixed_large+coco': {}, 'mixed_large+halpe': {}}
    # train_dict = {'mixed_same+coco': {}, 'mixed_same+halpe': {}, 'mixed_large+coco': {}, 'mixed_large+halpe': {}}
    # train_dict = {'mixed_large+coco': {}}
    # train_dict = {'mixed_large+halpe': {}}
    train_dict = {'crop+coco': {}}
    tasks = [framework] if framework in ['intent', 'attitude', 'action'] else ['intent', 'attitude', 'action']
    trainging_process = {}
    performance_model = {}
    for key in train_dict.keys():
        for t in tasks:
            trainging_process[key] = {'%s_accuracy' % t: [], '%s_f1' % t: []}
            performance_model[key] = {'%s_accuracy' % t: None, '%s_f1' % t: None, '%s_y_true' % t: None,
                                      '%s_y_pred' % t: None}

    if model == 'avg':
        batch_size = avg_batch_size
        epoch_limit = avg_train_epoch
    elif model == 'perframe':
        batch_size = perframe_batch_size
        epoch_limit = perframe_train_epoch
    elif model in ['lstm', 'gru']:
        batch_size = rnn_batch_size
        epoch_limit = rnn_train_epoch
    elif model == 'conv1d':
        batch_size = conv1d_batch_size
        epoch_limit = conv1d_train_epoch
    elif 'gnn' in model:
        batch_size = gnn_batch_size
        epoch_limit = gnn_train_epoch

    for hyperparameter_group in train_dict.keys():
        print('loading data for', hyperparameter_group)
        augment_method = hyperparameter_group.split('+')[0]
        is_coco = True if 'coco' in hyperparameter_group else False
        tra_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco,
                                                   ori_videos=ori_videos)
        trainset = Dataset(data_files=tra_files[int(len(tra_files) * valset_rate):], augment_method=augment_method,
                           is_coco=is_coco, body_part=body_part, data_format=data_format, model=model,
                           sample_fps=sample_fps, video_len=video_len, empty_frame=empty_frame)
        valset = Dataset(data_files=tra_files[:int(len(tra_files) * valset_rate)], augment_method=augment_method,
                         is_coco=is_coco, body_part=body_part, data_format=data_format, model=model,
                         sample_fps=sample_fps, video_len=video_len, empty_frame=empty_frame)
        testset = Dataset(data_files=test_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                          data_format=data_format, model=model, sample_fps=sample_fps, video_len=video_len,
                          empty_frame=empty_frame)
        max_length = max(trainset.max_length, valset.max_length, testset.max_length)
        print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (
            len(trainset), len(valset), len(testset)))
        if model in ['avg', 'perframe']:
            net = DNN(is_coco=is_coco, body_part=body_part, data_format=data_format, framework=framework)
        elif model in ['lstm', 'gru']:
            net = RNN(is_coco=is_coco, body_part=body_part, data_format=data_format, framework=framework,
                      bidirectional=True, gru=model == 'gru')
        elif model == 'conv1d':
            net = Cnn1D(is_coco=is_coco, body_part=body_part, data_format=data_format, framework=framework,
                        max_length=max_length)
        elif 'gnn' in model:
            net = GNN(is_coco=is_coco, body_part=body_part, data_format=data_format, framework=framework, model=model,
                      max_length=max_length, attention=True)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        train_dict[hyperparameter_group] = {'augment_method': augment_method, 'is_coco': is_coco,
                                            'trainset': trainset,
                                            'valset': valset, 'testset': testset, 'net': net,
                                            'optimizer': optimizer,
                                            'intent_best_f1': -1, 'attitude_best_f1': -1, 'action_best_f1': -1,
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
                        inputs = inputs.to(dtype).to(device)
                    elif model in ['lstm', 'gru']:
                        (inputs, (int_labels, att_labels, act_labels)), data_length = data
                        inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                        inputs = inputs.to(dtype).to(device)
                    elif 'gnn' in model:
                        x, (int_labels, att_labels, act_labels) = data
                        inputs = (x[0].to(dtype).to(device), x[1].to(torch.int64).to(device), x[
                            2].to(dtype).to(device))
                    int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(
                        device)
                    net.train()
                    if framework == 'intent':
                        int_outputs = net(inputs)
                        total_loss = functional.cross_entropy(int_outputs, int_labels)
                    elif framework == 'attitude':
                        att_outputs = net(inputs)
                        att_labels, att_outputs = filter_others_from_result(att_labels, att_outputs, 'attitude')
                        total_loss = functional.cross_entropy(att_outputs, att_labels)
                    elif framework == 'action':
                        act_outputs = net(inputs)
                        act_labels, act_outputs = filter_others_from_result(act_labels, act_outputs, 'action')
                        total_loss = functional.cross_entropy(act_outputs, act_labels)
                    else:
                        int_outputs, att_outputs, act_outputs = net(inputs)
                        att_labels, att_outputs = filter_others_from_result(att_labels, att_outputs, 'attitude')
                        act_labels, act_outputs = filter_others_from_result(act_labels, act_outputs, 'action')
                        loss_1 = functional.cross_entropy(int_outputs, int_labels)
                        loss_2 = functional.cross_entropy(att_outputs, att_labels)
                        loss_3 = functional.cross_entropy(act_outputs, act_labels)
                        total_loss = loss_1 + loss_2 + loss_3
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
                for data in val_loader:
                    if model in ['avg', 'perframe', 'conv1d']:
                        inputs, (int_labels, att_labels, act_labels) = data
                        inputs = inputs.to(dtype).to(device)
                    elif model in ['lstm', 'gru']:
                        (inputs, (int_labels, att_labels, act_labels)), data_length = data
                        inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                        inputs = inputs.to(dtype).to(device)
                    elif 'gnn' in model:
                        x, (int_labels, att_labels, act_labels) = data
                        inputs, edge_index = x[0].to(dtype).to(device), x[1].to(torch.int64).to(device)
                        # edge_index, edge_attr = x[0].to(dtype).to(torch.int64), x[1].to(dtype).to(device)
                        # inputs, edge_index, edge_attr = x[0].to(dtype).to(device), x[1].to(torch.int64).to(device), x[
                        #     2].to(dtype).to(device)
                    int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(
                        device)
                    net.eval()
                    if framework == 'intent':
                        int_outputs = net(inputs)
                    elif framework == 'attitude':
                        att_outputs = net(inputs)
                    elif framework == 'action':
                        act_outputs = net(inputs)
                    else:
                        int_outputs, att_outputs, act_outputs = net(inputs)
                    if 'intent' in tasks:
                        int_pred = int_outputs.argmax(dim=1)
                        int_y_true += int_labels.tolist()
                        int_y_pred += int_pred.tolist()
                    if 'attitude' in tasks:
                        att_pred = att_outputs.argmax(dim=1)
                        att_y_true += att_labels.tolist()
                        att_y_pred += att_pred.tolist()
                    if 'action' in tasks:
                        act_pred = act_outputs.argmax(dim=1)
                        act_y_true += act_labels.tolist()
                        act_y_pred += act_pred.tolist()

                result_str = '%s, epcoch: %d, ' % (hyperparameter_group, epoch)
                if 'intent' in tasks:
                    int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
                    if model == 'perframe':
                        int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred,
                                                                           train_dict[hyperparameter_group][
                                                                               'valset'].frame_number_list)
                    int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
                    int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
                    if int_f1 > train_dict[hyperparameter_group]['intent_best_f1']:
                        train_dict[hyperparameter_group]['intent_best_f1'] = int_f1
                        train_dict[hyperparameter_group]['unimproved_epoch'] = -1
                    result_str += 'int_acc: %s, int_f1: %s, ' % ("%.2f%%" % (int_acc * 100), "%.4f" % int_f1)
                if 'attitude' in tasks:
                    att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
                    if model == 'perframe':
                        att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                                           train_dict[hyperparameter_group][
                                                                               'valset'].frame_number_list)
                    att_y_true, att_y_pred = filter_others_from_result(att_y_true, att_y_pred, 'attitude')
                    att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
                    att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
                    if att_f1 > train_dict[hyperparameter_group]['attitude_best_f1']:
                        train_dict[hyperparameter_group]['attitude_best_f1'] = att_f1
                        train_dict[hyperparameter_group]['unimproved_epoch'] = -1
                    result_str += 'att_acc: %s, att_f1: %s, ' % ("%.2f%%" % (att_acc * 100), "%.4f" % att_f1)
                if 'action' in tasks:
                    act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
                    if model == 'perframe':
                        act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred,
                                                                           train_dict[hyperparameter_group][
                                                                               'valset'].frame_number_list)
                    act_y_true, act_y_pred = filter_others_from_result(act_y_true, act_y_pred, 'action')
                    act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
                    act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
                    if act_f1 > train_dict[hyperparameter_group]['action_best_f1']:
                        train_dict[hyperparameter_group]['action_best_f1'] = act_f1
                        train_dict[hyperparameter_group]['unimproved_epoch'] = -1
                    result_str += 'act_acc: %s, act_f1: %s, ' % ("%.2f%%" % (act_acc * 100), "%.4f" % act_f1)
                if train_dict[hyperparameter_group]['unimproved_epoch'] == -1:
                    train_dict[hyperparameter_group]['unimproved_epoch'] = 0
                else:
                    train_dict[hyperparameter_group]['unimproved_epoch'] += 1
                print(result_str + "loss: %.4f, unimproved_epoch: %d" % (
                    total_loss, train_dict[hyperparameter_group]['unimproved_epoch']))
            epoch += 1
            print('------------------------------------------')
            break

        for hyperparameter_group in train_dict:
            test_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['testset'],
                                        max_length=max_length, batch_size=batch_size, empty_frame=empty_frame)
            int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
            for data in test_loader:
                if model in ['avg', 'perframe', 'conv1d']:
                    inputs, (int_labels, att_labels, act_labels) = data
                    inputs = inputs.to(dtype).to(device)
                elif model in ['lstm', 'gru']:
                    (inputs, (int_labels, att_labels, act_labels)), data_length = data
                    inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                    inputs = inputs.to(dtype).to(device)
                elif 'gnn' in model:
                    x, (int_labels, att_labels, act_labels) = data
                    inputs, edge_index = x[0].to(dtype).to(device), x[1].to(torch.int64).to(device)
                    # edge_index, edge_attr = x[0].to(dtype).to(torch.int64), x[1].to(dtype).to(device)
                    # inputs, edge_index, edge_attr = x[0].to(dtype).to(device), x[1].to(torch.int64).to(device), x[
                    #     2].to(dtype).to(device)
                int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
                net.eval()
                if framework in ['intent', 'attitude', 'action']:
                    if framework == 'intent':
                        int_outputs = net(inputs)
                    elif framework == 'attitude':
                        att_outputs = net(inputs)
                    else:
                        act_outputs = net(inputs)
                else:
                    int_outputs, att_outputs, act_outputs = net(inputs)
                if 'intent' in tasks:
                    int_pred = int_outputs.argmax(dim=1)
                    int_y_true += int_labels.tolist()
                    int_y_pred += int_pred.tolist()
                if 'attitude' in tasks:
                    att_pred = att_outputs.argmax(dim=1)
                    att_y_true += att_labels.tolist()
                    att_y_pred += att_pred.tolist()
                if 'action' in tasks:
                    act_pred = act_outputs.argmax(dim=1)
                    act_y_true += act_labels.tolist()
                    act_y_pred += act_pred.tolist()
            result_str = '%s, ' % hyperparameter_group
            if 'intent' in tasks:
                int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
                if model == 'perframe':
                    int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred,
                                                                       train_dict[hyperparameter_group][
                                                                           'testset'].frame_number_list)
                int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
                int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
                performance_model[hyperparameter_group]['intent_accuracy'] = int_acc
                performance_model[hyperparameter_group]['intent_f1'] = int_f1
                performance_model[hyperparameter_group]['intent_y_true'] = int_y_true
                performance_model[hyperparameter_group]['intent_y_pred'] = int_y_pred
                result_str += 'int_acc: %s, int_f1: %s, ' % ("%.2f" % (int_acc * 100), "%.4f" % int_f1)
            if 'attitude' in tasks:
                att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
                if model == 'perframe':
                    att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                                       train_dict[hyperparameter_group][
                                                                           'testset'].frame_number_list)
                att_y_true, att_y_pred = filter_others_from_result(att_y_true, att_y_pred, 'attitude')
                att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
                att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
                performance_model[hyperparameter_group]['attitude_accuracy'] = att_acc
                performance_model[hyperparameter_group]['attitude_f1'] = att_f1
                performance_model[hyperparameter_group]['attitude_y_true'] = att_y_true
                performance_model[hyperparameter_group]['attitude_y_pred'] = att_y_pred
                result_str += 'att_acc: %s, att_f1: %s, ' % ("%.2f" % (att_acc * 100), "%.4f" % att_f1)
            if 'action' in tasks:
                act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
                if model == 'perframe':
                    act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred,
                                                                       train_dict[hyperparameter_group][
                                                                           'testset'].frame_number_list)
                act_y_true, act_y_pred = filter_others_from_result(act_y_true, act_y_pred, 'action')
                act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
                act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
                performance_model[hyperparameter_group]['action_accuracy'] = act_acc
                performance_model[hyperparameter_group]['action_f1'] = act_f1
                performance_model[hyperparameter_group]['action_y_true'] = act_y_true
                performance_model[hyperparameter_group]['action_y_pred'] = act_y_pred
                result_str += 'act_acc: %s, act_f1: %s, ' % ("%.2f" % (act_acc * 100), "%.4f" % act_f1)
            print(result_str)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # draw_training_process(trainging_process)
        return performance_model


if __name__ == '__main__':
    # model = 'avg'
    # model = 'perframe'
    # model = 'conv1d'
    # model = 'lstm'
    model = 'gnn_keypoint_conv1d'
    # model = 'gnn_keypoint_lstm'
    # model = 'gnn_time'
    # model = 'gnn2+1d'
    body_part = [True, True, True]
    data_format = 'coordinates'
    # data_format = 'manhattan'
    # data_format = 'coordinates+manhattan'

    # framework = 'intent'
    # framework = 'attitude'
    # framework = 'action'
    # framework = 'parallel'
    framework = 'tree'
    # framework = 'chain'
    ori_video = False
    sample_fps = 30
    video_len = 2
    empty_frame = 'same'
    performance_model = []
    i = 0
    while i < 2:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if video_len:
            p_m = train(model=model, body_part=body_part, data_format=data_format, framework=framework,
                        sample_fps=sample_fps, ori_videos=ori_video, video_len=video_len, empty_frame=empty_frame)
        else:
            p_m = train(model=model, body_part=body_part, data_format=data_format, framework=framework,
                        sample_fps=sample_fps, ori_videos=ori_video, empty_frame=empty_frame)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(performance_model, framework)
    print('model: %s, body_part:' % model, body_part,
          ', framework: %s, sample_fps: %d, video_len: %s, empty_frame: %s' % (
              framework, sample_fps, str(video_len), str(empty_frame)))
