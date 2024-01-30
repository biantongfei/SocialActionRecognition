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
perframe_train_epoch = 2
rnn_train_epoch = 5
conv1d_epoch = 3
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
attitude_classes = ['positive', 'neutral', 'negative', 'uninterested']
action_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw', 'uninterested', 'interested']


def draw_save(performance_model):
    att_best_acc = -1
    best_model = None
    with open('plots/performance.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        att_y_true = {}
        att_y_pred = {}
        act_y_true = {}
        act_y_pred = {}
        for index, p_m in enumerate(performance_model):
            data = [index + 1]
            for key in p_m.keys():
                if att_best_acc < p_m[key]['attitude_accuracy']:
                    att_best_acc = p_m[key]['attitude_accuracy']
                    best_model = p_m[key]['model']
                data.append(p_m[key]['attitude_accuracy'])
                data.append(p_m[key]['attitude_f1'])
                data.append(p_m[key]['action_accuracy'])
                data.append(p_m[key]['action_f1'])
                if key in att_y_true.keys():
                    att_y_true[key] = torch.cat((att_y_true[key], p_m[key]['attitude_y_true']), dim=0)
                    att_y_pred[key] = torch.cat((att_y_pred[key], p_m[key]['attitude_y_pred']), dim=0)
                    act_y_true[key] = torch.cat((act_y_true[key], p_m[key]['action_y_true']), dim=0)
                    act_y_pred[key] = torch.cat((act_y_pred[key], p_m[key]['action_y_pred']), dim=0)
                else:
                    att_y_true[key] = p_m[key]['attitude_y_true']
                    att_y_pred[key] = p_m[key]['attitude_y_pred']
                    act_y_true[key] = p_m[key]['action_y_true']
                    act_y_pred[key] = p_m[key]['action_y_pred']
            spamwriter.writerow(data)
        csvfile.close()
    for key in att_y_true.keys():
        plot_confusion_matrix(att_y_true[key], att_y_pred[key], attitude_classes, sub_name="%s_attitude" % key)
        plot_confusion_matrix(act_y_true[key], act_y_pred[key], action_classes, sub_name="%s_action" % key)
    torch.save(best_model.state_dict(), 'plots/model.pth')


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


def train(model, body_part, sample_fps, video_len=99999, ori_videos=False):
    """
    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}, 'mixed_same+coco': {},
                  'mixed_same+halpe': {}, 'mixed_large+coco': {}, 'mixed_large+halpe': {}}
    train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}, 'mixed_same+coco': {},
                  'mixed_same+halpe': {}}
    train_dict = {'mixed_large+coco': {}, 'mixed_large+halpe': {}}
    # if body_part[0]:
    #     train_dict = {'crop+coco': {}, 'crop+halpe': {}}
    # else:
    #     train_dict = {'crop+coco': {}}
    # if body_part[0]:
    #     train_dict = {'mixed_same+coco': {}, 'mixed_same+halpe': {}, 'mixed_large+coco': {}, 'mixed_same': {}}
    # else:
    #     train_dict = {'mixed_same+coco': {}, 'mixed_large+coco': {}}
    train_dict = {'mixed_large+halpe': {}}
    # train_dict = {'crop+coco': {}}
    trainging_process = {}
    performance_model = {}
    for key in train_dict.keys():
        trainging_process[key] = {'attitude_accuracy': [], 'attitude_f1': [], 'action_accuracy': [], 'action_f1': [],
                                  'loss': []}
        performance_model[key] = {'attitude_accuracy': None, 'attitude_f1': None, 'attitude_y_true': None,
                                  'attitude_y_pred': None, 'action_accuracy': None, 'action_f1': None,
                                  'action_y_true': None, 'action_y_pred': None, 'model': None}

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
        epoch_limit = conv1d_epoch
        learning_rate = conv1d_learning_rate

    for hyperparameter_group in train_dict.keys():
        print('loading data for', hyperparameter_group)
        augment_method = hyperparameter_group.split('+')[0]
        is_coco = True if 'coco' in hyperparameter_group else False
        tra_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco,
                                                   ori_videos=ori_videos)
        trainset = Dataset(data_files=tra_files[int(len(tra_files) * valset_rate):], augment_method=augment_method,
                           is_coco=is_coco, body_part=body_part, model=model, sample_fps=sample_fps,
                           video_len=video_len)
        valset = Dataset(data_files=tra_files[:int(len(tra_files) * valset_rate)], augment_method=augment_method,
                         is_coco=is_coco, body_part=body_part, model=model, sample_fps=sample_fps, video_len=video_len)
        testset = Dataset(data_files=test_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                          model=model, sample_fps=sample_fps, video_len=video_len)
        max_length = max(trainset.max_length, valset.max_length, testset.max_length)
        print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (
            len(trainset), len(valset), len(testset)))
        if model in ['avg', 'perframe']:
            net = DNN(is_coco=is_coco, body_part=body_part, model=model)
        elif model in ['lstm', 'gru']:
            net = RNN(is_coco=is_coco, body_part=body_part, bidirectional=True,
                      gru=model == 'gru')
        elif model == 'conv1d':
            net = Cnn1D(is_coco=is_coco, body_part=body_part)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        train_dict[hyperparameter_group] = {'augment_method': augment_method, 'is_coco': is_coco, 'trainset': trainset,
                                            'valset': valset, 'testset': testset, 'net': net, 'optimizer': optimizer,
                                            'att_best_acc': -1, 'att_best_f1': -1, 'act_best_acc': -1,
                                            'act_best_f1': -1, 'unimproved_epoch': 0}
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
                                         batch_size=batch_size, max_length=max_length, shuffle=True)
            val_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['valset'],
                                       max_length=max_length, batch_size=batch_size)
            net = train_dict[hyperparameter_group]['net']
            optimizer = train_dict[hyperparameter_group]['optimizer']
            for data in train_loader:
                if model in ['avg', 'perframe', 'conv1d']:
                    inputs, (att_labels, act_labels) = data
                elif model in ['lstm', 'gru']:
                    (inputs, (att_labels, act_labels)), data_length = data
                    inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs, att_labels, act_labels = inputs.to(dtype).to(device), att_labels.to(device), act_labels.to(
                    device)
                net.train()
                att_outputs, act_outputs = net(inputs)
                loss_1 = functional.cross_entropy(att_outputs, att_labels)
                loss_2 = functional.cross_entropy(act_outputs, act_labels)
                optimizer.zero_grad()
                total_loss = loss_1 + loss_2
                total_loss.backward()
                optimizer.step()

            att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], []
            for data in val_loader:
                if model in ['avg', 'perframe', 'conv1d']:
                    inputs, (att_labels, act_labels) = data
                elif model in ['lstm', 'gru']:
                    (inputs, (att_labels, act_labels)), data_length = data
                    inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs, att_labels, act_labels = inputs.to(dtype).to(device), att_labels.to(device), act_labels.to(
                    device)
                net.eval()
                att_outputs, act_outputs = net(inputs)
                att_pred = att_outputs.argmax(dim=1)
                act_pred = act_outputs.argmax(dim=1)
                # print(pred)
                att_y_true += att_labels.tolist()
                att_y_pred += att_pred.tolist()
                act_y_true += act_labels.tolist()
                act_y_pred += act_pred.tolist()
            att_y_true, att_y_pred, act_y_true, act_y_pred = torch.Tensor(att_y_true), torch.Tensor(
                att_y_pred), torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
            if model == 'perframe':
                att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                                   train_dict[hyperparameter_group][
                                                                       'valset'].frame_number_list)
                act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred,
                                                                   train_dict[hyperparameter_group][
                                                                       'valset'].frame_number_list)
            att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
            att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
            act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
            act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
            trainging_process[hyperparameter_group]['attitude_accuracy'].append(att_acc)
            trainging_process[hyperparameter_group]['attitude_f1'].append(att_f1)
            trainging_process[hyperparameter_group]['action_accuracy'].append(act_acc)
            trainging_process[hyperparameter_group]['action_f1'].append(act_f1)
            trainging_process[hyperparameter_group]['loss'].append(float(total_loss))
            if att_acc > train_dict[hyperparameter_group]['att_best_acc'] or att_f1 > train_dict[hyperparameter_group][
                'att_best_f1'] or act_acc > train_dict[hyperparameter_group]['act_best_acc'] or act_f1 > \
                    train_dict[hyperparameter_group]['act_best_f1']:
                train_dict[hyperparameter_group]['att_best_acc'] = att_acc if att_acc > \
                                                                              train_dict[hyperparameter_group][
                                                                                  'att_best_acc'] else \
                    train_dict[hyperparameter_group]['att_best_acc']
                train_dict[hyperparameter_group]['att_best_f1'] = att_f1 if att_f1 > train_dict[hyperparameter_group][
                    'att_best_f1'] else train_dict[hyperparameter_group]['att_best_f1']
                train_dict[hyperparameter_group]['act_best_acc'] = act_acc if act_acc > \
                                                                              train_dict[hyperparameter_group][
                                                                                  'act_best_acc'] else \
                    train_dict[hyperparameter_group]['act_best_acc']
                train_dict[hyperparameter_group]['act_best_f1'] = act_f1 if act_f1 > train_dict[hyperparameter_group][
                    'act_best_f1'] else train_dict[hyperparameter_group]['act_best_f1']
                train_dict[hyperparameter_group]['unimproved_epoch'] = 0
            else:
                train_dict[hyperparameter_group]['unimproved_epoch'] += 1
            print('%s, epcoch: %d, unimproved_epoch: %d, att_acc: %s, att_f1: %s, act_acc: %s, act_f1: %s, loss: %s' % (
                hyperparameter_group, epoch, train_dict[hyperparameter_group]['unimproved_epoch'],
                "%.2f%%" % (att_acc * 100), "%.4f" % att_f1, "%.2f%%" % (act_acc * 100), "%.4f" % act_f1,
                "%.4f" % total_loss))
        epoch += 1
        print('------------------------------------------')
        # break

    for hyperparameter_group in train_dict:
        test_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['testset'],
                                    max_length=max_length, batch_size=batch_size)
        att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], []
        for data in test_loader:
            if model in ['avg', 'perframe', 'conv1d']:
                inputs, (att_labels, act_labels) = data
            elif model in ['lstm', 'gru']:
                (inputs, (att_labels, act_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
            inputs, att_labels, act_labels = inputs.to(dtype).to(device), att_labels.to(device), act_labels.to(device)
            net = train_dict[hyperparameter_group]['net'].to(device)
            net.eval()
            att_outputs, act_outputs = net(inputs)
            att_pred = att_outputs.argmax(dim=1)
            act_pred = act_outputs.argmax(dim=1)
            att_y_true += att_labels.tolist()
            att_y_pred += att_pred.tolist()
            act_y_true += act_labels.tolist()
            act_y_pred += act_pred.tolist()
        att_y_true, att_y_pred, act_y_true, act_y_pred = torch.Tensor(att_y_true), torch.Tensor(
            att_y_pred), torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
        if model == 'perframe':
            att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                               train_dict[hyperparameter_group][
                                                                   'valset'].frame_number_list)
            act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred,
                                                               train_dict[hyperparameter_group][
                                                                   'valset'].frame_number_list)
        att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
        att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
        performance_model[hyperparameter_group]['attitude_accuracy'] = att_acc
        performance_model[hyperparameter_group]['attitude_f1'] = att_f1
        performance_model[hyperparameter_group]['attitude_y_true'] = att_y_true
        performance_model[hyperparameter_group]['attitude_y_pred'] = att_y_pred
        performance_model[hyperparameter_group]['action_accuracy'] = act_acc
        performance_model[hyperparameter_group]['action_f1'] = act_f1
        performance_model[hyperparameter_group]['action_y_true'] = act_y_true
        performance_model[hyperparameter_group]['action_y_pred'] = act_y_pred
        performance_model[hyperparameter_group]['model'] = net
        print('%s: att_acc: %s, att_f1: %s, act_acc: %s, act_f1: %s' % (
            hyperparameter_group, "%.2f%%" % (att_acc * 100), "%.4f" % att_f1, "%.2f%%" % (act_acc * 100),
            "%.4f" % act_f1))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # draw_training_process(trainging_process)
    return performance_model


if __name__ == '__main__':
    model = 'avg'
    body_part = [True, True, True]
    ori_video = False
    sample_fps = 6
    video_len = 2
    performance_model = []
    i = 0
    while i < 10:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if video_len:
            p_m = train(model=model, body_part=body_part, sample_fps=sample_fps, ori_videos=ori_video,
                        video_len=video_len)
        else:
            p_m = train(model=model, body_part=body_part, sample_fps=sample_fps, ori_videos=ori_video)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(performance_model)
    print('model: %s, body_part:' % model, body_part, ', sample_fps: %d, video_len: %s' % (sample_fps, str(video_len)))
