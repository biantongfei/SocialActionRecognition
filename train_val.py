from Dataset import Dataset, get_tra_test_files
from Models import DNN, RNN
from draw_utils import draw_training_process, plot_confusion_matrix

import torch
from torch.utils.data import DataLoader
from torch.nn import functional
from sklearn.metrics import f1_score
import csv

avg_batch_size = 128
perframe_batch_size = 512
valset_rate = 0.2
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
dtype = torch.float
model_save_path = 'models/'
ori_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw']
added_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw', 'not_interested', 'interested']
attitude_classes = ['interacting', 'not_interested', 'interested']


def save_performance(performance):
    with open('performance.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for index, trainging_process in enumerate(performance):
            data = [index + 1]
            for key in trainging_process.keys():
                data.append(trainging_process[key]['accuracy'])
                data.append(trainging_process[key]['f1'])
            spamwriter.writerow(data)


def transform_preframe_result(y_true, y_pred, frame_num_list):
    index_1 = 0
    index_2 = 0
    y, y_hat = [], []
    for frame_num in frame_num_list:
        index_2 += frame_num
        label = int(torch.mean(y_true[index_1:index_2]))
        predict = int(torch.mode(y_true[index_1:index_2]))
        y.append(label)
        y_hat.append(predict)
        index_1 += frame_num
    return torch.Tensor(y), torch.Tensor(y_hat)


def train(action_recognition, body_part=None, ori_videos=False, video_len=99999, form='normal'):
    """
    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    if body_part[0]:
        train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}}
    else:
        train_dict = {'crop+coco': {}, 'noise+coco': {}}
    # train_dict = {'noise+coco': {}, 'noise+halpe': {}}
    # train_dict = {'noise+coco': {}}
    trainging_process = {}
    performance_dict = {}
    for key in train_dict.keys():
        trainging_process[key] = {'accuracy': [], 'f1': [], 'loss': []}
        performance_dict[key] = {'accuracy': None, 'f1': None}

    for hyperparameter_group in train_dict.keys():
        print('loading data for', hyperparameter_group)
        is_crop = True if 'crop' in hyperparameter_group else False
        is_coco = True if 'coco' in hyperparameter_group else False
        tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco,
                                                   not_add_class=action_recognition == 1, ori_videos=ori_videos)
        trainset = Dataset(data_files=tra_files[int(len(tra_files) * valset_rate):],
                           action_recognition=action_recognition, is_crop=is_crop, is_coco=is_coco,
                           body_part=body_part, video_len=video_len, form=form)
        valset = Dataset(data_files=tra_files[:int(len(tra_files) * valset_rate)],
                         action_recognition=action_recognition, is_crop=is_crop, is_coco=is_coco,
                         body_part=body_part, video_len=video_len, form=form)
        testset = Dataset(data_files=test_files, action_recognition=action_recognition, is_crop=is_crop,
                          is_coco=is_coco, body_part=body_part, video_len=video_len, form=form)
        net = DNN(is_coco=is_coco, action_recognition=action_recognition, body_part=body_part)
        # net = LSTM(is_coco=is_coco, action_recognition=action_recognition, body_part=body_part, video_len=video_len,
        #            bidirectional=False)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        train_dict[hyperparameter_group] = {'is_crop': is_crop, 'is_coco': is_coco, 'trainset': trainset,
                                            'valset': valset, 'testset': testset, 'net': net, 'optimizer': optimizer,
                                            'best_acc': -1, 'best_f1': -1, 'unimproved_epoch': 0}
    print('Start Training!!!')
    epoch = 1
    continue_train = True
    while continue_train:
        continue_train = False
        for hyperparameter_group in train_dict.keys():
            if train_dict[hyperparameter_group]['unimproved_epoch'] < 3:
                continue_train = True
            else:
                continue
            train_loader = DataLoader(dataset=train_dict[hyperparameter_group]['trainset'], batch_size=avg_batch_size,
                                      shuffle=True, drop_last=False)
            val_loader = DataLoader(dataset=train_dict[hyperparameter_group]['valset'], batch_size=avg_batch_size, )
            net = train_dict[hyperparameter_group]['net']
            optimizer = train_dict[hyperparameter_group]['optimizer']
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                net.train()
                outputs = net(inputs)
                loss = functional.cross_entropy(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_true, y_pred = [], []
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                net.eval()
                outputs = net(inputs)
                pred = outputs.argmax(dim=1)
                y_true += labels.tolist()
                y_pred += pred.tolist()
            y_true, y_pred = torch.Tensor(y_true), torch.Tensor(y_pred)
            if form == 'perframe':
                y_true, y_pred = transform_preframe_result(y_true, y_pred,
                                                           train_dict[hyperparameter_group]['valset'].frame_number_list)
            acc = y_pred.eq(y_true).sum().float().item() / len(val_loader.dataset)
            f1 = f1_score(y_true, y_pred, average='weighted')
            trainging_process[hyperparameter_group]['accuracy'].append(acc)
            trainging_process[hyperparameter_group]['f1'].append(f1)
            trainging_process[hyperparameter_group]['loss'].append(float(loss))
            if acc > train_dict[hyperparameter_group]['best_acc'] or f1 > train_dict[hyperparameter_group]['best_f1']:
                train_dict[hyperparameter_group]['best_acc'] = acc if acc > train_dict[hyperparameter_group][
                    'best_acc'] else train_dict[hyperparameter_group]['best_acc']
                train_dict[hyperparameter_group]['best_f1'] = f1 if f1 > train_dict[hyperparameter_group][
                    'best_f1'] else train_dict[hyperparameter_group]['best_f1']
                train_dict[hyperparameter_group]['unimproved_epoch'] = 0
            else:
                train_dict[hyperparameter_group]['unimproved_epoch'] += 1
            print('%s, epcoch: %d, unimproved_epoch: %d, acc: %s, f1: %s, loss: %s' % (
                hyperparameter_group, epoch, train_dict[hyperparameter_group]['unimproved_epoch'],
                "%.2f%%" % (acc * 100),
                "%.4f" % (f1), "%.4f" % loss))
        epoch += 1
        print('------------------------------------------')
    if action_recognition == 1:
        classes = ori_classes
    elif action_recognition == 2:
        classes = added_classes
    else:
        classes = attitude_classes
    for hyperparameter_group in train_dict:
        test_loader = DataLoader(dataset=train_dict[hyperparameter_group]['testset'], batch_size=avg_batch_size)
        y_true, y_pred = [], []
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(dtype).to(device), labels.to(device)
            net = train_dict[hyperparameter_group]['net'].to(device)
            net.eval()
            outputs = net(inputs)
            pred = outputs.argmax(dim=1)
            y_true += labels.tolist()
            y_pred += pred.tolist()
        y_true, y_pred = torch.Tensor(y_true), torch.Tensor(y_pred)
        acc = y_pred.eq(y_true).sum().float().item() / len(test_loader.dataset)
        f1 = f1_score(y_true, y_pred, average='weighted')
        performance_dict[hyperparameter_group]['accuracy'] = acc
        performance_dict[hyperparameter_group]['f1'] = f1
        print('%s: acc: %s, f1_score: %s' % (hyperparameter_group, "%.2f%%" % (acc * 100), "%.4f" % (f1)))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # save(net.state_dict(), model_save_path + 'fuullvideo_avg_%s.pth' % (hyperparameter_group))
        plot_confusion_matrix(y_true, y_pred, classes, sub_name=hyperparameter_group)
    draw_training_process(trainging_process)
    return performance_dict


if __name__ == '__main__':
    performance = []
    for i in range(10):
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        p = train(action_recognition=False, body_part=[True, False, False], ori_videos=False, form='avg')
        performance.append(p)
    save_performance(performance)
