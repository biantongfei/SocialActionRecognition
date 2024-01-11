from Dataset import Dataset, get_tra_test_files
from Models import DNN, RNN
from draw_utils import draw_training_process, plot_confusion_matrix

import torch
from torch.utils.data import DataLoader
from torch.nn import functional
from sklearn.metrics import f1_score
import csv

avg_batch_size = 128
perframe_batch_size = 2048
avg_train_epoch = 3
perframe_train_epoch = 3
valset_rate = 0.2
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
ori_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw']
added_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw', 'not_interested', 'interested']
attitude_classes = ['interacting', 'not_interested', 'interested']


def draw_save(performance_model, action_recognition):
    if action_recognition == 1:
        classes = ori_classes
    elif action_recognition == 2:
        classes = added_classes
    else:
        classes = attitude_classes
    y_true = {}
    y_pred = {}
    best_acc = -1
    best_model = None
    with open('plots/performance.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for index, p_m in enumerate(performance_model):
            data = [index + 1]
            for key in p_m.keys():
                if best_acc < p_m[key]['accuracy']:
                    best_acc = p_m[key]['accuracy']
                    best_model = p_m[key]['model']
                data.append(p_m[key]['accuracy'])
                data.append(p_m[key]['f1'])
                if key in y_true.keys():
                    y_true[key] = torch.cat((y_true[key], p_m[key]['y_true']), dim=0)
                    y_pred[key] = torch.cat((y_pred[key], p_m[key]['y_pred']), dim=0)
                else:
                    y_true[key] = p_m[key]['y_true']
                    y_pred[key] = p_m[key]['y_pred']
                plot_confusion_matrix(y_true[key], y_pred[key], classes, sub_name=key)
            spamwriter.writerow(data)
        csvfile.close()
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


def train(model, action_recognition, body_part, sample_fps, video_len=99999, ori_videos=False, ):
    """
    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    if body_part[0]:
        train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}}
    else:
        train_dict = {'crop+coco': {}, 'noise+coco': {}}
    # if body_part[0]:
    #     train_dict = {'crop+coco': {}, 'crop+halpe': {}}
    # else:
    #     train_dict = {'crop+coco': {}}
    trainging_process = {}
    performance_model = {}
    for key in train_dict.keys():
        trainging_process[key] = {'accuracy': [], 'f1': [], 'loss': []}
        performance_model[key] = {'accuracy': None, 'f1': None, 'y_true': None, 'y_pred': None, 'model': None}

    if model == 'avg':
        batch_size = avg_batch_size
        epoch_limit = avg_train_epoch
    elif model == 'perframe':
        batch_size = perframe_batch_size
        epoch_limit = perframe_train_epoch

    for hyperparameter_group in train_dict.keys():
        print('loading data for', hyperparameter_group)
        is_crop = True if 'crop' in hyperparameter_group else False
        is_coco = True if 'coco' in hyperparameter_group else False
        tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco,
                                                   not_add_class=action_recognition == 1, ori_videos=ori_videos)
        trainset = Dataset(data_files=tra_files[int(len(tra_files) * valset_rate):],
                           action_recognition=action_recognition, is_crop=is_crop, is_coco=is_coco,
                           body_part=body_part, model=model, sample_fps=sample_fps, video_len=video_len)
        valset = Dataset(data_files=tra_files[:int(len(tra_files) * valset_rate)],
                         action_recognition=action_recognition, is_crop=is_crop, is_coco=is_coco,
                         body_part=body_part, model=model, sample_fps=sample_fps, video_len=video_len)
        testset = Dataset(data_files=test_files, action_recognition=action_recognition, is_crop=is_crop,
                          is_coco=is_coco, body_part=body_part, model=model, sample_fps=sample_fps, video_len=video_len)
        print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (
            len(trainset), len(valset), len(testset)))
        if model == 'avg' or model == 'perframe':
            net = DNN(is_coco=is_coco, action_recognition=action_recognition, body_part=body_part, model=model)
        elif model == 'lstm':
            net = RNN(is_coco=is_coco, action_recognition=action_recognition, body_part=body_part, video_len=video_len,
                      bidirectional=False)
        elif model == 'gru':
            net = RNN(is_coco=is_coco, action_recognition=action_recognition, body_part=body_part, video_len=video_len,
                      bidirectional=False, gru=True)
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
            if train_dict[hyperparameter_group]['unimproved_epoch'] < epoch_limit:
                continue_train = True
            else:
                continue
            train_loader = DataLoader(dataset=train_dict[hyperparameter_group]['trainset'], batch_size=batch_size,
                                      shuffle=True)
            val_loader = DataLoader(dataset=train_dict[hyperparameter_group]['valset'], batch_size=batch_size, )
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
            if model == 'perframe':
                y_true, y_pred = transform_preframe_result(y_true, y_pred,
                                                           train_dict[hyperparameter_group]['valset'].frame_number_list)
            acc = y_pred.eq(y_true).sum().float().item() / y_pred.size(dim=0)
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
    for hyperparameter_group in train_dict:
        test_loader = DataLoader(dataset=train_dict[hyperparameter_group]['testset'], batch_size=batch_size)
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
        if model == 'perframe':
            y_true, y_pred = transform_preframe_result(y_true, y_pred,
                                                       train_dict[hyperparameter_group]['testset'].frame_number_list)
        acc = y_pred.eq(y_true).sum().float().item() / y_pred.size(dim=0)
        f1 = f1_score(y_true, y_pred, average='weighted')
        performance_model[hyperparameter_group]['accuracy'] = acc
        performance_model[hyperparameter_group]['f1'] = f1
        performance_model[hyperparameter_group]['y_true'] = y_true
        performance_model[hyperparameter_group]['y_pred'] = y_pred
        performance_model[hyperparameter_group]['model'] = net
        print('%s: acc: %s, f1_score: %s' % (hyperparameter_group, "%.2f%%" % (acc * 100), "%.4f" % (f1)))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    draw_training_process(trainging_process)
    return performance_model


if __name__ == '__main__':
    action_recognition = 1
    body_part = [False, False, True]
    ori_video = False
    sample_fps = 30
    performance_model = []
    for i in range(10):
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        p_m = train(model='perframe', action_recognition=action_recognition, body_part=body_part, sample_fps=sample_fps,
                    ori_videos=ori_video)
        performance_model.append(p_m)
    draw_save(performance_model, action_recognition=action_recognition)
