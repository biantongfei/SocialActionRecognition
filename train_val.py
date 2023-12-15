from Datasets.AvgDataset import AvgDataset, get_tra_test_files
from Datasets.PerFrameDataset import PerFrameDataset
from Models import DNN
from draw_utils import draw_performance, plot_confusion_matrix

from torch.utils.data import DataLoader
from torch import device, cuda, optim, float, save, backends, Tensor
from torch.nn import functional
import random
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

avg_batch_size = 128
perframe_batch_size = 512
valset_rate = 0.2
if cuda.is_available():
    device = device("cuda:0")
elif backends.mps.is_available():
    device = device('mps')
else:
    device = device('cpu')
dtype = float
model_save_path = 'models/'
ori_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw']
added_classes = ['hand_shake', 'hug', 'pet', 'wave', 'point-converse', 'punch', 'throw', 'not_interested', 'interested']
attitude_classes = ['interacting', 'not_interested', 'interested']


def full_video_train_avg(action_recognition=False, body_part=4, ori_videos=False):
    """

    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    # train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'medium_noise+coco': {}, 'medium_noise+halpe': {}}
    train_dict = {'crop+coco': {}}
    performance_dict = {}
    for key in train_dict.keys():
        performance_dict[key] = {'accuracy': [], 'f1': [], 'auc': [], 'loss': []}

    for hyperparameter_group in train_dict.keys():
        print('loading data for', hyperparameter_group)
        is_crop = True if 'crop' in hyperparameter_group else False
        is_coco = True if 'coco' in hyperparameter_group else False
        sigma = None if '_' not in hyperparameter_group else hyperparameter_group.split('_')[0]
        tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco, sigma=sigma,
                                                   not_add_class=action_recognition == 1, ori_videos=ori_videos)
        trainset = AvgDataset(data_files=tra_files[int(len(tra_files) * valset_rate):],
                              action_recognition=action_recognition, is_crop=is_crop, sigma=sigma, is_coco=is_coco,
                              body_part=body_part)
        valset = AvgDataset(data_files=tra_files[:int(len(tra_files) * valset_rate)],
                            action_recognition=action_recognition, is_crop=is_crop, sigma=sigma, is_coco=is_coco,
                            body_part=body_part)
        testset = AvgDataset(data_files=test_files, action_recognition=action_recognition,
                             is_crop=is_crop, sigma=sigma, is_coco=is_coco, body_part=body_part)
        net = DNN(is_coco=is_coco, action_recognition=action_recognition, body_part=body_part)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        train_dict[hyperparameter_group] = {'is_crop': is_crop, 'sigma': sigma, 'is_coco': is_coco,
                                            'trainset': trainset, 'valset': valset,
                                            'testset': testset, 'net': net, 'optimizer': optimizer, 'best_acc': 0,
                                            'best_f1': 0, 'unimproved_epoch': 0}
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
            train_loader = DataLoader(dataset=train_dict[hyperparameter_group]['trainset'], batch_size=avg_batch_size)
            val_loader = DataLoader(dataset=train_dict[hyperparameter_group]['valset'], batch_size=avg_batch_size)
            net = train_dict[hyperparameter_group]['net']
            optimizer = train_dict[hyperparameter_group]['optimizer']
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                outputs = net(inputs)
                loss = functional.cross_entropy(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_ture, y_pred, y_score = [], [], []
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                outputs = net(inputs)
                print(outputs.sum(axis=0))
                print(inputs.shape,outputs.shape)
                pred = outputs.argmax(dim=1)
                y_ture += labels.tolist()
                y_pred += pred.tolist()
                y_score += outputs.tolist()
            y_ture, y_pred, y_score = Tensor(y_ture), Tensor(y_pred), Tensor(y_score)
            acc = y_pred.eq(y_ture).sum().float().item()
            f1 = f1_score(y_ture, y_pred, average='weighted')
            auc = roc_auc_score(y_ture, y_score, multi_class='ovo')
            performance_dict[hyperparameter_group]['accuracy'].append(acc)
            performance_dict[hyperparameter_group]['f1'].append(f1)
            performance_dict[hyperparameter_group]['auc'].append(auc)
            performance_dict[hyperparameter_group]['loss'].append(loss)
            if acc > train_dict[hyperparameter_group]['best_acc'] or f1 > train_dict[hyperparameter_group]['beat_f1']:
                train_dict[hyperparameter_group]['best_acc'] = acc if acc > train_dict[hyperparameter_group][
                    'best_acc'] else train_dict[hyperparameter_group]['best_acc']
                train_dict[hyperparameter_group]['best_f1'] = f1 if f1 > train_dict[hyperparameter_group][
                    'best_f1'] else train_dict[hyperparameter_group]['best_f1']
                train_dict[hyperparameter_group]['unimproved_epoch'] = 0
            else:
                train_dict[hyperparameter_group]['unimproved_epoch'] += 1
            print(
                'epcoch: %d, hyperparameter_group: %s, unimproved_epoch: %d, acc: %s, f1_score: %s, auc: %s, loss: %s' % (
                    epoch, hyperparameter_group, train_dict[hyperparameter_group]['unimproved_epoch'],
                    "%.4f%%" % (acc * 100), "%.4f%%" % (f1), "%.4f%%" % (auc), "%.4f" % loss))
        epoch += 1
        print('------------------------------------------')
    best_acc = 0
    hg = ''
    for hyperparameter_group in train_dict:
        test_loader = DataLoader(dataset=train_dict[hyperparameter_group]['testset'], batch_size=avg_batch_size)
        total_correct = 0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(dtype).to(device), labels.to(device)
            net = train_dict[hyperparameter_group]['net'].to(device)
            outputs = net(inputs)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(labels).sum().float().item()
            total_correct += correct
        acc = total_correct / len(test_loader.dataset)
        if acc > best_acc:
            y_true = labels
            y_pred = pred
            best_acc = acc
            hg = hyperparameter_group
        print('hyperparameter_group: %s, acc: %s, f1_score: %s, auc: %s' % (
            hyperparameter_group, "%.4f%%" % (acc), "%.4f%%" % (f1), "%.4f%%" % (auc)))
        print('----------------------------------------------------')
        # save(net.state_dict(), model_save_path + 'fuullvideo_avg_%s.pth' % (hyperparameter_group))
        if action_recognition == 1:
            classes = ori_classes
        elif action_recognition == 2:
            classes = added_classes
        else:
            classes = attitude_classes
        plot_confusion_matrix(y_true, y_pred, classes, sub_name=hg)
        draw_performance(performance_dict, sub_name=hg)
    return


def train_perframe(action_recognition=True, body_part=4):
    """

        :param
        action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
        :return:
        """
    # train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'small_noise+coco': {}, 'small_noise+halpe': {},
    #               'medium_noise+coco': {}, 'medium_noise+halpe': {}, 'big_noise+coco': {}, 'big_noise+halpe': {}}
    # accuracy_loss_dict = {'crop+coco': [[], []], 'crop+halpe': [[], []], 'small_noise+coco': [[], []],
    #                       'small_noise+halpe': [[], []], 'medium_noise+coco': [[], []], 'medium_noise+halpe': [[], []],
    #                       'big_noise+coco': [[], []], 'big_noise+halpe': [[], []]}
    train_dict = {'crop+coco': {}}
    accuracy_loss_dict = {'crop+coco': [[], []]}

    for hyperparameter_group in train_dict.keys():
        is_crop = True if 'crop' in hyperparameter_group else False
        is_coco = True if 'coco' in hyperparameter_group else False
        sigma = None if '_' not in hyperparameter_group else hyperparameter_group.split('_')[0]
        tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco, sigma=sigma,
                                                   not_add_class=action_recognition == 1)
        net = DNN(is_coco=is_coco, action_recognition=action_recognition)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        train_dict[hyperparameter_group] = {'is_crop': is_crop, 'sigma': sigma, 'is_coco': is_coco,
                                            'tra_files': tra_files[int(len(tra_files) * valset_rate):],
                                            'val_files': tra_files[:int(len(tra_files) * valset_rate)],
                                            'test_files': test_files, 'net': net, 'optimizer': optimizer, 'best_acc': 0,
                                            'unimproved_epoch': 0}

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
                random.shuffle(train_dict[hyperparameter_group]['tra_files'])
                trainset = PerFrameDataset(data_files=train_dict[hyperparameter_group]['tra_files'],
                                           action_recognition=action_recognition,
                                           is_crop=train_dict[hyperparameter_group]['is_crop'],
                                           sigma=train_dict[hyperparameter_group]['sigma'],
                                           is_coco=train_dict[hyperparameter_group]['is_coco'])
                train_loader = DataLoader(dataset=trainset, batch_size=perframe_batch_size)
                net = train_dict[hyperparameter_group]['net']
                optimizer = train_dict[hyperparameter_group]['optimizer']
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = functional.cross_entropy(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_correct = 0
                val_files = train_dict[hyperparameter_group]['val_files']
                for val in val_files:
                    val_set = PerFrameDataset(data_files=[val], action_recognition=action_recognition,
                                              is_crop=train_dict[hyperparameter_group]['is_crop'],
                                              sigma=train_dict[hyperparameter_group]['sigma'],
                                              is_coco=train_dict[hyperparameter_group]['is_coco'])
                    val_dataloader = DataLoader(val_set, batch_size=perframe_batch_size)
                    pred_list = []
                    for data in val_dataloader:
                        inputs, labels = data
                        inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                        outputs = net(inputs)
                        pred = outputs.argmax(dim=1)
                        pred_list += pred.tolist()
                        label = labels[0]
                    total_correct += 1 if np.argmax(np.bincount(pred_list)) == label else 0
                acc = total_correct / len(val_files)
                accuracy_loss_dict[hyperparameter_group][0].append(acc)
                accuracy_loss_dict[hyperparameter_group][1].append(loss)
                if acc > train_dict[hyperparameter_group]['best_acc']:
                    train_dict[hyperparameter_group]['best_acc'] = acc
                    train_dict[hyperparameter_group]['unimproved_epoch'] = 0
                else:
                    train_dict[hyperparameter_group]['unimproved_epoch'] += 1
                print('epcoch: %d, hyperparameter_group: %s, acc: %s, unimproved_epoch: %d, loss: %s' % (
                    epoch, hyperparameter_group, "%.2f%%" % (acc * 100),
                    train_dict[hyperparameter_group]['unimproved_epoch'], "%.5f" % loss))
            epoch += 1
            print('------------------------------------------')
        best_acc = 0
        hg = ''
        for hyperparameter_group in train_dict:
            total_correct = 0
            test_files = train_dict[hyperparameter_group]['test_files']
            for test in test_files:
                test_set = PerFrameDataset(data_files=[test], action_recognition=action_recognition,
                                           is_crop=train_dict[hyperparameter_group]['is_crop'],
                                           sigma=train_dict[hyperparameter_group]['sigma'],
                                           is_coco=train_dict[hyperparameter_group]['is_coco'])
                test_dataloader = DataLoader(test_set, batch_size=perframe_batch_size)
                pred_list = []
                for data in test_dataloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                    outputs = net(inputs)
                    pred = outputs.argmax(dim=1)
                    pred_list += pred.tolist()
                    label = labels[0]
                total_correct += 1 if np.argmax(np.bincount(pred_list)) == label else 0
            acc = total_correct / len(test_files)
            if acc > best_acc:
                y_true = labels
                y_pred = pred
                best_acc = acc
                hg = hyperparameter_group
            print('hyperparameter_group: %s, acc: %s,' % (
                hyperparameter_group, "%.2f%%" % (acc * 100)))
            print('----------------------------------------------------')
            save(net.state_dict(), model_save_path + 'avg_fcnn_%s.pth' % (hyperparameter_group))
        if action_recognition == 1:
            classes = ori_classes
        elif action_recognition == 2:
            classes = added_classes
        else:
            classes = attitude_classes
        plot_confusion_matrix(y_true, y_pred, classes, sub_name=hg)
        draw_performance(accuracy_loss_dict, sub_name=hg)
        return


if __name__ == '__main__':
    for i in range(10):
        full_video_train_avg(action_recognition=1, body_part=[True, True, True], ori_videos=True)
    # traine_perframe(action_recognition=2, body_part=4)
