from Datasets.AvgDataset import AvgDataset, get_tra_test_files
from Datasets.PerFrameDataset import PerFrameDataset, cal_acc
from Models import FCNN, CNN
from draw_utils import draw_performance, plot_confusion_matrix

from torch.utils.data import DataLoader
from torch import device, cuda, optim, float, save, backends
from torch.nn import functional
import random
import numpy as np

batch_size = 128
valset_rate = 0.1
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


def train_avg(lr, action_recognition=False, dimension=1):
    """

    :param
    action_recognition: 0 for origin 7 classes; 1 for add not interested and interested; False for attitude recognition
    dimension: 1 for fcnn; 2 for cnn
    :return:
    """
    train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'small_noise+coco': {}, 'small_noise+halpe': {},
                  'medium_noise+coco': {}, 'medium_noise+halpe': {}, 'big_noise+coco': {}, 'big_noise+halpe': {}}
    accuracy_loss_dict = {'crop+coco': [[], []], 'crop+halpe': [[], []], 'small_noise+coco': [[], []],
                          'small_noise+halpe': [[], []], 'medium_noise+coco': [[], []], 'medium_noise+halpe': [[], []],
                          'big_noise+coco': [[], []], 'big_noise+halpe': [[], []]}
    # train_dict = {'crop+coco': {}}
    # accuracy_dict = {'crop+coco': []}

    for hyperparameter_group in train_dict.keys():
        is_crop = True if 'crop' in hyperparameter_group else False
        is_coco = True if 'coco' in hyperparameter_group else False
        sigma = None if '_' not in hyperparameter_group else hyperparameter_group.split('_')[0]
        tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco, sigma=sigma,
                                                   add_class=action_recognition)
        testset = AvgDataset(data_files=test_files, action_recognition=action_recognition,
                             is_crop=is_crop, sigma=sigma, is_coco=is_coco, dimension=dimension)
        if dimension == 1:
            net = FCNN(is_coco=is_coco, action_recognition=action_recognition)
        else:
            net = CNN(is_coco=is_coco, action_recognition=action_recognition)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        train_dict[hyperparameter_group] = {'is_crop': is_crop, 'sigma': sigma, 'is_coco': is_coco,
                                            'dimension': dimension, 'tra_files': tra_files, 'testset': testset,
                                            'net': net, 'optimizer': optimizer, 'best_acc': 0, 'unimproved_epoch': 0}

    print('Start Training!!!')
    epoch = 1
    continue_train = True
    while continue_train:
        continue_train = False
        for hyperparameter_group in train_dict.keys():
            if train_dict[hyperparameter_group]['unimproved_epoch'] < 5:
                continue_train = True
            else:
                continue
            random.shuffle(train_dict[hyperparameter_group]['tra_files'])
            trainset = AvgDataset(data_files=train_dict[hyperparameter_group]['tra_files'][
                                             int(len(train_dict[hyperparameter_group]['tra_files']) * valset_rate):],
                                  action_recognition=action_recognition,
                                  is_crop=train_dict[hyperparameter_group]['is_crop'],
                                  sigma=train_dict[hyperparameter_group]['sigma'],
                                  is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            valset = AvgDataset(data_files=train_dict[hyperparameter_group]['tra_files'][
                                           :int(len(train_dict[hyperparameter_group]['tra_files']) * valset_rate)],
                                action_recognition=action_recognition,
                                is_crop=train_dict[hyperparameter_group]['is_crop'],
                                sigma=train_dict[hyperparameter_group]['sigma'],
                                is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            train_loader = DataLoader(dataset=trainset, batch_size=batch_size)
            val_loader = DataLoader(dataset=valset, batch_size=batch_size)
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
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                outputs = net(inputs)
                pred = outputs.argmax(dim=1)
                correct = pred.eq(labels).sum().float().item()
                total_correct += correct
            acc = total_correct / len(val_loader.dataset)
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
    for hyperparameter_group in train_dict:
        test_loader = DataLoader(dataset=train_dict[hyperparameter_group]['testset'], batch_size=batch_size)
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
        print('hyperparameter_group: %s, acc: %s,' % (
            hyperparameter_group, "%.2f%%" % (acc * 100)))
        print('----------------------------------------------------')
        save(net.state_dict(), model_save_path + 'avg_fcnn_%s.pth' % (hyperparameter_group))
    if action_recognition:
        classes = added_classes
    elif action_recognition == 0:
        classes = ori_classes
    else:
        classes = attitude_classes
    plot_confusion_matrix(y_true, y_pred, classes)
    return accuracy_loss_dict


def traine_perframe(action_recognition=True):
    train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'small_noise+coco': {}, 'small_noise+halpe': {},
                  'medium_noise+coco': {}, 'medium_noise+halpe': {}, 'big_noise+coco': {}, 'big_noise+halpe': {}}
    accuracy_dict = {'crop+coco': [], 'crop+halpe': [], 'small_noise+coco': [], 'small_noise+halpe': [],
                     'medium_noise+coco': [], 'medium_noise+halpe': [], 'big_noise+coco': [], 'big_noise+halpe': []}

    dimension = 1  # FCNN
    # dimension = 2  # CNN
    for hyperparameter_group in train_dict.keys():
        is_crop = True if 'crop' in hyperparameter_group else False
        is_coco = True if 'coco' in hyperparameter_group else False
        sigma = None if '_' not in hyperparameter_group else hyperparameter_group.split('_')[0]
        tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco, sigma=sigma,
                                                   add_class=action_recognition)
        testset = PerFrameDataset(data_files=test_files, action_recognition=action_recognition,
                                  is_crop=is_crop, sigma=sigma, is_coco=is_coco, dimension=dimension)
        net = FCNN(is_coco=is_coco, action_recognition=action_recognition)
        # net = CNN(is_coco=is_coco, action_recognition=action_recognition)
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        train_dict[hyperparameter_group] = {'is_crop': is_crop, 'sigma': sigma, 'is_coco': is_coco,
                                            'dimension': dimension, 'tra_files': tra_files, 'testset': testset,
                                            'net': net, 'optimizer': optimizer, 'best_acc': 0}

    print('Start Training!!!')
    epoch = 1
    improved = False
    unimproved_epoch = 0
    while unimproved_epoch < 3:
        for hyperparameter_group in train_dict.keys():
            random.shuffle(train_dict[hyperparameter_group]['tra_files'])
            trainset = PerFrameDataset(data_files=train_dict[hyperparameter_group]['tra_files'][
                                                  int(len(
                                                      train_dict[hyperparameter_group]['tra_files']) * valset_rate):],
                                       action_recognition=action_recognition,
                                       is_crop=train_dict[hyperparameter_group]['is_crop'],
                                       sigma=train_dict[hyperparameter_group]['sigma'],
                                       is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            val_files = train_dict[hyperparameter_group]['tra_files'][
                        :int(len(train_dict[hyperparameter_group]['tra_files']) * valset_rate)]
            valset = PerFrameDataset(data_files=val_files, action_recognition=action_recognition,
                                     is_crop=train_dict[hyperparameter_group]['is_crop'],
                                     sigma=train_dict[hyperparameter_group]['sigma'],
                                     is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            train_loader = DataLoader(dataset=trainset, batch_size=batch_size)
            val_loader = DataLoader(dataset=valset, batch_size=batch_size)
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

            val_output = np.array((1,))
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                outputs = net(inputs)
                if val_output.shape[0] == 1:
                    val_output = outputs
                else:
                    val_output = np.append(val_output, outputs, axis=0)
            acc = cal_acc(val_output)
            accuracy_dict[hyperparameter_group].append(acc)
            if acc > train_dict[hyperparameter_group]['best_acc']:
                improved = True
                train_dict[hyperparameter_group]['best_acc'] = acc
            print('epcoch: %d, hyperparameter_group: %s, acc: %s, unimproved_epoch: %d, loss: %s' % (
                epoch, hyperparameter_group, "%.2f%%" % (acc * 100), unimproved_epoch, "%.5f" % loss))
        if improved:
            improved = False
            unimproved_epoch = 0
        else:
            unimproved_epoch += 1
        epoch += 1
        print('------------------------------------------')
    for hyperparameter_group in train_dict:
        test_loader = DataLoader(dataset=train_dict[hyperparameter_group]['testset'], batch_size=batch_size)
        test_output = np.array((1,))
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(dtype).to(device), labels.to(device)
            outputs = net(inputs)
            if test_output.shape[0] == 1:
                test_output = outputs
            else:
                test_output = np.append(test_output, outputs, axis=0)
        acc = cal_acc(test_output)
        print('hyperparameter_group: %s, acc: %s,' % (
            hyperparameter_group, "%.2f%%" % (acc * 100)))
        print('----------------------------------------------------')
        save(net.state_dict(), model_save_path + 'perframe_fcnn_%s.pth' % (hyperparameter_group))
    return accuracy_dict


if __name__ == '__main__':
    for lr in [1e-2, 1e-3, 1e-4, 1e-4, 1e-6]:
        accuracy_loss_dict = train_avg(lr, action_recognition=1, dimension=1)
        draw_performance(accuracy_loss_dict)
