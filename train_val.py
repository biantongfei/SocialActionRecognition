from Datasets.AvgDataset import AvgDataset, get_tra_test_files
from Models import FCNN
from draw_utils import draw_performance

from torch.utils.data import DataLoader
from torch import device, cuda, optim, float, save, int64
from torch.nn import functional
import random

batch_size = 128
valset_rate = 0.1
device = device("cuda:0" if cuda.is_available() else "cpu")
dtype = float
model_save_path = 'models/'


def train_avg(action_recognition=True):
    train_dict = {'crop+coco': {}, 'crop+halpe': {}}
    # train_dict = {'small_noise+coco': {}, 'small_noise+halpe': {}}
    # train_dict = {'medium_noise+coco': {}, 'medium_noise+halpe': {}}
    # train_dict = {'big_noise+coco': {}, 'big_noise+halpe': {}}
    dimension = 1  # FCNN
    # dimension = 2  # CNN
    for hyperparameter_group in train_dict.keys():
        is_crop = True if 'crop' in hyperparameter_group else False
        is_coco = True if 'coco' in hyperparameter_group else False
        tra_files, test_files = get_tra_test_files(is_crop=is_crop, is_coco=is_coco)
        testset = AvgDataset(data_files=test_files, action_recognition=action_recognition,
                             is_crop=is_crop, is_coco=is_coco, dimension=dimension)
        net = FCNN(is_coco=is_coco, action_recognition=action_recognition)
        # net = CNN(is_coco=is_coco, action_recognition=action_recognition)
        net.to(device)
        optimizer = optim.SGD(net.parameters(), lr=1e-4)
        # optimizer = optim.adam(net.parameters(), lr=1e-4)
        train_dict[hyperparameter_group] = {'is_crop': is_crop, 'is_coco': is_coco, 'dimension': dimension,
                                            'tra_files': tra_files, 'testset': testset, 'net': net,
                                            'optimizer': optimizer, 'best_acc': 0}

    print('Start Training!!!')
    accuracy_dict = {'crop+coco': [], 'crop+halpe': [], 'noise+coco': [], 'noise+halpe': []}
    epoch = 1
    improved = False
    unimproved_epoch = 0
    while unimproved_epoch < 3:
        for hyperparameter_group in train_dict.keys():
            random.shuffle(train_dict[hyperparameter_group]['tra_files'])
            trainset = AvgDataset(data_files=train_dict[hyperparameter_group]['tra_files'][
                                             int(len(train_dict[hyperparameter_group]['tra_files']) * valset_rate):],
                                  action_recognition=action_recognition,
                                  is_crop=train_dict[hyperparameter_group]['is_crop'],
                                  is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            valset = AvgDataset(data_files=train_dict[hyperparameter_group]['tra_files'][
                                           :int(len(train_dict[hyperparameter_group]['tra_files']) * valset_rate)],
                                action_recognition=action_recognition,
                                is_crop=train_dict[hyperparameter_group]['is_crop'],
                                is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            train_loader = DataLoader(dataset=trainset, batch_size=batch_size)
            val_loader = DataLoader(dataset=valset, batch_size=batch_size)
            net = train_dict[hyperparameter_group]['net']
            optimizer = train_dict[hyperparameter_group]['optimizer']
            for idx, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                outputs = net(inputs)
                labels_onehot = functional.one_hot(labels.to(int64))
                loss = functional.mse_loss(outputs, labels)
                # loss = functional.cross_entropy(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_correct = 0
            for idx, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(device)
                outputs = net(inputs)
                pred = outputs.argmax(dim=1)
                correct = pred.eq(labels).sum().float().item()
                total_correct += correct
            acc = total_correct / len(val_loader.dataset)
            accuracy_dict[hyperparameter_group].append(acc)
            if acc > train_dict[hyperparameter_group]['best_acc']:
                improved = True
                train_dict[hyperparameter_group]['best_acc'] = acc
            print('epcoch: %d, hyperparameter_group: %s, acc: %s, unimproved_epoch: %d' % (
                epoch, hyperparameter_group, "%.2f%%" % (acc * 100), unimproved_epoch))
        if improved:
            improved = False
            unimproved_epoch = 0
        else:
            unimproved_epoch += 1
        epoch += 1
        print('------------------------------------------')
    for hyperparameter_group in train_dict:
        test_loader = DataLoader(dataset=train_dict[hyperparameter_group]['testset'], batch_size=batch_size)
        for idx, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(dtype).to(device), labels.to(device)
            net = train_dict[hyperparameter_group]['net'].to(device)
            outputs = net(inputs)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(labels).sum().float().item()
            total_correct += correct
        acc = total_correct / len(val_loader.dataset)
        print('hyperparameter_group: %s, acc: %s,' % (
            hyperparameter_group, "%.2f%%" % (acc * 100)))
        print('----------------------------------------------------')
        save(net.state_dict(), model_save_path + 'avg_fcnn_%s.pth' % (hyperparameter_group))
    return train_dict


def cal_avg_performance(train_log):
    hyperparam_dict = {}
    for log in train_log:
        for hyperparameter_group in log.keys:
            if hyperparameter_group in hyperparam_dict.keys():
                for index, acc in enumerate(log[hyperparameter_group]):
                    hyperparam_dict[hyperparameter_group][index] += acc / len(train_log)
            else:
                hyperparam_dict[hyperparameter_group] = [a / len(train_log) for a in log[hyperparameter_group]]
    return hyperparam_dict


if __name__ == '__main__':
    train_log = []
    train_dict = train_avg(action_recognition=True)
    train_log.append(train_dict)
    train_dict = cal_avg_performance(train_log)
    draw_performance(train_dict)
