from Dataset import AvgDataset, get_tra_test_files
from Models import FCNN
from draw_utils import draw_performance

from torch.utils.data import DataLoader
from torch import device, cuda, optim, float, int64
from torch.nn import CrossEntropyLoss, functional
import random

batch_size = 128
valset_rate = 0.1
device = device("cuda:0" if cuda.is_available() else "cpu")
dtype = float


def train_avg(trained_model_num, action_recognition=True):
    train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}}
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
                                            'optimizer': optimizer}

    print('data loaded')
    accuracy_dict = {'crop+coco': [], 'crop+halpe': [], 'noise+coco': [], 'noise+halpe': []}
    epoch = 0
    unimproved_epoches = 0
    while int(unimproved_epoches / len(train_dict.keys())) + 1 < 5:
        for hyperparameter_group in train_dict.keys():
            random.shuffle(train_dict[hyperparameter_group]['tra_files'])
            trainset = AvgDataset(data_files=train_dict[hyperparameter_group]['tra_files'][
                                             :int(len(train_dict[hyperparameter_group]['tra_files']) * valset_rate)],
                                  action_recognition=action_recognition,
                                  is_crop=train_dict[hyperparameter_group]['is_crop'],
                                  is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            valset = AvgDataset(data_files=train_dict[hyperparameter_group]['tra_files'][
                                           int(len(train_dict[hyperparameter_group]['tra_files']) * valset_rate):],
                                action_recognition=action_recognition,
                                is_crop=train_dict[hyperparameter_group]['is_crop'],
                                is_coco=train_dict[hyperparameter_group]['is_coco'], dimension=dimension)
            train_loader = DataLoader(dataset=trainset, batch_size=batch_size)
            val_loader = DataLoader(dataset=valset, batch_size=batch_size)
            net = train_dict[hyperparameter_group]['net']
            optimizer = train_dict[hyperparameter_group]['optimizer']
            for idx, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(dtype).to(device)
                outputs = net(inputs)
                labels_onehot = functional.one_hot(labels.to(int64))
                loss = CrossEntropyLoss()
                loss(outputs, labels_onehot)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_correct = 0
            for idx, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(dtype).to(device), labels.to(dtype).to(device)
                outputs = net(inputs)
                pred = outputs.argmax(dim=1)
                correct = pred.eq(labels).sum().float().item()
                total_correct += correct
            acc = total_correct / len(val_loader.dataset)
            accuracy_dict[hyperparameter_group].append(acc)
            if acc <= accuracy_dict[hyperparameter_group][-2]:
                unimproved_epoches += 1
            else:
                unimproved_epoches = 0
            print('epcoch: %d, hyperparameter_group: %s, acc: %s, unimproved_epoch: %d, trained_model_num: %d' % (
                epoch, hyperparameter_group, '{.2%f}' % (acc * 100),
                int(unimproved_epoches / len(train_dict.keys())) + 1, trained_model_num))
    for hyperparameter_group in train_dict:
        test_loader = DataLoader(dataset=train_dict[hyperparameter_group]['testset'], batch_size=batch_size)
        for idx, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(dtype).to(device), labels.to(dtype).to(device)
            outputs = net(inputs)
            pred = outputs.argmax(dim=1)
            correct = pred.eq(labels).sum().float().item()
            total_correct += correct
        acc = total_correct / len(val_loader.dataset)
        print('hyperparameter_group: %s, acc: %s' % (hyperparameter_group, '{.2f}' % (acc * 100)))
        print('----------------------------------------------------')
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
    for i in range(5):
        train_dict = train_avg(trained_model_num=i, action_recognition=True)
        train_log.append(train_dict)
    train_dict = cal_avg_performance(train_log)
    draw_performance(train_dict)
