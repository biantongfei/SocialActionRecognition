import time

from Dataset import Dataset, get_tra_test_files
from Models import DNN, RNN, Cnn1D, GNN
from draw_utils import draw_training_process, plot_confusion_matrix

import torch
from DataLoader import JPLDataLoader
from torch.nn import functional
import torch.nn.utils.rnn as rnn_utils

from sklearn.metrics import f1_score
import csv
import smtplib
from email.mime.text import MIMEText
from tqdm import tqdm

bless_str = ("                         _oo0oo_\n"
             "                        o8888888o\n"
             "                        88\" . \"88\n"
             "                        (| -_- |)\n"
             "                        0\  =  /0\n"
             "                      ___/`---'\___\n"
             "                    .' \\|     |// '.\n"
             "                   / \\|||  :  |||// \ \n"
             "                  / _||||| -:- |||||- \ \n"
             "                 |   | \\\  - /// |   |\n"
             "                 | \_|  ''\---/''  |_/ |\n"
             "                 \  .-\__  '-'  ___/-. /\n"
             "               ___'. .'  /--.--\  `. .'___\n"
             "            .\"\" '<  `.___\_<|>_/___.' >' \"\".\n"
             "           | | :  `- \`.;`\ _ /`;.`/ - ` : | |\n"
             "           \  \ `_.   \_ __\ /__ _/   .-` /  /\n"
             "       =====`-.____`.___ \_____/___.-`___.-'=====\n"
             "                         `=---='\n"
             "       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
             "                 BLESS ME WITH NO BUGS\n"
             )
print(bless_str)


def send_email(body):
    subject = "Training Is Done"
    sender = "bian2016buaa@163.com"
    recipients = ["tongfeibian@gmail.com"]
    password = "EJWYZRYQNDKOQSFH"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.163.com', 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Email sent!")


avg_batch_size = 128
perframe_batch_size = 2048
rnn_batch_size = 128
conv1d_batch_size = 128
gcn_batch_size = 128
epoch_limit = 1
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
intention_class = ['interacting', 'interested', 'not_interested']
attitude_classes = ['positive', 'negative', 'no_interacting']
action_classes = ['hand_shake', 'hug', 'pet', 'wave', 'punch', 'throw', 'point-converse', 'gaze', 'leave',
                  'no_response']


def draw_save(name, performance_model, framework):
    tasks = [framework] if framework in ['intention', 'attitude', 'action'] else ['intention', 'attitude', 'action']
    with open('plots/%s.csv' % name, 'w', newline='') as csvfile:
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
                if 'intention' in tasks:
                    data.append(p_m[key]['intention_accuracy'])
                    data.append(p_m[key]['intention_f1'])
                    if key in int_y_true.keys():
                        int_y_true[key] = torch.cat((int_y_true[key], p_m[key]['intention_y_true']), dim=0)
                        int_y_pred[key] = torch.cat((int_y_pred[key], p_m[key]['intention_y_pred']), dim=0)
                    else:
                        int_y_true[key] = p_m[key]['intention_y_true']
                        int_y_pred[key] = p_m[key]['intention_y_pred']
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
    for key in keys:
        if 'intention' in tasks:
            plot_confusion_matrix(int_y_true[key], int_y_pred[key], intention_class,
                                  sub_name="cm_%s_intention" % name)
        if 'attitude' in tasks:
            plot_confusion_matrix(att_y_true[key], att_y_pred[key], attitude_classes,
                                  sub_name="cm_%s_attitude" % name)
        if 'action' in tasks:
            plot_confusion_matrix(act_y_true[key], act_y_pred[key], action_classes,
                                  sub_name="cm_%s_action" % name)


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


def train(model, body_part, framework, sample_fps, video_len=99999, ori_videos=False):
    """
    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    # train_dict = {'crop+coco': {}, 'crop+halpe': {}, 'noise+coco': {}, 'noise+halpe': {}}
    # train_dict = {'mixed+coco': {}, 'mixed+halpe': {}}
    # train_dict = {'mixed+coco': {}}
    # train_dict = {'mixed+halpe': {}}
    train_dict = {'crop+coco': {}}
    tasks = [framework] if framework in ['intention', 'attitude', 'action'] else ['intention', 'attitude', 'action']
    trainging_process = {}
    performance_model = {}
    for key in train_dict.keys():
        for t in tasks:
            trainging_process[key] = {'%s_accuracy' % t: [], '%s_f1' % t: []}
            performance_model[key] = {'%s_accuracy' % t: None, '%s_f1' % t: None, '%s_y_true' % t: None,
                                      '%s_y_pred' % t: None}

    if model == 'avg':
        batch_size = avg_batch_size
    elif model == 'perframe':
        batch_size = perframe_batch_size
    elif model in ['lstm', 'gru']:
        batch_size = rnn_batch_size
    elif model == 'conv1d':
        batch_size = conv1d_batch_size
    elif 'gcn' in model:
        batch_size = gcn_batch_size

    for hyperparameter_group in train_dict.keys():
        print('loading data for', hyperparameter_group)
        augment_method = hyperparameter_group.split('+')[0]
        is_coco = True if 'coco' in hyperparameter_group else False
        tra_files, val_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco,
                                                              ori_videos=ori_videos)
        trainset = Dataset(data_files=tra_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                           model=model, sample_fps=sample_fps, video_len=video_len)
        valset = Dataset(data_files=val_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                         model=model, sample_fps=sample_fps, video_len=video_len)
        testset = Dataset(data_files=test_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                          model=model, sample_fps=sample_fps, video_len=video_len)
        max_length = max(trainset.max_length, valset.max_length, testset.max_length)
        print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (
            len(trainset), len(valset), len(testset)))
        if model in ['avg', 'perframe']:
            net = DNN(is_coco=is_coco, body_part=body_part, framework=framework)
        elif model in ['lstm', 'gru']:
            net = RNN(is_coco=is_coco, body_part=body_part, framework=framework, gru=model == 'gru')
        elif model == 'conv1d':
            net = Cnn1D(is_coco=is_coco, body_part=body_part, framework=framework, max_length=max_length)
        elif 'gcn' in model:
            net = GNN(is_coco=is_coco, body_part=body_part, framework=framework, model=model, max_length=max_length)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        train_dict[hyperparameter_group] = {'augment_method': augment_method, 'is_coco': is_coco, 'trainset': trainset,
                                            'valset': valset, 'testset': testset, 'net': net, 'optimizer': optimizer,
                                            'intention_best_f1': -1, 'attitude_best_f1': -1, 'action_best_f1': -1,
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
                                             batch_size=batch_size, max_length=max_length, drop_last=False,
                                             shuffle=True)
                val_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['valset'],
                                           max_length=max_length, drop_last=False, batch_size=batch_size)
                net = train_dict[hyperparameter_group]['net']
                optimizer = train_dict[hyperparameter_group]['optimizer']
                net.train()
                print('Training')
                progress_bar = tqdm(total=len(train_loader), desc='Progress')
                for data in train_loader:
                    progress_bar.update(1)
                    if model in ['avg', 'perframe', 'conv1d']:
                        inputs, (int_labels, att_labels, act_labels) = data
                        inputs = inputs.to(dtype).to(device)
                    elif model in ['lstm', 'gru']:
                        (inputs, (int_labels, att_labels, act_labels)), data_length = data
                        inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                        inputs = inputs.to(dtype).to(device)
                    elif 'gcn' in model:
                        x, (int_labels, att_labels, act_labels) = data
                        inputs = (x[0].to(dtype).to(device), x[1].to(torch.int64).to(device), x[
                            2].to(dtype).to(device))
                    int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(
                        device)
                    if framework == 'intention':
                        int_outputs = net(inputs)
                        total_loss = functional.cross_entropy(int_outputs, int_labels)
                    elif framework == 'attitude':
                        att_outputs = net(inputs)
                        total_loss = functional.cross_entropy(att_outputs, att_labels)
                    elif framework == 'action':
                        act_outputs = net(inputs)
                        total_loss = functional.cross_entropy(act_outputs, act_labels)
                    else:
                        int_outputs, att_outputs, act_outputs = net(inputs)
                        loss_1 = functional.cross_entropy(int_outputs, int_labels)
                        loss_2 = functional.cross_entropy(att_outputs, att_labels)
                        loss_3 = functional.cross_entropy(act_outputs, act_labels)
                        total_loss = loss_1 + loss_2 + loss_3
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                progress_bar.close()
                print('Validating')
                int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
                net.eval()
                for data in val_loader:
                    if model in ['avg', 'perframe', 'conv1d']:
                        inputs, (int_labels, att_labels, act_labels) = data
                        inputs = inputs.to(dtype).to(device)
                    elif model in ['lstm', 'gru']:
                        (inputs, (int_labels, att_labels, act_labels)), data_length = data
                        inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                        inputs = inputs.to(dtype).to(device)
                    elif 'gcn' in model:
                        x, (int_labels, att_labels, act_labels) = data
                        inputs = (x[0].to(dtype).to(device), x[1].to(torch.int64).to(device), x[
                            2].to(dtype).to(device))
                    int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(
                        device)
                    if framework == 'intention':
                        int_outputs = net(inputs)
                    elif framework == 'attitude':
                        att_outputs = net(inputs)
                    elif framework == 'action':
                        act_outputs = net(inputs)
                    else:
                        int_outputs, att_outputs, act_outputs = net(inputs)
                    if 'intention' in tasks:
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
                if 'intention' in tasks:
                    int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
                    if model == 'perframe':
                        int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred,
                                                                           train_dict[hyperparameter_group][
                                                                               'valset'].frame_number_list)
                    int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
                    int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
                    if int_f1 > train_dict[hyperparameter_group]['intention_best_f1']:
                        train_dict[hyperparameter_group]['intention_best_f1'] = int_f1
                        train_dict[hyperparameter_group]['unimproved_epoch'] = -1
                    result_str += 'int_acc: %s, int_f1: %s, ' % ("%.2f%%" % (int_acc * 100), "%.4f" % int_f1)
                if 'attitude' in tasks:
                    att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
                    if model == 'perframe':
                        att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                                           train_dict[hyperparameter_group][
                                                                               'valset'].frame_number_list)
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

        print('Testing')
        for hyperparameter_group in train_dict:
            test_loader = JPLDataLoader(model=model, dataset=train_dict[hyperparameter_group]['testset'],
                                        max_length=max_length, batch_size=batch_size, drop_last=False)
            int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
            process_time = 0
            for data in test_loader:
                if model in ['avg', 'perframe', 'conv1d']:
                    inputs, (int_labels, att_labels, act_labels) = data
                    inputs = inputs.to(dtype).to(device)
                elif model in ['lstm', 'gru']:
                    (inputs, (int_labels, att_labels, act_labels)), data_length = data
                    inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                    inputs = inputs.to(dtype).to(device)
                elif 'gcn' in model:
                    x, (int_labels, att_labels, act_labels) = data
                    inputs = (x[0].to(dtype).to(device), x[1].to(torch.int64).to(device), x[
                        2].to(dtype).to(device))
                int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
                net.eval()
                start_time = time.time()
                if framework in ['intention', 'attitude', 'action']:
                    if framework == 'intention':
                        int_outputs = net(inputs)
                    elif framework == 'attitude':
                        att_outputs = net(inputs)
                    else:
                        act_outputs = net(inputs)
                else:
                    int_outputs, att_outputs, act_outputs = net(inputs)
                end_time = time.time()
                process_time += end_time - start_time
                if 'intention' in tasks:
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
            if 'intention' in tasks:
                int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
                if model == 'perframe':
                    int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred,
                                                                       train_dict[hyperparameter_group][
                                                                           'testset'].frame_number_list)
                int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
                int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
                performance_model[hyperparameter_group]['intention_accuracy'] = int_acc
                performance_model[hyperparameter_group]['intention_f1'] = int_f1
                performance_model[hyperparameter_group]['intention_y_true'] = int_y_true
                performance_model[hyperparameter_group]['intention_y_pred'] = int_y_pred
                result_str += 'int_acc: %s, int_f1: %s, ' % ("%.2f" % (int_acc * 100), "%.4f" % int_f1)
            if 'attitude' in tasks:
                att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
                if model == 'perframe':
                    att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred,
                                                                       train_dict[hyperparameter_group][
                                                                           'testset'].frame_number_list)
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
                act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
                act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
                performance_model[hyperparameter_group]['action_accuracy'] = act_acc
                performance_model[hyperparameter_group]['action_f1'] = act_f1
                performance_model[hyperparameter_group]['action_y_true'] = act_y_true
                performance_model[hyperparameter_group]['action_y_pred'] = act_y_pred
                result_str += 'act_acc: %s, act_f1: %s, ' % ("%.2f" % (act_acc * 100), "%.4f" % act_f1)
            print(result_str + 'process_time_pre_frame: %.2f' % (
                    process_time * 1000 / len(testset) / (video_len * sample_fps)))
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # draw_training_process(trainging_process)
        return performance_model


if __name__ == '__main__':
    # model = 'avg'
    # model = 'perframe'
    # model = 'conv1d'
    # model = 'lstm'
    # model = 'gcn_conv1d'
    # model = 'gcn_lstm'
    # model = 'gcn_gcn'
    model = 'stgcn'
    body_part = [True, True, True]

    # framework = 'intention'
    # framework = 'attitude'
    # framework = 'action'
    framework = 'parallel'
    # framework = 'tree'
    # framework = 'chain'
    ori_video = False
    sample_fps = 30
    video_len = 2
    performance_model = []
    i = 0
    while i < 1:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if video_len:
            p_m = train(model=model, body_part=body_part, framework=framework, sample_fps=sample_fps,
                        ori_videos=ori_video, video_len=video_len)
        else:
            p_m = train(model=model, body_part=body_part, framework=framework, sample_fps=sample_fps,
                        ori_videos=ori_video)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save('topk', performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sample_fps: %d, video_len: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sample_fps, str(video_len))
    print(result_str)
    send_email(result_str)
