from Dataset import JPL_Dataset, get_tra_test_files, ImagesDataset, HARPER_Dataset, split_harper_subsets
from Models import DNN, RNN, Cnn1D, GNN, STGCN, MSGCN, Transformer, DGSTGCN, R3D, Classifier
from draw_utils import draw_training_process, plot_confusion_matrix
from DataLoader import Pose_DataLoader
from constants import dtype, device, avg_batch_size, perframe_batch_size, conv1d_batch_size, rnn_batch_size, \
    gcn_batch_size, stgcn_batch_size, msgcn_batch_size, learning_rate, tran_batch_size, attn_learning_rate, \
    intention_class, attitude_classes, action_classes, dgstgcn_batch_size, r3d_batch_size

import torch
from torch.nn import functional
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import StepLR

import numpy as np
from sklearn.metrics import f1_score, recall_score
import csv
import smtplib
from email.mime.text import MIMEText
from tqdm import tqdm
import time


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


def draw_save(name, performance_model, framework, augment_method=False):
    tasks = [framework] if framework in ['intention', 'attitude', 'action'] else ['intention', 'attitude', 'action']
    with open('plots/%s.csv' % name, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for index, p_m in enumerate(performance_model):
            data = [index + 1]
            if 'intention' in tasks:
                data.append(p_m['intention_accuracy'])
                data.append(p_m['intention_f1'])
                data.append(p_m['intention_confidence_score'])
                if index == 0:
                    int_y_true = p_m['intention_y_true']
                    int_y_pred = p_m['intention_y_pred']
                else:
                    int_y_true = torch.cat((int_y_true, p_m['intention_y_true']), dim=0)
                    int_y_pred = torch.cat((int_y_pred, p_m['intention_y_pred']), dim=0)
            if 'attitude' in tasks:
                data.append(p_m['attitude_accuracy'])
                data.append(p_m['attitude_f1'])
                data.append(p_m['attitude_confidence_score'])
                if index == 0:
                    att_y_true = p_m['attitude_y_true']
                    att_y_pred = p_m['attitude_y_pred']
                else:
                    att_y_true = torch.cat((att_y_true, p_m['attitude_y_true']), dim=0)
                    att_y_pred = torch.cat((att_y_pred, p_m['attitude_y_pred']), dim=0)
            if 'action' in tasks:
                data.append(p_m['action_accuracy'])
                data.append(p_m['action_f1'])
                data.append(p_m['action_confidence_score'])
                if index == 0:
                    act_y_true = p_m['action_y_true']
                    act_y_pred = p_m['action_y_pred']
                else:
                    act_y_true = torch.cat((act_y_true, p_m['action_y_true']), dim=0)
                    act_y_pred = torch.cat((act_y_pred, p_m['action_y_pred']), dim=0)
            if augment_method == 'gen':
                r_int_y_true, r_int_y_pred, r_att_y_true, r_att_y_pred = get_unseen_sample(int_y_true, int_y_pred,
                                                                                           att_y_true, att_y_pred,
                                                                                           act_y_true, augment_method)
                int_recall = recall_score(r_int_y_true, r_int_y_pred, average='macro')
                att_recall = recall_score(r_att_y_true, r_att_y_pred, average='macro')
                data.append(int_recall)
                data.append(att_recall)
            spamwriter.writerow(data)
        csvfile.close()
    if 'intention' in tasks:
        plot_confusion_matrix(int_y_true, int_y_pred, intention_class, sub_name="cm_%s_intention" % name)
    # if 'attitude' in tasks:
    #     plot_confusion_matrix(att_y_true, att_y_pred, attitude_classes, sub_name="cm_%s_attitude" % name)
    if 'action' in tasks:
        plot_confusion_matrix(act_y_true, act_y_pred, action_classes, sub_name="cm_%s_action" % name)


def transform_preframe_result(y_true, y_pred, sequence_length):
    index = 0
    y, y_hat = [], []
    while index < y_true.shape[0]:
        label = int(torch.mean(y_true[index:index + sequence_length]))
        predict = int(torch.mode(y_pred[index:index + sequence_length])[0])
        y.append(label)
        y_hat.append(predict)
        index += sequence_length
    return torch.Tensor(y), torch.Tensor(y_hat)


def filter_not_interacting_sample(att_y_true, att_y_output):
    _, pred = torch.max(att_y_output, dim=1)
    mask = (att_y_true == 2) | (pred == 2)
    att_y_true = att_y_true[mask]
    att_y_output = att_y_output[mask].reshape(-1, att_y_output.size(1))
    return att_y_true, att_y_output


def get_unseen_sample(int_y_true, int_y_pred, att_y_true, att_y_pred, action_y_true):
    indexes = []
    for i in range(action_y_true.shape[0]):
        if action_y_true[i] in [1, 2, 4, 7, 8]:
            indexes.append(i)
    indexes = torch.Tensor(indexes).to(torch.int64)
    int_y_true = torch.index_select(int_y_true, 0, indexes)
    int_y_pred = torch.index_select(int_y_pred, 0, indexes)
    att_y_true = torch.index_select(att_y_true, 0, indexes)
    att_y_pred = torch.index_select(att_y_pred, 0, indexes)
    print(int_y_true)
    print(int_y_pred)
    print(att_y_true)
    print(att_y_pred)
    return int_y_true, int_y_pred, att_y_true, att_y_pred


def find_wrong_cases(int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred, test_files):
    different_indices_int = torch.nonzero(torch.ne(int_y_true, int_y_pred)).squeeze()
    different_indices_att = torch.nonzero(torch.ne(att_y_true, att_y_pred)).squeeze()
    different_indices_act = torch.nonzero(torch.ne(act_y_true, act_y_pred)).squeeze()
    print('Intention:')
    print(different_indices_int.shape)
    for i in range(different_indices_int.shape):
        index = different_indices_int[i]
        print(test_files[index], int_y_true[index], int_y_pred[index])
    print('Attitude:')
    for i in range(different_indices_att.shape):
        index = different_indices_att[i]
        print(test_files[index], att_y_true[index], att_y_pred[index])
    print('Action:')
    for i in range(different_indices_act.shape):
        index = different_indices_act[i]
        print(test_files[index], act_y_true[index], act_y_pred[index])


def train_jpl(wandb, model, body_part, framework, train_epochs, frame_sample_hop, sequence_length=99999,
              ori_videos=False, dataset='mixed+coco', oneshot=False):
    """
    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    # dataset = 'mixed+coco'
    # dataset = 'crop+coco'
    # dataset = 'noise+halpe'
    # dataset = '0+coco'
    tasks = [framework] if framework in ['intention', 'attitude', 'action'] else ['intention', 'attitude', 'action']
    for t in tasks:
        performance_model = {'%s_accuracy' % t: None, '%s_f1' % t: None, '%s_confidence_score' % t: None,
                             '%s_y_true' % t: None, '%s_y_pred' % t: None}
    num_workers = 8
    if model == 'avg':
        batch_size = avg_batch_size
    elif model == 'perframe':
        batch_size = perframe_batch_size
    elif model == 'lstm':
        batch_size = rnn_batch_size
    elif model == 'conv1d':
        batch_size = conv1d_batch_size
    elif model == 'tran':
        batch_size = tran_batch_size
    elif 'gcn_' in model:
        batch_size = gcn_batch_size
    elif model == 'stgcn':
        batch_size = stgcn_batch_size
        num_workers = 1
    elif model == 'msgcn':
        batch_size = msgcn_batch_size
        num_workers = 1
    elif model == 'dgstgcn':
        batch_size = dgstgcn_batch_size
        num_workers = 1
    elif model == 'r3d':
        batch_size = r3d_batch_size

    print('loading data for %s' % dataset)
    augment_method = dataset.split('+')[0]
    is_coco = True if 'coco' in dataset else False
    if model != 'r3d':
        tra_files, val_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco,
                                                              ori_videos=ori_videos)
        trainset = JPL_Dataset(data_files=tra_files, augment_method=augment_method, is_coco=is_coco,
                               body_part=body_part, model=model, frame_sample_hop=frame_sample_hop,
                               sequence_length=sequence_length, subset='train')
        valset = JPL_Dataset(data_files=val_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                             model=model, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length,
                             subset='validation')
        testset = JPL_Dataset(data_files=test_files, augment_method=augment_method, is_coco=is_coco,
                              body_part=body_part, model=model, frame_sample_hop=frame_sample_hop,
                              sequence_length=sequence_length, subset='test')
    else:
        tra_files, val_files, test_files = get_tra_test_files(augment_method='crop', is_coco=is_coco,
                                                              ori_videos=ori_videos)
        trainset = ImagesDataset(data_files=tra_files, frame_sample_hop=frame_sample_hop,
                                 sequence_length=sequence_length)
        valset = ImagesDataset(data_files=val_files, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length)
        testset = ImagesDataset(data_files=test_files, frame_sample_hop=frame_sample_hop,
                                sequence_length=sequence_length)
    print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (len(trainset), len(valset), len(testset)))
    if model in ['avg', 'perframe']:
        net = DNN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'lstm':
        net = RNN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'conv1d':
        net = Cnn1D(is_coco=is_coco, body_part=body_part, framework=framework, sequence_length=sequence_length)
    elif model == 'tran':
        net = Transformer(is_coco=is_coco, body_part=body_part, framework=framework, sequence_length=sequence_length)
    elif 'gcn_' in model:
        net = GNN(is_coco=is_coco, body_part=body_part, framework=framework, model=model,
                  sequence_length=sequence_length, frame_sample_hop=frame_sample_hop)
    elif model == 'stgcn':
        net = STGCN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'msgcn':
        net = MSGCN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'dgstgcn':
        net = DGSTGCN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'r3d':
        net = R3D(framework=framework)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # if 'gcn_' not in model:
    #     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # else:
    #     optimizer = torch.optim.Adam([
    #         {'params': net.other_parameters, 'lr': learning_rate},
    #         {'params': net.attn_parameters, 'lr': attn_learning_rate}
    #     ])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1
    # csv_file = 'plots/attention_weight_log.csv'
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Attentions'])
    #     file.close()
    while True:
        train_loader = Pose_DataLoader(is_coco=is_coco, model=model, dataset=trainset, batch_size=batch_size,
                                       sequence_length=sequence_length, frame_sample_hop=frame_sample_hop,
                                       drop_last=True, shuffle=True, num_workers=num_workers)
        val_loader = Pose_DataLoader(is_coco=is_coco, model=model, dataset=valset, sequence_length=sequence_length,
                                     frame_sample_hop=frame_sample_hop, drop_last=False, batch_size=batch_size,
                                     num_workers=num_workers)
        net.train()
        print('Training')
        progress_bar = tqdm(total=len(train_loader), desc='Progress')
        for data in train_loader:
            progress_bar.update(1)
            if model in ['avg', 'perframe', 'conv1d', 'tran', 'r3d']:
                inputs, (int_labels, att_labels, act_labels) = data
                inputs = inputs.to(dtype=dtype, device=device)
            elif model == 'lstm':
                (inputs, (int_labels, att_labels, act_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs = inputs.to(dtype=dtype, device=device)
            elif 'gcn' in model:
                inputs, (int_labels, att_labels, act_labels) = data
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            if framework == 'intention':
                int_outputs = net(inputs)
                total_loss = functional.cross_entropy(int_outputs, int_labels)
            elif framework == 'attitude':
                att_outputs = net(inputs)
                att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
                total_loss = functional.cross_entropy(att_outputs, att_labels)
            elif framework == 'action':
                act_outputs = net(inputs)
                total_loss = functional.cross_entropy(act_outputs, act_labels)
            else:
                int_outputs, att_outputs, act_outputs = net(inputs)
                # int_outputs, att_outputs, act_outputs, _ = net(inputs)
                loss_1 = functional.cross_entropy(int_outputs, int_labels)
                loss_2 = functional.cross_entropy(att_outputs, att_labels)
                loss_3 = functional.cross_entropy(act_outputs, act_labels)
                total_loss = loss_1 + loss_2 + loss_3
                # losses = [loss_1, loss_2, loss_3]

                # Compute inverse loss weights
                # weights = [1.0 / (loss.item() + epsilon) for loss in losses]
                # weight_sum = sum(weights)
                # weights = [w / weight_sum for w in weights]
                # # Compute weighted loss
                # total_loss = sum(weight * loss for weight, loss in zip(weights, losses))

                # if epoch == 1:
                #     initial_losses = [loss.item() for loss in losses]
                # gradnorm_loss = compute_gradnorm(losses, initial_losses).to(device=device, dtype=dtype)
                # weights = torch.softmax(net.task_weights, dim=0).to(device=device, dtype=dtype)
                # total_loss = sum(weight * loss for weight, loss in zip(weights, losses)) + gradnorm_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        scheduler.step()
        progress_bar.close()
        print('Validating')
        int_y_true, int_y_pred, int_y_score, att_y_true, att_y_pred, att_y_score, act_y_true, act_y_pred, act_y_score = [], [], [], [], [], [], [], [], []
        net.eval()
        for data in val_loader:
            if model in ['avg', 'perframe', 'conv1d', 'tran', 'r3d']:
                inputs, (int_labels, att_labels, act_labels) = data
                inputs = inputs.to(dtype=dtype, device=device)
            elif model == 'lstm':
                (inputs, (int_labels, att_labels, act_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs = inputs.to(dtype=dtype, device=device)
            elif 'gcn' in model:
                inputs, (int_labels, att_labels, act_labels) = data
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.int64, device=device), att_labels.to(
                dtype=torch.int64, device=device), act_labels.to(dtype=torch.int64, device=device)
            if framework == 'intention':
                int_outputs = net(inputs)
            elif framework == 'attitude':
                att_outputs = net(inputs)
            elif framework == 'action':
                act_outputs = net(inputs)
            else:
                int_outputs, att_outputs, act_outputs = net(inputs)
                # int_outputs, att_outputs, act_outputs, _ = net(inputs)
            if 'intention' in tasks:
                int_outputs = torch.softmax(int_outputs, dim=1)
                score, pred = torch.max(int_outputs, dim=1)
                # int_pred = int_outputs.argmax(dim=1)
                int_y_true += int_labels.tolist()
                int_y_pred += pred.tolist()
                int_y_score += score.tolist()
            if 'attitude' in tasks:
                att_outputs = torch.softmax(att_outputs, dim=1)
                att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
                score, pred = torch.max(att_outputs, dim=1)
                # att_pred = att_outputs.argmax(dim=1)
                att_y_true += att_labels.tolist()
                att_y_pred += pred.tolist()
                att_y_score += score.tolist()
            if 'action' in tasks:
                act_outputs = torch.softmax(act_outputs, dim=1)
                score, pred = torch.max(act_outputs, dim=1)
                # act_pred = act_outputs.argmax(dim=1)
                act_y_true += act_labels.tolist()
                act_y_pred += pred.tolist()
                act_y_score += score.tolist()
        result_str = 'model: %s, epoch: %d, ' % (model, epoch)
        if 'intention' in tasks:
            int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
            if model == 'perframe':
                int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred, sequence_length)
            int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
            int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
            int_score = np.mean(int_y_score)
            result_str += 'int_acc: %.2f, int_f1: %.4f, int_confidence_score: %.4f, ' % (
                int_acc * 100, int_f1, int_score)
            wandb.log({'train_int_acc': int_acc, 'train_int_f1': int_f1})
        if 'attitude' in tasks:
            att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
            if model == 'perframe':
                att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, sequence_length)
            att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
            att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
            att_score = np.mean(att_y_score)
            result_str += 'att_acc: %.2f, att_f1: %.4f, att_confidence_score: %.4f, ' % (
                att_acc * 100, att_f1, att_score)
            wandb.log({'train_att_acc': att_acc, 'train_att_f1': att_f1})
        if 'action' in tasks:
            act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
            if model == 'perframe':
                act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, sequence_length)
            act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
            act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
            act_score = np.mean(act_y_score)
            result_str += 'act_acc: %.2f%%, act_f1: %.4f, act_confidence_score: %.4f, ' % (
                act_acc * 100, act_f1, act_score)
            wandb.log({'train_act_acc': act_acc, 'train_act_f1': act_f1})
        print(result_str + 'loss: %.4f' % total_loss)
        torch.cuda.empty_cache()
        if epoch == train_epochs:
            break
        else:
            epoch += 1
            print('------------------------------------------')
            # break

    print('Testing')
    test_loader = Pose_DataLoader(is_coco=is_coco, model=model, dataset=testset, sequence_length=sequence_length,
                                  frame_sample_hop=frame_sample_hop, batch_size=batch_size, drop_last=False,
                                  num_workers=num_workers)
    if oneshot:
        net.train()
        print('Oneshot')
        progress_bar = tqdm(total=len(test_loader), desc='Progress')
        for data in test_loader:
            progress_bar.update(1)
            if model in ['avg', 'perframe', 'conv1d', 'tran', 'r3d']:
                inputs, (int_labels, att_labels, act_labels) = data
                inputs = inputs.to(dtype=dtype, device=device)
            elif model == 'lstm':
                (inputs, (int_labels, att_labels, act_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs = inputs.to(dtype=dtype, device=device)
            elif 'gcn' in model:
                inputs, (int_labels, att_labels, act_labels) = data
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            if framework == 'intention':
                int_outputs = net(inputs)
                total_loss = functional.cross_entropy(int_outputs, int_labels)
            elif framework == 'attitude':
                att_outputs = net(inputs)
                att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
                total_loss = functional.cross_entropy(att_outputs, att_labels)
            elif framework == 'action':
                act_outputs = net(inputs)
                total_loss = functional.cross_entropy(act_outputs, act_labels)
            else:
                int_outputs, att_outputs, act_outputs = net(inputs)
                # int_outputs, att_outputs, act_outputs, _ = net(inputs)
                att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
                loss_1 = functional.cross_entropy(int_outputs, int_labels)
                loss_2 = functional.cross_entropy(att_outputs, att_labels)
                loss_3 = functional.cross_entropy(act_outputs, act_labels)
                total_loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        scheduler.step()
        progress_bar.close()
    int_y_true, int_y_pred, int_y_score, att_y_true, att_y_pred, att_y_score, act_y_true, act_y_pred, act_y_score = [], [], [], [], [], [], [], [], []
    attn_weight = []
    process_time = 0
    net.eval()
    progress_bar = tqdm(total=len(test_loader), desc='Progress')
    for index, data in enumerate(test_loader):
        progress_bar.update(1)
        if index == 0:
            total_params = sum(p.numel() for p in net.parameters())
        start_time = time.time()
        if model in ['avg', 'perframe', 'conv1d', 'tran', 'r3d']:
            inputs, (int_labels, att_labels, act_labels) = data
            inputs = inputs.to(dtype=dtype, device=device)
        elif model == 'lstm':
            (inputs, (int_labels, att_labels, act_labels)), data_length = data
            inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
            inputs = inputs.to(dtype=dtype, device=device)
        elif 'gcn' in model:
            inputs, (int_labels, att_labels, act_labels) = data
        int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
        if framework in ['intention', 'attitude', 'action']:
            if framework == 'intention':
                int_outputs = net(inputs)
            elif framework == 'attitude':
                att_outputs = net(inputs)
            else:
                act_outputs = net(inputs)
        else:
            int_outputs, att_outputs, act_outputs = net(inputs)
            # int_outputs, att_outputs, act_outputs, attention_weight = net(inputs)
            # attn_weight.append(attention_weight)
        process_time += time.time() - start_time
        if 'intention' in tasks:
            int_outputs = torch.softmax(int_outputs, dim=1)
            score, pred = torch.max(int_outputs, dim=1)
            # int_pred = int_outputs.argmax(dim=1)
            int_y_true += int_labels.tolist()
            int_y_pred += pred.tolist()
            int_y_score += score.tolist()
        if 'attitude' in tasks:
            att_outputs = torch.softmax(att_outputs, dim=1)
            att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
            score, pred = torch.max(att_outputs, dim=1)
            # att_pred = att_outputs.argmax(dim=1)
            att_y_true += att_labels.tolist()
            att_y_pred += pred.tolist()
            att_y_score += score.tolist()
        if 'action' in tasks:
            act_outputs = torch.softmax(act_outputs, dim=1)
            score, pred = torch.max(act_outputs, dim=1)
            # act_pred = act_outputs.argmax(dim=1)
            act_y_true += act_labels.tolist()
            act_y_pred += pred.tolist()
            act_y_score += score.tolist()
        torch.cuda.empty_cache()
    progress_bar.close()
    result_str = ''
    if 'intention' in tasks:
        int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
        if model == 'perframe':
            int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred, sequence_length)
        int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
        int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
        int_score = np.mean(int_y_score)
        performance_model['intention_accuracy'] = int_acc
        performance_model['intention_f1'] = int_f1
        performance_model['intention_confidence_score'] = int_score
        performance_model['intention_y_true'] = int_y_true
        performance_model['intention_y_pred'] = int_y_pred
        result_str += 'int_acc: %.2f, int_f1: %.4f, int_confidence_score :%.4f, ' % (int_acc * 100, int_f1, int_score)
        wandb.log({'test_int_acc': int_acc, 'test_int_f1': int_f1})
    if 'attitude' in tasks:
        att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
        if model == 'perframe':
            att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, sequence_length)
        att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
        att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
        att_score = np.mean(att_y_score)
        performance_model['attitude_accuracy'] = att_acc
        performance_model['attitude_f1'] = att_f1
        performance_model['attitude_confidence_score'] = att_score
        performance_model['attitude_y_true'] = att_y_true
        performance_model['attitude_y_pred'] = att_y_pred
        result_str += 'att_acc: %.2f, att_f1: %.4f, att_confidence_score: %.4f, ' % (att_acc * 100, att_f1, att_score)
        wandb.log({'test_att_acc': att_acc, 'test_att_f1': att_f1})
    if 'action' in tasks:
        act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
        if model == 'perframe':
            act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, sequence_length)
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
        act_score = np.mean(act_y_score)
        performance_model['action_accuracy'] = act_acc
        performance_model['action_f1'] = act_f1
        performance_model['action_confidence_score'] = act_score
        performance_model['action_y_true'] = act_y_true
        performance_model['action_y_pred'] = act_y_pred
        result_str += 'act_acc: %.2f, act_f1: %.4f, act_confidence_score: %.4f, ' % (act_acc * 100, act_f1, act_score)
        wandb.log({'test_act_acc': act_acc, 'test_act_f1': act_f1})
    if augment_method not in ['mixed', 'crop', 'noise']:
        r_int_y_true, r_int_y_pred, r_att_y_true, r_att_y_pred = get_unseen_sample(int_y_true, int_y_pred,
                                                                                   att_y_true, att_y_pred,
                                                                                   act_y_true, augment_method)
        int_recall = recall_score(r_int_y_true, r_int_y_pred, average='micro')
        att_recall = recall_score(r_att_y_true, r_att_y_pred, average='micro')
        result_str += 'int_recall: %.2f%%, att_recall: %.2f%%, ' % (int_recall * 100, att_recall * 100)
    print(result_str + 'Params: %d, process_time_pre_sample: %.2f ms' % (
        (total_params, process_time * 1000 / len(testset))))
    # find_wrong_cases(int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred, test_files)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    torch.save(net, 'models/jpl_%s_fps10.pt' % model)
    # send_email(str(attention_weight.itme()))
    # draw_training_process(trainging_process)
    # attn_weight = torch.cat(attn_weight, dim=0)
    # print(attn_weight.shape)
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(attn_weight.tolist())
    #     file.close()
    return performance_model


def train_harper(wandb, train_epochs, model, sequence_length, body_part, pretrained=True, new_classifier=False,
                 train=True):
    data_path = '../HARPER/pose_sequences/'
    tasks = ['intention', 'attitude'] if pretrained and not new_classifier else ['intention', 'attitude', 'action',
                                                                                 'contact']
    for t in tasks:
        performance_model = {'%s_accuracy' % t: None, '%s_f1' % t: None, '%s_confidence_score' % t: None,
                             '%s_y_true' % t: None, '%s_y_pred' % t: None}
    train_files, val_files, test_files = split_harper_subsets(data_path)
    train_dataset = HARPER_Dataset(data_path=data_path, files=train_files, body_part=body_part, sequence_length=10,
                                   train=True)
    val_dataset = HARPER_Dataset(data_path=data_path, files=val_files, body_part=body_part, sequence_length=10)
    test_dataset = HARPER_Dataset(data_path=data_path, files=test_files, body_part=body_part, sequence_length=10)
    print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (
        len(train_dataset), len(val_dataset), len(test_dataset)))

    if pretrained:
        net = torch.load('models/jpl_gcn_lstm_fps10.pt')
        net.sequence_length = sequence_length
    elif model in ['avg', 'perframe']:
        net = DNN(is_coco=True, body_part=[True, True, True], framework='chain+contact')
    elif model == 'lstm':
        net = RNN(is_coco=True, body_part=[True, True, True], framework='chain+contact')
    elif model == 'conv1d':
        net = Cnn1D(is_coco=True, body_part=[True, True, True], framework='chain+contact',
                    sequence_length=sequence_length)
    elif model == 'tran':
        net = Transformer(is_coco=True, body_part=[True, True, True], framework='chain+contact',
                          sequence_length=sequence_length)
    elif 'gcn_' in model:
        net = GNN(is_coco=True, body_part=[True, True, True], framework='chain+contact', model=model,
                  sequence_length=sequence_length, train_classifier=not new_classifier)
    elif model == 'stgcn':
        net = STGCN(is_coco=True, body_part=[True, True, True], framework='chain+contact')
    elif model == 'msgcn':
        net = MSGCN(is_coco=True, body_part=[True, True, True], framework='chain+contact')
    elif model == 'dgstgcn':
        net = DGSTGCN(is_coco=True, body_part=[True, True, True], framework='chain+contact')
    elif model == 'r3d':
        net = R3D(framework='chain+contact')
    net.to(device)
    if new_classifier:
        H_Classifier = Classifier(framework='chain+contact')
        H_Classifier.to(device)
        optimizer = torch.optim.Adam(H_Classifier.parameters(), lr=learning_rate)
        net.train_classifier = False
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1
    while train:
        train_loader = Pose_DataLoader(is_coco=True, model=model, dataset=train_dataset, batch_size=16,
                                       sequence_length=sequence_length, frame_sample_hop=1, drop_last=True,
                                       shuffle=True, num_workers=1, contact=True)
        val_loader = Pose_DataLoader(is_coco=True, model=model, dataset=val_dataset, sequence_length=sequence_length,
                                     frame_sample_hop=1, drop_last=True, batch_size=16, num_workers=1, contact=True)
        net.train()
        print('Training')
        progress_bar = tqdm(total=len(train_loader), desc='Progress')
        for data in train_loader:
            progress_bar.update(1)
            if model in ['avg', 'perframe', 'conv1d', 'tran', 'r3d']:
                inputs, (int_labels, att_labels, act_labels, contact_labels) = data
                inputs = inputs.to(dtype=dtype, device=device)
            elif model == 'lstm':
                (inputs, (int_labels, att_labels, act_labels, contact_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs = inputs.to(dtype=dtype, device=device)
            elif 'gcn' in model:
                inputs, (int_labels, att_labels, act_labels, contact_labels) = data
            int_labels, att_labels, act_labels, contact_labels = int_labels.to(dtype=torch.long,
                                                                               device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device), contact_labels.to(
                dtype=torch.long, device=device)
            if pretrained:
                if new_classifier:
                    int_outputs, att_outputs, act_outputs, contact_outputs = H_Classifier(net(inputs))
                    loss_1 = functional.cross_entropy(int_outputs, int_labels)
                    loss_2 = functional.cross_entropy(att_outputs, att_labels)
                    loss_3 = functional.cross_entropy(act_outputs, act_labels)
                    loss_4 = functional.cross_entropy(contact_outputs, contact_labels)
                    total_loss = loss_1 + loss_2 + loss_3 + loss_4
                else:
                    int_outputs, att_outputs, _ = net(inputs)
                    # int_outputs, att_outputs, act_outputs, _ = net(inputs)
                    loss_1 = functional.cross_entropy(int_outputs, int_labels)
                    loss_2 = functional.cross_entropy(att_outputs, att_labels)
                    # loss_3 = functional.cross_entropy(act_outputs, act_labels)
                    total_loss = loss_1 + loss_2

                    # losses = [loss_1, loss_2, loss_3]
                    # Compute inverse loss weights
                    # weights = [1.0 / (loss.item() + epsilon) for loss in losses]
                    # weight_sum = sum(weights)
                    # weights = [w / weight_sum for w in weights]
                    # # Compute weighted loss
                    # total_loss = sum(weight * loss for weight, loss in zip(weights, losses))
            else:
                int_outputs, att_outputs, act_outputs, contact_outputs = net(inputs)
                loss_1 = functional.cross_entropy(int_outputs, int_labels)
                loss_2 = functional.cross_entropy(att_outputs, att_labels)
                loss_3 = functional.cross_entropy(act_outputs, act_labels)
                loss_4 = functional.cross_entropy(contact_outputs, contact_labels)
                total_loss = loss_1 + loss_2 + loss_3 + loss_4

            # if epoch == 1:
            #     initial_losses = [loss.item() for loss in losses]
            # gradnorm_loss = compute_gradnorm(losses, initial_losses).to(device=device, dtype=dtype)
            # weights = torch.softmax(net.task_weights, dim=0).to(device=device, dtype=dtype)
            # total_loss = sum(weight * loss for weight, loss in zip(weights, losses)) + gradnorm_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        scheduler.step()
        progress_bar.close()
        print('Validating')
        int_y_true, int_y_pred, int_y_score, att_y_true, att_y_pred, att_y_score, act_y_true, act_y_pred, act_y_score, contact_y_true, contact_y_pred, contact_y_score = [], [], [], [], [], [], [], [], [], [], [], []
        net.eval()
        for data in val_loader:
            if model in ['avg', 'perframe', 'conv1d', 'tran', 'r3d']:
                inputs, (int_labels, att_labels, act_labels, contact_labels) = data
                inputs = inputs.to(dtype=dtype, device=device)
            elif model == 'lstm':
                (inputs, (int_labels, att_labels, act_labels, contact_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs = inputs.to(dtype=dtype, device=device)
            elif 'gcn' in model:
                inputs, (int_labels, att_labels, act_labels, contact_labels) = data
            int_labels, att_labels, act_labels, contact_labels = int_labels.to(dtype=torch.int64,
                                                                               device=device), att_labels.to(
                dtype=torch.int64, device=device), act_labels.to(dtype=torch.int64, device=device), contact_labels.to(
                dtype=torch.int64, device=device)
            if pretrained:
                if new_classifier:
                    int_outputs, att_outputs, act_outputs, contact_outputs = H_Classifier(net(inputs))
                else:
                    int_outputs, att_outputs, _ = net(inputs)
            else:
                int_outputs, att_outputs, act_outputs, contact_outputs = net(inputs)
        result_str = 'model: %s, epoch: %d, ' % (model, epoch)
        if 'intention' in tasks:
            int_outputs = torch.softmax(int_outputs, dim=1)
            score, pred = torch.max(int_outputs, dim=1)
            # int_pred = int_outputs.argmax(dim=1)
            int_y_true += int_labels.tolist()
            int_y_pred += pred.tolist()
            int_y_score += score.tolist()
            int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
            if model == 'perframe':
                int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred, sequence_length)
            int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
            int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
            int_score = np.mean(int_y_score)
            result_str += 'int_acc: %.2f, int_f1: %.4f, int_confidence_score: %.4f, ' % (
                int_acc * 100, int_f1, int_score)
            wandb.log({'train_int_acc': int_acc, 'train_int_f1': int_f1})
        if 'attitude' in tasks:
            att_outputs = torch.softmax(att_outputs, dim=1)
            att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
            score, pred = torch.max(att_outputs, dim=1)
            # att_pred = att_outputs.argmax(dim=1)
            att_y_true += att_labels.tolist()
            att_y_pred += pred.tolist()
            att_y_score += score.tolist()
            att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
            if model == 'perframe':
                att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, sequence_length)
            att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
            att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
            att_score = np.mean(att_y_score)
            result_str += 'att_acc: %.2f, att_f1: %.4f, att_confidence_score: %.4f, ' % (
                att_acc * 100, att_f1, att_score)
            wandb.log({'train_att_acc': att_acc, 'train_att_f1': att_f1})
        if 'action' in tasks:
            act_outputs = torch.softmax(act_outputs, dim=1)
            score, pred = torch.max(act_outputs, dim=1)
            # act_pred = act_outputs.argmax(dim=1)
            act_y_true += act_labels.tolist()
            act_y_pred += pred.tolist()
            act_y_score += score.tolist()
            act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
            if model == 'perframe':
                act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, sequence_length)
            act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
            act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
            act_score = np.mean(act_y_score)
            result_str += 'act_acc: %.2f, act_f1: %.4f, act_confidence_score: %.4f, ' % (
                act_acc * 100, act_f1, act_score)
            wandb.log({'train_act_acc': act_acc, 'train_act_f1': act_f1})
        if 'contact' in tasks:
            contact_outputs = torch.softmax(contact_outputs, dim=1)
            score, pred = torch.max(contact_outputs, dim=1)
            # contact_pred = contact_outputs.argmax(dim=1)
            contact_y_true += contact_labels.tolist()
            contact_y_pred += pred.tolist()
            contact_y_score += score.tolist()
            contact_y_true, contact_y_pred = torch.Tensor(contact_y_true), torch.Tensor(contact_y_pred)
            if model == 'perframe':
                contact_y_true, contact_y_pred = transform_preframe_result(contact_y_true, contact_y_pred,
                                                                           sequence_length)
            contact_acc = contact_y_pred.eq(contact_y_true).sum().float().item() / contact_y_pred.size(dim=0)
            contact_f1 = f1_score(contact_y_true, contact_y_pred, average='weighted')
            contact_score = np.mean(contact_y_score)
            result_str += 'contact_acc: %.2f, contact_f1: %.4f, contact_confidence_score: %.4f, ' % (
                contact_acc * 100, contact_f1, contact_score)
            wandb.log({'train_contact_acc': contact_acc, 'train_contact_f1': contact_f1})
        print(result_str + 'loss: %.4f' % total_loss)
        torch.cuda.empty_cache()
        if epoch == 10:
            break
        else:
            epoch += 1
            print('------------------------------------------')
            # break

    print('Testing')
    test_loader = Pose_DataLoader(is_coco=True, model=model, dataset=test_dataset, sequence_length=sequence_length,
                                  frame_sample_hop=1, batch_size=16, drop_last=False, num_workers=1, contact=True)
    int_y_true, int_y_pred, int_y_score, att_y_true, att_y_pred, att_y_score, act_y_true, act_y_pred, act_y_score, contact_y_true, contact_y_pred, contact_y_score = [], [], [], [], [], [], [], [], [], [], [], []
    process_time = 0
    net.eval()
    progress_bar = tqdm(total=len(test_loader), desc='Progress')
    for index, data in enumerate(test_loader):
        progress_bar.update(1)
        if index == 0:
            total_params = sum(p.numel() for p in net.parameters())
        start_time = time.time()
        if model in ['avg', 'perframe', 'conv1d', 'tran', 'r3d']:
            inputs, (int_labels, att_labels, act_labels, contact_labels) = data
            inputs = inputs.to(dtype=dtype, device=device)
        elif model == 'lstm':
            (inputs, (int_labels, att_labels, act_labels, contact_labels)), data_length = data
            inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
            inputs = inputs.to(dtype=dtype, device=device)
        elif 'gcn' in model:
            inputs, (int_labels, att_labels, act_labels, contact_labels) = data
            int_labels, att_labels, act_labels, contact_labels = int_labels.to(dtype=torch.int64,
                                                                               device=device), att_labels.to(
                dtype=torch.int64, device=device), act_labels.to(dtype=torch.int64, device=device), contact_labels.to(
                dtype=torch.int64, device=device)
        if pretrained:
            if new_classifier:
                int_outputs, att_outputs, act_outputs, contact_outputs = H_Classifier(net(inputs))
            else:
                int_outputs, att_outputs, _ = net(inputs)
        else:
            int_outputs, att_outputs, act_outputs, contact_outputs = net(inputs)
            # int_outputs, att_outputs, act_outputs, attention_weight = net(inputs)
            # attn_weight.append(attention_weight)
        process_time += time.time() - start_time
        int_outputs = torch.softmax(int_outputs, dim=1)
        score, pred = torch.max(int_outputs, dim=1)
        # int_pred = int_outputs.argmax(dim=1)
        int_y_true += int_labels.tolist()
        int_y_pred += pred.tolist()
        int_y_score += score.tolist()
        att_outputs = torch.softmax(att_outputs, dim=1)
        att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
        score, pred = torch.max(att_outputs, dim=1)
        # att_pred = att_outputs.argmax(dim=1)
        att_y_true += att_labels.tolist()
        att_y_pred += pred.tolist()
        att_y_score += score.tolist()
        if not pretrained or new_classifier:
            act_outputs = torch.softmax(act_outputs, dim=1)
            score, pred = torch.max(act_outputs, dim=1)
            # act_pred = act_outputs.argmax(dim=1)
            act_y_true += act_labels.tolist()
            act_y_pred += pred.tolist()
            act_y_score += score.tolist()
            contact_outputs = torch.softmax(contact_outputs, dim=1)
            score, pred = torch.max(contact_outputs, dim=1)
            # contact_pred = contact_outputs.argmax(dim=1)
            contact_y_true += contact_labels.tolist()
            contact_y_pred += pred.tolist()
            contact_y_score += score.tolist()
        torch.cuda.empty_cache()
    progress_bar.close()
    result_str = ''
    int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
    if model == 'perframe':
        int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred, sequence_length)
    int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
    int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
    int_score = np.mean(int_y_score)
    performance_model['intention_accuracy'] = int_acc
    performance_model['intention_f1'] = int_f1
    performance_model['intention_confidence_score'] = int_score
    performance_model['intention_y_true'] = int_y_true
    performance_model['intention_y_pred'] = int_y_pred
    result_str += 'int_acc: %.2f, int_f1: %.4f, int_confidence_score :%.4f, ' % (int_acc * 100, int_f1, int_score)
    wandb.log({'test_int_acc': int_acc, 'test_int_f1': int_f1})
    att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
    if model == 'perframe':
        att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, sequence_length)
    att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
    att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
    att_score = np.mean(att_y_score)
    performance_model['attitude_accuracy'] = att_acc
    performance_model['attitude_f1'] = att_f1
    performance_model['attitude_confidence_score'] = att_score
    performance_model['attitude_y_true'] = att_y_true
    performance_model['attitude_y_pred'] = att_y_pred
    result_str += 'att_acc: %.2f, att_f1: %.4f, att_confidence_score: %.4f, ' % (att_acc * 100, att_f1, att_score)
    wandb.log({'test_att_acc': att_acc, 'test_att_f1': att_f1})
    if not pretrained or new_classifier:
        act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
        if model == 'perframe':
            act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, sequence_length)
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
        act_score = np.mean(act_y_score)
        performance_model['action_accuracy'] = act_acc
        performance_model['action_f1'] = act_f1
        performance_model['action_confidence_score'] = act_score
        performance_model['action_y_true'] = act_y_true
        performance_model['action_y_pred'] = act_y_pred
        result_str += 'act_acc: %.2f, act_f1: %.4f, act_confidence_score: %.4f, ' % (act_acc * 100, act_f1, act_score)
        wandb.log({'test_act_acc': act_acc, 'test_act_f1': act_f1})
        contact_y_true, contact_y_pred = torch.Tensor(contact_y_true), torch.Tensor(contact_y_pred)
        if model == 'perframe':
            contact_y_true, contact_y_pred = transform_preframe_result(contact_y_true, contact_y_pred, sequence_length)
        contact_acc = contact_y_pred.eq(contact_y_true).sum().float().item() / contact_y_pred.size(dim=0)
        contact_f1 = f1_score(contact_y_true, contact_y_pred, average='weighted')
        contact_score = np.mean(contact_y_score)
        performance_model['contact_accuracy'] = contact_acc
        performance_model['contact_f1'] = contact_f1
        performance_model['contact_confidence_score'] = contact_score
        performance_model['contact_y_true'] = contact_y_true
        performance_model['contact_y_pred'] = contact_y_pred
        result_str += 'contact_acc: %.2f, contact_f1: %.4f, contact_confidence_score: %.4f, ' % (
            contact_acc * 100, contact_f1, contact_score)
        wandb.log({'test_contact_acc': contact_acc, 'test_contact_f1': contact_f1})
    print(result_str + 'Params: %d, process_time_pre_sample: %.2f ms' % (
        (total_params, process_time * 1000 / len(test_dataset))))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    torch.save(net, 'models/harper_%s_%s_%s.pt' % (
        model, 'pretrained' if pretrained else '', 'new_classifier' if new_classifier else ''))
    return performance_model


if __name__ == '__main__':
    model = 'gcn_lstm'
    sequence_length = 10
    body_part = [True, True, True]
    pretrained = False
    new_classifier = False
    performance_model = []
    i = 0
    while i < 5:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        p_m = train_harper(model=model, body_part=body_part, sequence_length=sequence_length, pretrained=pretrained,
                           new_classifier=new_classifier, train=True)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    result_str = 'model: %s, sequence_length: %d, pretrained: %s, new_classifier: %s' % (
        model, sequence_length, 'pretarined' if pretrained else 'no', 'new' if new_classifier else 'old')
    print(result_str)
    # send_email(result_str)
