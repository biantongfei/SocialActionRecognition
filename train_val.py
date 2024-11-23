from Dataset import JPL_Dataset, get_tra_test_files, ImagesDataset, HARPER_Dataset, get_jpl_dataset
from Models import DNN, RNN, Cnn1D, GNN, STGCN, MSGCN, Transformer, DGSTGCN, R3D, Classifier
from draw_utils import plot_confusion_matrix
from DataLoader import Pose_DataLoader
from constants import dtype, device, avg_batch_size, perframe_batch_size, conv1d_batch_size, rnn_batch_size, \
    gcn_batch_size, stgcn_batch_size, msgcn_batch_size, learning_rate, tran_batch_size, attn_learning_rate, \
    intention_classes, attitude_classes, action_classes, dgstgcn_batch_size, r3d_batch_size

import torch
from torch.nn import functional
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import StepLR
from torch import nn

import numpy as np
from sklearn.metrics import f1_score
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


def draw_save(name, performance_model, framework):
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
            spamwriter.writerow(data)
        csvfile.close()
    if 'intention' in tasks:
        plot_confusion_matrix(int_y_true, int_y_pred, intention_classes, sub_name="cm_%s_intention" % name)
    # if 'attitude' in tasks:
    #     plot_confusion_matrix(att_y_true, att_y_pred, attitude_classes, sub_name="cm_%s_attitude" % name)
    # if 'action' in tasks:
    #     plot_confusion_matrix(act_y_true, act_y_pred, action_classes, sub_name="cm_%s_action" % name)


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


class UncertaintyWeightingLoss(nn.Module):
    def __init__(self, task_count):
        super(UncertaintyWeightingLoss, self).__init__()
        # 每个任务的不确定性参数 (log_vars 初始化为 0)
        self.log_vars = nn.Parameter(torch.zeros(task_count))

    def forward(self, losses):
        """
        losses: 列表，每个任务的损失值
        返回: 加权后的总损失
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.log_vars[i])  # 不确定性越大，权重越小
            total_loss += weight * loss + self.log_vars[i]  # 包括正则化项
        return total_loss


def pareto_optimization(task_losses, epsilon=0.01):
    weights = torch.ones(len(task_losses), requires_grad=True)  # 初始化权重
    for i, loss in enumerate(task_losses):
        gradient = torch.autograd.grad(loss, weights, retain_graph=True)[0]
        weights[i] += epsilon * gradient
    return weights / weights.sum()


def dynamic_weight_average(prev_losses, curr_losses, temp=2.0):
    # 计算损失比率
    ratios = torch.tensor(curr_losses) / torch.tensor(prev_losses)
    weights = torch.exp(ratios / temp)  # 动态调整
    return weights / weights.sum()


def train_jpl(wandb, model, body_part, framework, frame_sample_hop, sequence_length, trainset, valset, testset):
    if wandb:
        run = wandb.init()
        # print(
        #     'hyperparameters--> fc2: %d, loss_type: %s, times: %d' % (wandb.config.fc_hidden2, wandb.config.loss_type,
        #                                                               wandb.config.times))
    tasks = [framework] if framework in ['intention', 'attitude', 'action'] else ['intention', 'attitude', 'action']
    performance_model = {}
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
    if model in ['avg', 'perframe']:
        net = DNN(body_part=body_part, framework=framework)
    elif model == 'lstm':
        net = RNN(body_part=body_part, framework=framework)
    elif model == 'conv1d':
        net = Cnn1D(body_part=body_part, framework=framework, sequence_length=sequence_length)
    elif model == 'tran':
        net = Transformer(body_part=body_part, framework=framework, sequence_length=sequence_length)
    elif 'gcn_' in model:
        if wandb:
            net = GNN(body_part=body_part, framework=framework, model=model,
                      sequence_length=sequence_length, frame_sample_hop=frame_sample_hop,
                      keypoint_hidden_dim=wandb.config.keypoints_hidden_dim,
                      time_hidden_dim=wandb.config.time_hidden_dim, fc_hidden1=64, fc_hidden2=16)
        else:
            net = GNN(body_part=body_part, framework=framework, model=model, sequence_length=sequence_length,
                      frame_sample_hop=frame_sample_hop, keypoint_hidden_dim=16, time_hidden_dim=2, fc_hidden1=32,
                      fc_hidden2=16)
    elif model == 'stgcn':
        net = STGCN(body_part=body_part, framework=framework)
    elif model == 'msgcn':
        net = MSGCN(body_part=body_part, framework=framework)
    elif model == 'dgstgcn':
        net = DGSTGCN(body_part=body_part, framework=framework)
    elif model == 'r3d':
        net = R3D(framework=framework)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1
    train_loader = Pose_DataLoader(model=model, dataset=trainset, batch_size=batch_size,
                                   sequence_length=sequence_length, frame_sample_hop=frame_sample_hop,
                                   drop_last=True, shuffle=True, num_workers=num_workers)
    val_loader = Pose_DataLoader(model=model, dataset=valset, sequence_length=sequence_length,
                                 frame_sample_hop=frame_sample_hop, drop_last=False, batch_size=batch_size,
                                 num_workers=num_workers)
    while True:
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
                if wandb.config.loss_type == 'sum':
                    total_loss = loss_1 + loss_2 + loss_3
                elif wandb.config.loss_type == 'dynamic':
                    weights = 1.0 / (torch.tensor([loss_1, loss_2, loss_3]) + 1e-8)
                    weights = weights / weights.sum()
                    total_loss = weights[0] * loss_1 + weights[1] * loss_2 + weights[2] * loss_3
                elif wandb.config.loss_type == 'uncertain':
                    total_loss = (torch.exp(-net.log_sigma1) * loss_1 + net.log_sigma1 + torch.exp(
                        -net.log_sigma2) * loss_2 + net.log_sigma2 + torch.exp(
                        -net.log_sigma3) * loss_3 + net.log_sigma3)
                elif wandb.config.loss_type == 'pareto':
                    optimizer.zero_grad()
                    loss_1.backward(retain_graph=True)  # 保留计算图
                    g1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in net.parameters()]
                    optimizer.zero_grad()
                    loss_2.backward(retain_graph=True)
                    g2 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in net.parameters()]
                    optimizer.zero_grad()
                    loss_3.backward(retain_graph=True)
                    g3 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in net.parameters()]

                    # 合并梯度（例如使用简单加权或 MGDA）
                    combined_grad = [g1[i] + g2[i] + g3[i] for i in range(len(g1))]
                    for i, p in enumerate(net.parameters()):
                        p.grad = combined_grad[i]
                    total_loss = loss_1 + loss_2 + loss_3
                elif wandb.config.loss_type == 'dwa':
                    if epoch == 1:
                        prev_losses = [1, 1, 1]
                    weights = dynamic_weight_average(prev_losses, [loss_1, loss_2, loss_3])
                    total_loss = weights[0] * loss_1 + weights[1] * loss_2 + weights[2] * loss_3
                    prev_losses = [loss_1, loss_2, loss_3]
            if wandb.config.loss_type != 'pareto':
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
        wandb_log = {'epoch': epoch}
        if 'intention' in tasks:
            int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
            if model == 'perframe':
                int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred, sequence_length)
            int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
            int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
            int_score = np.mean(int_y_score)
            result_str += 'int_acc: %.2f, int_f1: %.4f, int_confidence_score: %.4f, ' % (
                int_acc * 100, int_f1, int_score)
            wandb_log['val_int_acc'] = int_acc
            wandb_log['val_int_f1'] = int_f1
        if 'attitude' in tasks:
            att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
            if model == 'perframe':
                att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, sequence_length)
            att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
            att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
            att_score = np.mean(att_y_score)
            result_str += 'att_acc: %.2f, att_f1: %.4f, att_confidence_score: %.4f, ' % (
                att_acc * 100, att_f1, att_score)
            wandb_log['val_att_acc'] = att_acc
            wandb_log['val_att_f1'] = att_f1
        if 'action' in tasks:
            act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
            if model == 'perframe':
                act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, sequence_length)
            act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
            act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
            act_score = np.mean(act_y_score)
            result_str += 'act_acc: %.2f%%, act_f1: %.4f, act_confidence_score: %.4f, ' % (
                act_acc * 100, act_f1, act_score)
            wandb_log['val_act_acc'] = act_acc
            wandb_log['val_act_f1'] = act_f1
        print(result_str + 'loss: %.4f' % total_loss)
        if wandb:
            wandb.log(wandb_log)
        torch.cuda.empty_cache()
        if epoch == wandb.config.epochs:
            break
        else:
            epoch += 1
            print('------------------------------------------')
            # break

    print('Testing')
    test_loader = Pose_DataLoader(model=model, dataset=testset, sequence_length=sequence_length,
                                  frame_sample_hop=frame_sample_hop, batch_size=batch_size, drop_last=False,
                                  num_workers=num_workers)
    int_y_true, int_y_pred, int_y_score, att_y_true, att_y_pred, att_y_score, act_y_true, act_y_pred, act_y_score = [], [], [], [], [], [], [], [], []
    process_time = 0
    net.eval()
    progress_bar = tqdm(total=len(test_loader), desc='Progress')
    for index, data in enumerate(test_loader):
        progress_bar.update(1)
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
    wandb_log = {}
    params = sum(p.numel() for p in net.parameters())
    total_f1, total_acc = 0, 0
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
        wandb_log['test_int_acc'] = int_acc
        wandb_log['test_int_f1'] = int_f1
        total_acc += int_acc
        total_f1 += int_f1
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
        wandb_log['test_att_acc'] = att_acc
        wandb_log['test_att_f1'] = att_f1
        total_acc += att_acc
        total_f1 += att_f1
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
        wandb_log['test_act_acc'] = act_acc
        wandb_log['test_act_f1'] = act_f1
        total_acc += act_acc
        total_f1 += act_f1
    print(result_str + 'Params: %d, process_time_pre_sample: %.2f ms' % (params, process_time * 1000 / len(testset)))
    performance_model['params'] = params
    performance_model['latency'] = process_time * 1000 / len(testset)
    wandb_log['avg_f1'] = total_f1 / len(tasks)
    wandb_log['avg_acc'] = total_acc / len(tasks)
    wandb_log['params'] = params
    wandb_log['process_time'] = process_time * 1000 / len(testset)
    model_name = 'jpl_%s_fps%d.pt' % (model, int(sequence_length / frame_sample_hop))
    torch.save(net, 'models/%s' % model_name)
    if wandb:
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file("models/%s" % model_name)
        wandb.log_artifact(artifact)
        wandb.log(wandb_log)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # send_email(str(attention_weight.itme()))
    # attn_weight = torch.cat(attn_weight, dim=0)
    # print(attn_weight.shape)
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(attn_weight.tolist())
    #     file.close()
    return performance_model


def train_harper(wandb, model, sequence_length, trainset, valset, testset):
    run = wandb.init()
    pretrained = wandb.config.pretrained
    new_classifier = wandb.config.new_classifier
    tasks = ['intention', 'attitude'] if pretrained and not new_classifier else ['intention', 'attitude', 'action',
                                                                                 'contact']
    performance_model = {}
    if pretrained:
        print('Loading SocialEgoNet ' + ('without classifier' if new_classifier else 'with classifier'))
        net = torch.load('models/jpl_gcn_lstm_fps10.pt')
        net.sequence_length = sequence_length
        net.frame_sample_hop = 1
    elif model in ['avg', 'perframe']:
        net = DNN(body_part=[True, True, True], framework='chain+contact')
    elif model == 'lstm':
        net = RNN(body_part=[True, True, True], framework='chain+contact')
    elif model == 'conv1d':
        net = Cnn1D(body_part=[True, True, True], framework='chain+contact',
                    sequence_length=sequence_length)
    elif model == 'tran':
        net = Transformer(body_part=[True, True, True], framework='chain+contact',
                          sequence_length=sequence_length)
    elif 'gcn_' in model:
        net = GNN(body_part=[True, True, True], framework='chain+contact', model=model,
                  sequence_length=sequence_length, frame_sample_hop=1, keypoint_hidden_dim=16, time_hidden_dim=4,
                  fc_hidden1=64, fc_hidden2=16, train_classifier=not new_classifier)
    elif model == 'stgcn':
        net = STGCN(body_part=[True, True, True], framework='chain+contact')
    elif model == 'msgcn':
        net = MSGCN(body_part=[True, True, True], framework='chain+contact')
    elif model == 'dgstgcn':
        net = DGSTGCN(body_part=[True, True, True], framework='chain+contact')
    elif model == 'r3d':
        net = R3D(framework='chain+contact')
    net.to(device)
    if new_classifier:
        H_Classifier = Classifier(framework='chain+contact', in_feature_size=16)
        H_Classifier.to(device)
        optimizer = torch.optim.Adam(H_Classifier.parameters(), lr=learning_rate)
        net.train_classifier = False
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1
    while epoch <= wandb.config.epochs:
        train_loader = Pose_DataLoader(model=model, dataset=trainset, batch_size=16, sequence_length=sequence_length,
                                       frame_sample_hop=1, drop_last=False, shuffle=True, num_workers=1, contact=True)
        val_loader = Pose_DataLoader(model=model, dataset=valset, sequence_length=sequence_length, frame_sample_hop=1,
                                     drop_last=False, batch_size=16, num_workers=1, contact=True)
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

            else:
                int_outputs, att_outputs, act_outputs, contact_outputs = net(inputs)
                loss_1 = functional.cross_entropy(int_outputs, int_labels)
                loss_2 = functional.cross_entropy(att_outputs, att_labels)
                loss_3 = functional.cross_entropy(act_outputs, act_labels)
                loss_4 = functional.cross_entropy(contact_outputs, contact_labels)
                total_loss = loss_1 + loss_2 + loss_3 + loss_4

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
        wandb_log = {'epoch': epoch}
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
            wandb_log['val_int_acc'] = int_acc
            wandb_log['val_int_f1'] = int_f1
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
            wandb_log['val_att_acc'] = att_acc
            wandb_log['val_att_f1'] = att_f1
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
            wandb_log['val_act_acc'] = act_acc
            wandb_log['val_act_f1'] = act_f1
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
            wandb_log['val_contact_acc'] = contact_acc
            wandb_log['val_contact_f1'] = contact_f1
        print(result_str + 'loss: %.4f' % total_loss)
        wandb.log(wandb_log)
        torch.cuda.empty_cache()
        epoch += 1
        print('------------------------------------------')
        # break

    print('Testing')
    test_loader = Pose_DataLoader(model=model, dataset=testset, sequence_length=sequence_length, frame_sample_hop=1,
                                  batch_size=16, drop_last=False, num_workers=1, contact=True)
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
    wandb_log = {}
    total_acc, total_f1 = 0, 0
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
    wandb_log['test_int_acc'] = int_acc
    wandb_log['test_int_f1'] = int_f1
    total_acc += int_acc
    total_f1 += int_f1
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
    wandb_log['test_att_acc'] = att_acc
    wandb_log['test_att_f1'] = att_f1
    total_acc += att_acc
    total_f1 += att_f1
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
        wandb_log['test_act_acc'] = act_acc
        wandb_log['test_act_f1'] = act_f1
        total_acc += act_acc
        total_f1 += act_f1
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
        wandb_log['test_contact_acc'] = contact_acc
        wandb_log['test_contact_f1'] = contact_f1
        total_acc += contact_acc
        total_f1 += contact_f1
    print(result_str + 'Params: %d, process_time_pre_sample: %.2f ms' % (
        (total_params, process_time * 1000 / len(testset))))
    wandb_log['avg_acc'] = total_acc / len(tasks)
    wandb_log['avg_f1'] = total_f1 / len(tasks)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    model_name = 'harper_%s_%s_%s.pt' % (
        model, 'pretrained' if pretrained else '', 'new_classifier' if new_classifier else '')
    torch.save(net, 'models/' + model_name)
    if wandb:
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file("models/%s" % model_name)
        wandb.log_artifact(artifact)
        wandb.log(wandb_log)
    return performance_model


if __name__ == '__main__':
    model = 'msgcn'
    # framework = 'intention'
    # framework = 'attitude'
    # framework = 'action'
    # framework = 'parallel'
    # framework = 'tree'
    framework = 'chain'
    ori_video = False
    frame_sample_hop = 1
    sequence_length = 30
    body_part = [True, True, True]
    trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length,
                                                augment_method='mixed', ori_videos=ori_video)
    for i in range(4):
        print(i)
        p_m = train_jpl(wandb=None, model=model, body_part=body_part, framework=framework,
                        sequence_length=sequence_length, frame_sample_hop=frame_sample_hop, trainset=trainset,
                        valset=valset, testset=testset)

    # with open('body_part.csv', 'w', newline='') as csvfile:
    #     spamwriter = csv.writer(csvfile)
    #     for body_part in [[True, False, False], [False, True, False], [False, False, True], [True, True, False],
    #                       [True, False, True], [False, True, True], [True, True, True]]:
    #         spamwriter.writerow([str(body_part)])
    #         trainset, valset, testset = get_jpl_dataset(model, body_part, frame_sample_hop, sequence_length,
    #                                                     augment_method='mixed', ori_videos=ori_video)
    #         for i in range(4):
    #             print('body_part: %s, times: %d' % (str(body_part), i))
    #             p_m = train_jpl(wandb=None, model=model, body_part=body_part, framework=framework,
    #                             sequence_length=sequence_length, frame_sample_hop=frame_sample_hop, trainset=trainset,
    #                             valset=valset, testset=testset)
    #             spamwriter.writerow(
    #                 [i, p_m['intention_accuracy'], p_m['intention_f1'], p_m['attitude_accuracy'], p_m['attitude_f1'],
    #                  p_m['action_accuracy'], p_m['action_f1'], p_m['params'], p_m['latency']])
