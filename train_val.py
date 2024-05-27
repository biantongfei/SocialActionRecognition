from Dataset import Dataset, get_tra_test_files
from Models import DNN, RNN, Cnn1D, GNN, STGCN, MSGCN
from draw_utils import draw_training_process, plot_confusion_matrix
from DataLoader import JPLDataLoader
from constants import intention_class, attitude_classes, action_classes, dtype, device, avg_batch_size, \
    perframe_batch_size, conv1d_batch_size, rnn_batch_size, gcn_batch_size, stgcn_batch_size, msgcn_batch_size, \
    learning_rate

import torch
from torch.nn import functional
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import StepLR

import numpy as np
from sklearn.metrics import f1_score
import csv
import smtplib
from email.mime.text import MIMEText
from tqdm import tqdm
from thop import profile
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
        plot_confusion_matrix(int_y_true, int_y_pred, intention_class, sub_name="cm_%s_intention" % name)
    if 'attitude' in tasks:
        plot_confusion_matrix(att_y_true, att_y_pred, attitude_classes, sub_name="cm_%s_attitude" % name)
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


def compute_gradnorm(losses, initial_losses, alpha=0.16):
    gradnorm_loss = 0.0
    avg_loss = sum(losses) / len(losses)
    for i, loss in enumerate(losses):
        norm_loss = (loss / initial_losses[i]) ** alpha
        gradnorm_loss += torch.abs(norm_loss - avg_loss)
    return gradnorm_loss


def train(model, body_part, framework, frame_sample_hop, sequence_length=99999, ori_videos=False):
    """
    :param
    action_recognition: 1 for origin 7 classes; 2 for add not interested and interested; False for attitude recognition
    :return:
    """
    dataset = 'mixed+coco'
    # dataset = 'crop+halpe'
    # dataset = 'noise+halpe'
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
    elif 'gcn_' in model:
        batch_size = gcn_batch_size
    elif model == 'stgcn':
        batch_size = stgcn_batch_size
        num_workers = 1
    elif model == 'msgcn':
        batch_size = msgcn_batch_size
        num_workers = 1

    print('loading data for %s' % dataset)
    augment_method = dataset.split('+')[0]
    is_coco = True if 'coco' in dataset else False
    tra_files, val_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco,
                                                          ori_videos=ori_videos)
    trainset = Dataset(data_files=tra_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                       model=model, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length)
    valset = Dataset(data_files=val_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                     model=model, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length)
    testset = Dataset(data_files=test_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                      model=model, frame_sample_hop=frame_sample_hop, sequence_length=sequence_length)
    print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (len(trainset), len(valset), len(testset)))
    if model in ['avg', 'perframe']:
        net = DNN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'lstm':
        net = RNN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'conv1d':
        net = Cnn1D(is_coco=is_coco, body_part=body_part, framework=framework, sequence_length=sequence_length)
    elif 'gcn_' in model:
        net = GNN(is_coco=is_coco, body_part=body_part, framework=framework, model=model,
                  sequence_length=sequence_length)
    elif model == 'stgcn':
        net = STGCN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model == 'msgcn':
        net = MSGCN(is_coco=is_coco, body_part=body_part, framework=framework)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1
    epsilon = 1e-6
    while True:
        train_loader = JPLDataLoader(is_coco=is_coco, model=model, dataset=trainset, batch_size=batch_size,
                                     sequence_length=sequence_length, drop_last=True, shuffle=True,
                                     num_workers=num_workers)
        val_loader = JPLDataLoader(is_coco=is_coco, model=model, dataset=valset, sequence_length=sequence_length,
                                   drop_last=True, batch_size=batch_size, num_workers=num_workers)
        net.train()
        print('Training')
        progress_bar = tqdm(total=len(train_loader), desc='Progress')
        for data in train_loader:
            progress_bar.update(1)
            if model in ['avg', 'perframe', 'conv1d']:
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
                total_loss = functional.cross_entropy(att_outputs, att_labels)
            elif framework == 'action':
                act_outputs = net(inputs)
                total_loss = functional.cross_entropy(act_outputs, act_labels)
            else:
                int_outputs, att_outputs, act_outputs = net(inputs)
                loss_1 = functional.cross_entropy(int_outputs, int_labels)
                loss_2 = functional.cross_entropy(att_outputs, att_labels)
                loss_3 = functional.cross_entropy(act_outputs, act_labels)
                # total_loss = loss_1 + loss_2 + loss_3
                losses = [loss_1, loss_2, loss_3]

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
                #
                total_loss = loss_1 + loss_2 + loss_3
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
            if model in ['avg', 'perframe', 'conv1d']:
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
            if 'intention' in tasks:
                int_outputs = torch.softmax(int_outputs, dim=1)
                score, pred = torch.max(int_outputs, dim=1)
                # int_pred = int_outputs.argmax(dim=1)
                int_y_true += int_labels.tolist()
                int_y_pred += pred.tolist()
                int_y_score += score.tolist()
            if 'attitude' in tasks:
                att_outputs = torch.softmax(att_outputs, dim=1)
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
        if 'attitude' in tasks:
            att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
            if model == 'perframe':
                att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, sequence_length)
            att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
            att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
            att_score = np.mean(att_y_score)
            result_str += 'att_acc: %.2f, att_f1: %.4f, att_confidence_score: %.4f, ' % (
                att_acc * 100, att_f1, att_score)
        if 'action' in tasks:
            act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
            if model == 'perframe':
                act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, sequence_length)
            act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
            act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
            act_score = np.mean(act_y_score)
            result_str += 'act_acc: %.2f%%, act_f1: %.4f, act_confidence_score: %.4f, ' % (
                act_acc * 100, act_f1, act_score)
        print(result_str + 'loss: %.4f' % total_loss)
        torch.cuda.empty_cache()
        if epoch == 50:
            break
        else:
            epoch += 1
            print('------------------------------------------')
            # break

    print('Testing')
    test_loader = JPLDataLoader(is_coco=is_coco, model=model, dataset=testset, sequence_length=sequence_length,
                                batch_size=batch_size, drop_last=False, num_workers=num_workers)
    int_y_true, int_y_pred, int_y_score, att_y_true, att_y_pred, att_y_score, act_y_true, act_y_pred, act_y_score = [], [], [], [], [], [], [], [], []
    process_time = 0
    start_time = time.time()
    net.eval()
    for index, data in enumerate(test_loader):
        if model in ['avg', 'perframe', 'conv1d']:
            inputs, (int_labels, att_labels, act_labels) = data
            inputs = inputs.to(dtype=dtype, device=device)
        elif model == 'lstm':
            (inputs, (int_labels, att_labels, act_labels)), data_length = data
            inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
            inputs = inputs.to(dtype=dtype, device=device)
        elif 'gcn' in model:
            inputs, (int_labels, att_labels, act_labels) = data
        int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
        if index == 0:
            macs, _ = profile(net, inputs=(inputs,), verbose=False)
            MFlops = 1000 * macs * 2.0 / pow(10, 9) / batch_size / sequence_length
        if framework in ['intention', 'attitude', 'action']:
            if framework == 'intention':
                int_outputs = net(inputs)
            elif framework == 'attitude':
                att_outputs = net(inputs)
            else:
                act_outputs = net(inputs)
        else:
            int_outputs, att_outputs, act_outputs = net(inputs)
        if 'intention' in tasks:
            int_outputs = torch.softmax(int_outputs, dim=1)
            score, pred = torch.max(int_outputs, dim=1)
            # int_pred = int_outputs.argmax(dim=1)
            int_y_true += int_labels.tolist()
            int_y_pred += pred.tolist()
            int_y_score += score.tolist()
        if 'attitude' in tasks:
            att_outputs = torch.softmax(att_outputs, dim=1)
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
    end_time = time.time()
    process_time += end_time - start_time
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
    print(result_str + 'Model Size: %.2f MB, process_time_pre_frame: %.3f ms' % (
        (MFlops, process_time * 1000 / len(testset))))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # draw_training_process(trainging_process)
    return performance_model


if __name__ == '__main__':
    model = 'avg'
    # model = 'conv1d'
    # model = 'lstm'
    # model = 'gcn_conv1d'
    # model = 'gcn_lstm'
    # model = 'gcn_gcn'
    # model = 'stgcn'
    # model = 'msgcn'
    body_part = [True, True, True]

    # framework = 'intention'
    # framework = 'attitude'
    # framework = 'action'
    framework = 'parallel'
    # framework = 'tree'
    # framework = 'chain'
    ori_video = False
    frame_sample_hop = 1
    sequence_length = 30
    performance_model = []
    i = 0
    while i < 1:
        print('~~~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~' % i)
        # try:
        if sequence_length:
            p_m = train(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video, sequence_length=sequence_length)
        else:
            p_m = train(model=model, body_part=body_part, framework=framework, frame_sample_hop=frame_sample_hop,
                        ori_videos=ori_video)
        # except ValueError:
        #     continue
        performance_model.append(p_m)
        i += 1
    draw_save(model, performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sequence_length: %d, frame_hop: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sequence_length, frame_sample_hop)
    print(result_str)
    # send_email(result_str)
