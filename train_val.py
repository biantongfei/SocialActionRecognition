from Dataset import Dataset, get_tra_test_files
from Models import DNN, RNN, Cnn1D, GNN, STGCN
from draw_utils import draw_training_process, plot_confusion_matrix
from DataLoader import JPLDataLoader
from constants import intention_class, attitude_classes, action_classes, dtype, device, avg_batch_size, \
    perframe_batch_size, conv1d_batch_size, rnn_batch_size, gcn_batch_size, stgcn_batch_size, learning_rate

import torch
from torch.nn import functional
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import StepLR

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
                if index == 0:
                    int_y_true = p_m['intention_y_true']
                    int_y_pred = p_m['intention_y_pred']
                else:
                    int_y_true = torch.cat((int_y_true, p_m['intention_y_true']), dim=0)
                    int_y_pred = torch.cat((int_y_pred, p_m['intention_y_pred']), dim=0)
            if 'attitude' in tasks:
                data.append(p_m['attitude_accuracy'])
                data.append(p_m['attitude_f1'])
                if index == 0:
                    att_y_true = p_m['attitude_y_true']
                    att_y_pred = p_m['attitude_y_pred']
                else:
                    att_y_true = torch.cat((att_y_true, p_m['attitude_y_true']), dim=0)
                    att_y_pred = torch.cat((att_y_pred, p_m['attitude_y_pred']), dim=0)
            if 'action' in tasks:
                data.append(p_m['action_accuracy'])
                data.append(p_m['action_f1'])
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
    # dataset = 'mixed+coco'
    dataset = 'crop+coco'
    # dataset = 'noise+coco'
    tasks = [framework] if framework in ['intention', 'attitude', 'action'] else ['intention', 'attitude', 'action']
    for t in tasks:
        performance_model = {'%s_accuracy' % t: None, '%s_f1' % t: None, '%s_y_true' % t: None, '%s_y_pred' % t: None}

    if model == 'avg':
        batch_size = avg_batch_size
    elif model == 'perframe':
        batch_size = perframe_batch_size
    elif model in ['lstm', 'gru']:
        batch_size = rnn_batch_size
    elif model == 'conv1d':
        batch_size = conv1d_batch_size
    elif 'gcn_' in model:
        batch_size = gcn_batch_size
    elif model == 'stgcn':
        batch_size = stgcn_batch_size

    print('loading data for %s' % dataset)
    augment_method = dataset.split('+')[0]
    is_coco = True if 'coco' in dataset else False
    tra_files, val_files, test_files = get_tra_test_files(augment_method=augment_method, is_coco=is_coco,
                                                          ori_videos=ori_videos)
    trainset = Dataset(data_files=tra_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                       model=model, sample_fps=sample_fps, video_len=video_len)
    valset = Dataset(data_files=val_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                     model=model, sample_fps=sample_fps, video_len=video_len)
    testset = Dataset(data_files=test_files, augment_method=augment_method, is_coco=is_coco, body_part=body_part,
                      model=model, sample_fps=sample_fps, video_len=video_len)
    max_length = max(trainset.max_length, valset.max_length, testset.max_length)
    print('Train_set_size: %d, Validation_set_size: %d, Test_set_size: %d' % (len(trainset), len(valset), len(testset)))
    if model in ['avg', 'perframe']:
        net = DNN(is_coco=is_coco, body_part=body_part, framework=framework)
    elif model in ['lstm', 'gru']:
        net = RNN(is_coco=is_coco, body_part=body_part, framework=framework, gru=model == 'gru')
    elif model == 'conv1d':
        net = Cnn1D(is_coco=is_coco, body_part=body_part, framework=framework, max_length=max_length)
    elif 'gcn_' in model:
        net = GNN(is_coco=is_coco, body_part=body_part, framework=framework, model=model, max_length=max_length)
    elif model == 'stgcn':
        net = STGCN(is_coco=is_coco, body_part=body_part, framework=framework)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    intention_best_f1 = -1
    attitude_best_f1 = -1
    action_best_f1 = -1
    epoch = 1
    while True:
        train_loader = JPLDataLoader(is_coco=is_coco, model=model, dataset=trainset, batch_size=batch_size,
                                     max_length=max_length, drop_last=False, shuffle=True)
        val_loader = JPLDataLoader(is_coco=is_coco, model=model, dataset=valset, max_length=max_length, drop_last=False,
                                   batch_size=batch_size)
        net.train()
        print('Training')
        progress_bar = tqdm(total=len(train_loader), desc='Progress')
        for data in train_loader:
            progress_bar.update(1)
            if model in ['avg', 'perframe', 'conv1d']:
                inputs, (int_labels, att_labels, act_labels) = data
                inputs = inputs.to(dtype=dtype, device=device)
            elif model in ['lstm', 'gru']:
                (inputs, (int_labels, att_labels, act_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs = inputs.to(dtype=dtype, device=device)
            elif 'gcn_' in model:
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
                total_loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        scheduler.step()
        progress_bar.close()
        print('Validating')
        int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
        net.eval()
        for data in val_loader:
            if model in ['avg', 'perframe', 'conv1d']:
                inputs, (int_labels, att_labels, act_labels) = data
                inputs = inputs.to(dtype=dtype, device=device)
            elif model in ['lstm', 'gru']:
                (inputs, (int_labels, att_labels, act_labels)), data_length = data
                inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
                inputs = inputs.to(dtype=dtype, device=device)
            elif 'gcn_' in model:
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
        result_str = 'model: %s, epoch: %d, ' % (model, epoch)
        if 'intention' in tasks:
            int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
            if model == 'perframe':
                int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred, valset.frame_number_list)
            int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
            int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
            result_str += 'int_acc: %.2f, int_f1: %.4f, ' % (int_acc * 100, int_f1)
        if 'attitude' in tasks:
            att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
            if model == 'perframe':
                att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, valset.frame_number_list)
            att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
            att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
            result_str += 'att_acc: %.2f, att_f1: %.4f, ' % (att_acc * 100, att_f1)
        if 'action' in tasks:
            act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
            if model == 'perframe':
                act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, valset.frame_number_list)
            act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
            act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
            result_str += 'act_acc: %.2f%%, act_f1: %.4f, ' % (act_acc * 100, act_f1)
            print(result_str + 'loss: %.4f' % total_loss)
            torch.cuda.empty_cache()
        # if int_f1 < intention_best_f1 and att_f1 < attitude_best_f1 and act_f1 < action_best_f1:
        if epoch == 20:
            break
        else:
            intention_best_f1 = int_f1 if int_f1 > intention_best_f1 else intention_best_f1
            attitude_best_f1 = att_f1 if att_f1 > attitude_best_f1 else attitude_best_f1
            action_best_f1 = act_f1 if act_f1 > action_best_f1 else action_best_f1
            epoch += 1
            print('------------------------------------------')
            # break

    print('Testing')
    test_loader = JPLDataLoader(is_coco=is_coco, model=model, dataset=testset, max_length=max_length,
                                batch_size=batch_size, drop_last=False)
    int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
    process_time = 0
    start_time = time.time()
    net.eval()
    for index, data in enumerate(test_loader):
        if model in ['avg', 'perframe', 'conv1d']:
            inputs, (int_labels, att_labels, act_labels) = data
            inputs = inputs.to(dtype=dtype, device=device)
        elif model in ['lstm', 'gru']:
            (inputs, (int_labels, att_labels, act_labels)), data_length = data
            inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
            inputs = inputs.to(dtype=dtype, device=device)
        elif 'gcn_' in model:
            inputs, (int_labels, att_labels, act_labels) = data
        int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
        if index == 0:
            macs, _ = profile(net, inputs=(inputs,), verbose=False)
            MFlops = 1000 * macs * 2.0 / pow(10, 9) / batch_size
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
        torch.cuda.empty_cache()
    end_time = time.time()
    process_time += end_time - start_time
    result_str = ''
    if 'intention' in tasks:
        int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
        if model == 'perframe':
            int_y_true, int_y_pred = transform_preframe_result(int_y_true, int_y_pred, testset.frame_number_list)
        int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
        int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
        performance_model['intention_accuracy'] = int_acc
        performance_model['intention_f1'] = int_f1
        performance_model['intention_y_true'] = int_y_true
        performance_model['intention_y_pred'] = int_y_pred
        result_str += 'int_acc: %.2f, int_f1: %.4f, ' % (int_acc * 100, int_f1)
    if 'attitude' in tasks:
        att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
        if model == 'perframe':
            att_y_true, att_y_pred = transform_preframe_result(att_y_true, att_y_pred, testset.frame_number_list)
        att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
        att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
        performance_model['attitude_accuracy'] = att_acc
        performance_model['attitude_f1'] = att_f1
        performance_model['attitude_y_true'] = att_y_true
        performance_model['attitude_y_pred'] = att_y_pred
        result_str += 'att_acc: %.2f, att_f1: %.4f, ' % (att_acc * 100, att_f1)
    if 'action' in tasks:
        act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
        if model == 'perframe':
            act_y_true, act_y_pred = transform_preframe_result(act_y_true, act_y_pred, testset.frame_number_list)
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
        performance_model['action_accuracy'] = act_acc
        performance_model['action_f1'] = act_f1
        performance_model['action_y_true'] = act_y_true
        performance_model['action_y_pred'] = act_y_pred
        result_str += 'act_acc: %.2f, act_f1: %.4f, ' % (act_acc * 100, act_f1)
    print(result_str + 'Computational complexity: %.2f MFLOPs, process_time_pre_frame: %.3f' % (
        (MFlops, process_time * 1000 / len(testset) / (video_len * sample_fps))))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # draw_training_process(trainging_process)
    return performance_model


if __name__ == '__main__':
    # model = 'avg'
    # model = 'perframe'
    # model = 'conv1d'
    # model = 'lstm'
    # model = 'gru'
    # model = 'gcn_conv1d'
    model = 'gcn_lstm'
    # model = 'gcn_gru'
    # model = 'gcn_gcn'
    # model = 'stgcn'
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
    draw_save(model, performance_model, framework)
    result_str = 'model: %s, body_part: [%s, %s, %s], framework: %s, sample_fps: %d, video_len: %s' % (
        model, body_part[0], body_part[1], body_part[2], framework, sample_fps, str(video_len))
    print(result_str)
    # send_email(result_str)
