import os

from DataLoader import Pose_DataLoader
from constants import device, gcn_batch_size, learning_rate
from Models import GNN, MSGCN
from train_val import filter_not_interacting_sample
from Dataset import get_jpl_dataset

import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import random

teacher_batch_size = 8


def calculate_teacher_outputs(teacher_model, dataset, batch_size, sequence_length, frame_sample_hop):
    for pt_file in os.listdir('./teacher_tensor/'):
        os.system('rm -rf ./teacher_tensor/%s' % pt_file)
    if teacher_model == 'msgcn':
        teacher_dict = torch.load('models/pretrained_jpl_msgcn_fps30.pt')
        teacher_net = MSGCN([True, True, True], 'chain', 16)
        teacher_net.load_state_dict(teacher_dict)
        teacher_net.to(device)
        teacher_net.eval()
    teacher_dataloader = Pose_DataLoader(model='msgcn', dataset=dataset, batch_size=batch_size,
                                         sequence_length=sequence_length, frame_sample_hop=frame_sample_hop,
                                         drop_last=False, num_workers=1)
    print('Loading teacher outputs')
    progress_bar = tqdm(total=len(teacher_dataloader), desc='Progress')
    for index, data in enumerate(teacher_dataloader):
        inputs, _ = data
        int_outputs, att_outputs, act_outputs = teacher_net(inputs)
        torch.save(int_outputs, "./teacher_tensor/teacher_int_outputs_%d.pt" % index)
        torch.save(att_outputs, "./teacher_tensor/teacher_att_outputs_%d.pt" % index)
        torch.save(act_outputs, "./teacher_tensor/teacher_act_outputs_%d.pt" % index)
        progress_bar.update(1)
    torch.cuda.empty_cache()
    progress_bar.close()


def load_teacher_outputs(index, student_batch_size):
    start_index = int(index * student_batch_size / teacher_batch_size)
    for i in range(int(student_batch_size / teacher_batch_size)):
        if i == 0:
            teacher_int_outputs = torch.load('./teacher_tensor/teacher_int_outputs_%d.pt' % (start_index + i))
            teacher_att_outputs = torch.load('./teacher_tensor/teacher_att_outputs_%d.pt' % (start_index + i))
            teacher_act_outputs = torch.load('./teacher_tensor/teacher_act_outputs_%d.pt' % (start_index + i))
        else:
            try:
                teacher_int_outputs = torch.cat(
                    (teacher_int_outputs, torch.load('./teacher_tensor/teacher_int_outputs_%d.pt' % (start_index + i))),
                    0)
                teacher_att_outputs = torch.cat(
                    (teacher_att_outputs, torch.load('./teacher_tensor/teacher_att_outputs_%d.pt' % (start_index + i))),
                    0)
                teacher_act_outputs = torch.cat(
                    (teacher_act_outputs, torch.load('./teacher_tensor/teacher_act_outputs_%d.pt' % (start_index + i))),
                    0)
            except FileNotFoundError:
                break
    return teacher_int_outputs, teacher_att_outputs, teacher_act_outputs


def train_student(student_model, student_trainset, student_valset, student_testset):
    run = wandb.init()
    T = wandb.config.T
    learning_rate = wandb.config.learning_rate
    student_body_part = wandb.config.student_body_part
    student_sequence_length = wandb.config.student_sequence_length
    student_frame_sample_hop = wandb.config.student_frame_sample_hop
    if 'gcn_' in student_model:
        student_net = GNN(body_part=student_body_part, framework='chain', model=student_model,
                          sequence_length=student_sequence_length, frame_sample_hop=student_frame_sample_hop,
                          keypoint_hidden_dim=wandb.config.keypoint_hidden_dim,
                          time_hidden_dim=wandb.config.time_hidden_dim, fc_hidden1=wandb.config.fc_hidden1,
                          fc_hidden2=wandb.config.fc_hidden2, is_harper=False)
        batch_size = gcn_batch_size
    student_net.to(device)
    optimizer = torch.optim.Adam(student_net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1
    student_train_loader = Pose_DataLoader(model='gcn_lstm', dataset=student_trainset, batch_size=batch_size,
                                           sequence_length=student_sequence_length,
                                           frame_sample_hop=student_frame_sample_hop, drop_last=False, shuffle=False,
                                           num_workers=8)
    val_loader = Pose_DataLoader(model='gcn_lstm', dataset=student_valset, batch_size=128,
                                 sequence_length=student_sequence_length, frame_sample_hop=student_frame_sample_hop,
                                 drop_last=False, shuffle=False, num_workers=8)
    int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
    while epoch <= wandb.config.epochs:
        student_net.train()
        print('Training student model')
        progress_bar = tqdm(total=len(student_train_loader), desc='Progress')
        for index, student_data in enumerate(student_train_loader):
            progress_bar.update(1)
            student_inputs, (int_labels, att_labels, act_labels) = student_data
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            teacher_int_outputs, teacher_att_outputs, teacher_act_outputs = load_teacher_outputs(index, batch_size)
            student_int_outputs, student_att_outputs, student_act_outputs = student_net(student_inputs)
            loss_1 = F.cross_entropy(student_int_outputs, int_labels)
            loss_2 = F.cross_entropy(student_att_outputs, att_labels)
            loss_3 = F.cross_entropy(student_act_outputs, act_labels)
            loss_ce = loss_1 + loss_2 + loss_3

            teacher_int_soft = F.softmax(teacher_int_outputs / T, dim=1)
            teacher_att_soft = F.softmax(teacher_att_outputs / T, dim=1)
            teacher_act_soft = F.softmax(teacher_act_outputs / T, dim=1)
            student_int_log_soft = F.log_softmax(student_int_outputs / T, dim=1)
            student_att_log_soft = F.log_softmax(student_att_outputs / T, dim=1)
            student_act_log_soft = F.log_softmax(student_act_outputs / T, dim=1)
            loss_4 = F.kl_div(student_int_log_soft, teacher_int_soft, reduction="batchmean") * (T ** 2)
            loss_5 = F.kl_div(student_att_log_soft, teacher_att_soft, reduction="batchmean") * (T ** 2)
            loss_6 = F.kl_div(student_act_log_soft, teacher_act_soft, reduction="batchmean") * (T ** 2)
            loss_kd = loss_4 + loss_5 + loss_6
            if wandb.config.loss_type == 'sum':
                total_loss = loss_ce + loss_kd
            elif wandb.config.loss_type == 'weighted':
                total_loss = loss_ce + wandb.config.loss_weight * loss_kd
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            int_y_true += int_labels.tolist()
            pred = student_int_outputs.argmax(dim=1)
            int_y_pred += pred.tolist()
            att_labels, student_att_outputs = filter_not_interacting_sample(att_labels, student_att_outputs)
            att_y_true += att_labels.tolist()
            pred = student_att_outputs.argmax(dim=1)
            att_y_pred += pred.tolist()
            act_y_true += act_labels.tolist()
            pred = student_act_outputs.argmax(dim=1)
            act_y_pred += pred.tolist()
        scheduler.step()
        progress_bar.close()
        result_str = 'training result--> student_model: %s, epoch: %d, ' % (student_model, epoch)
        wandb_log = {'epoch': epoch}
        int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
        int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
        int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
        result_str += 'int_acc: %.2f, int_f1: %.2f, ' % (int_acc * 100, int_f1 * 100)
        wandb_log['train_int_acc'] = int_acc
        wandb_log['train_int_f1'] = int_f1
        att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
        att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
        att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
        result_str += 'att_acc: %.2f, att_f1: %.2f, ' % (att_acc * 100, att_f1 * 100)
        wandb_log['train_att_acc'] = att_acc
        wandb_log['train_att_f1'] = att_f1
        act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
        result_str += 'act_acc: %.2f, act_f1: %.2f, ' % (act_acc * 100, act_f1 * 100)
        wandb_log['train_act_acc'] = act_acc
        wandb_log['train_act_f1'] = act_f1
        result_str += 'avg_acc: %.2f, avg_f1: %.2f, ' % (
            (int_acc + att_acc + act_acc) * 100 / 3, (int_f1 + att_f1 + act_f1) * 100 / 3)
        print(result_str + 'loss: %.4f' % total_loss)
        print('Validating student model')
        int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
        student_net.eval()
        for data in val_loader:
            inputs, (int_labels, att_labels, act_labels) = data
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.int64, device=device), att_labels.to(
                dtype=torch.int64, device=device), act_labels.to(dtype=torch.int64, device=device)
            int_outputs, att_outputs, act_outputs = student_net(inputs)
            int_outputs = torch.softmax(int_outputs, dim=1)
            pred = int_outputs.argmax(dim=1)
            int_y_true += int_labels.tolist()
            int_y_pred += pred.tolist()
            att_outputs = torch.softmax(att_outputs, dim=1)
            att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
            pred = att_outputs.argmax(dim=1)
            att_y_true += att_labels.tolist()
            att_y_pred += pred.tolist()
            act_outputs = torch.softmax(act_outputs, dim=1)
            pred = act_outputs.argmax(dim=1)
            act_y_true += act_labels.tolist()
            act_y_pred += pred.tolist()
        result_str = 'validating result--> student_model: %s, epoch: %d, ' % (student_model, epoch)
        int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
        int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
        int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
        result_str += 'int_acc: %.2f, int_f1: %.2f, ' % (int_acc * 100, int_f1 * 100)
        wandb_log['val_int_acc'] = int_acc
        wandb_log['val_int_f1'] = int_f1
        att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
        att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
        att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
        result_str += 'att_acc: %.2f, att_f1: %.2f, ' % (att_acc * 100, att_f1 * 100)
        wandb_log['val_att_acc'] = att_acc
        wandb_log['val_att_f1'] = att_f1
        act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
        result_str += 'act_acc: %.2f, act_f1: %.2f, ' % (act_acc * 100, act_f1 * 100)
        wandb_log['val_act_acc'] = act_acc
        wandb_log['val_act_f1'] = act_f1
        result_str += 'avg_acc: %.2f, avg_f1: %.2f, ' % (
            (int_acc + att_acc + act_acc) * 100 / 3, (int_f1 + att_f1 + act_f1) * 100 / 3)
        print(result_str + 'loss: %.4f' % total_loss)
        wandb.log(wandb_log)
        torch.cuda.empty_cache()
        epoch += 1
        print('------------------------------------------')
        # break
    print('Testing student model')
    test_loader = Pose_DataLoader(model=student_model, dataset=student_testset, batch_size=128,
                                  sequence_length=student_sequence_length, frame_sample_hop=student_frame_sample_hop,
                                  drop_last=False, shuffle=False, num_workers=8)
    int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
    student_net.eval()
    progress_bar = tqdm(total=len(test_loader), desc='Progress')
    for index, data in enumerate(test_loader):
        progress_bar.update(1)
        if index == 0:
            total_params = sum(p.numel() for p in student_net.parameters())
        inputs, (int_labels, att_labels, act_labels) = data
        int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
        int_outputs, att_outputs, act_outputs = student_net(inputs)
        int_outputs = torch.softmax(int_outputs, dim=1)
        pred = int_outputs.argmax(dim=1)
        int_y_true += int_labels.tolist()
        int_y_pred += pred.tolist()
        att_outputs = torch.softmax(att_outputs, dim=1)
        att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
        pred = att_outputs.argmax(dim=1)
        att_y_true += att_labels.tolist()
        att_y_pred += pred.tolist()
        act_outputs = torch.softmax(act_outputs, dim=1)
        pred = act_outputs.argmax(dim=1)
        act_y_true += act_labels.tolist()
        act_y_pred += pred.tolist()
        torch.cuda.empty_cache()
    progress_bar.close()
    result_str = ''
    wandb_log = {}
    total_f1, total_acc = 0, 0
    int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
    int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
    int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
    result_str += 'int_acc: %.2f, int_f1: %.2f, ' % (int_acc * 100, int_f1 * 100)
    wandb_log['test_int_acc'] = int_acc
    wandb_log['test_int_f1'] = int_f1
    total_acc += int_acc
    total_f1 += int_f1
    att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
    att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
    att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
    result_str += 'att_acc: %.2f, att_f1: %.2f, ' % (att_acc * 100, att_f1 * 100)
    wandb_log['test_att_acc'] = att_acc
    wandb_log['test_att_f1'] = att_f1
    total_acc += att_acc
    total_f1 += att_f1
    act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
    act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
    act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
    result_str += 'act_acc: %.2f, act_f1: %.2f, ' % (act_acc * 100, act_f1 * 100)
    wandb_log['test_act_acc'] = act_acc
    wandb_log['test_act_f1'] = act_f1
    total_acc += act_acc
    total_f1 += act_f1
    print(result_str + 'Params: %d' % (total_params))
    wandb_log['avg_f1'] = total_f1 / 3
    wandb_log['avg_acc'] = total_acc / 3
    wandb_log['params'] = total_params
    # model_name = 'jpl_t-%s&s-%s_fps%d.pt' % (
    #     teacher_model, student_model, int(teacher_net.sequence_length / teacher_net.frame_sample_hop))
    # torch.save(student_net, 'models/%s' % model_name)
    # artifact = wandb.Artifact(model_name, type="model")
    # artifact.add_file("student_models/%s" % model_name)
    # wandb.log_artifact(artifact)
    wandb.log(wandb_log)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


if __name__ == '__main__':
    randnum = random.randint(0, 100)
    # randnum = 25

    student_body_part = [True, True, True]
    student_frame_sample_hop = 3
    student_sequence_length = 30

    print('Loading data for student with body_part: %s, frame_sample_hop: %d' % (
        str(student_body_part), student_frame_sample_hop))
    student_trainset, student_valset, student_testset = get_jpl_dataset('gcn_lstm', student_body_part,
                                                                        student_frame_sample_hop,
                                                                        student_sequence_length,
                                                                        augment_method='ori', randnum=randnum)

    print('Loading data for teacher')
    teacher_trainset = get_jpl_dataset('msgcn', [True, True, True], 1, 30, augment_method='ori',
                                       subset='train', randnum=randnum, fixed_files=student_trainset.out_files)
    calculate_teacher_outputs('msgcn', teacher_trainset, teacher_batch_size, 30, 1)
    del teacher_trainset


    def train():
        train_student(student_model='gcn_lstm', student_trainset=student_trainset, student_valset=student_valset,
                      student_testset=student_testset)


    sweep_config = {
        # 'method': 'bayes',
        # 'method': 'random',
        'method': 'grid',
        'metric': {
            'name': 'avg_f1',
            'goal': 'maximize',
        },
        'parameters': {
            'epochs': {"values": [40, 50]},
            # 'epochs': {"values": [1]},
            'loss_type': {"values": ['weighted']},
            # 'loss_type': {"values": ['sum']},
            'loss_weight': {'values': [0.5, 1, 2]},
            'T': {'values': [6]},
            # 'T': {'values': [3]},
            'learning_rate': {'values': [1e-2]},
            # 'learning_rate': {'values': [1e-3]},
            'keypoint_hidden_dim': {'values': [16]},
            'time_hidden_dim': {'values': [2, 4]},
            'fc_hidden1': {'values': [64]},
            'fc_hidden2': {'values': [16]},
            'student_body_part': {'values': [student_body_part]},
            'student_frame_sample_hop': {'values': [student_frame_sample_hop]},
            'student_sequence_length': {'values': [student_sequence_length]},
            'times': {'values': [ii for ii in range(10)]},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='MS-SEN_JPL')
    # wandb.agent(sweep_id, function=train, count=20)
    wandb.agent(sweep_id, function=train)
