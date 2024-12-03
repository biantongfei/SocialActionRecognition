from DataLoader import Pose_DataLoader, TeacherDataloader
from constants import device, msgcn_batch_size, gcn_batch_size, learning_rate
from Models import GNN, MSGCN
from train_val import filter_not_interacting_sample, dynamic_weight_average
from Dataset import get_jpl_dataset

import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def train_student(student_model, teacher_dataloader, student_trainset, student_valset, student_testset):
    run = wandb.init()
    T = wandb.config.T
    student_body_part = wandb.config.student_body_part
    student_sequence_length = wandb.config.student_sequence_length
    student_frame_sample_hop = wandb.config.student_frame_sample_hop
    if 'gcn_' in student_model:
        student_net = GNN(body_part=student_body_part, framework='chain', model=student_model,
                          sequence_length=student_sequence_length, frame_sample_hop=student_frame_sample_hop,
                          keypoint_hidden_dim=wandb.config.keypoint_hidden_dim,
                          time_hidden_dim=wandb.config.time_hidden_dim, fc_hidden1=wandb.config.fc_hidden1,
                          fc_hidden2=wandb.config.fc_hidden2)
    student_net.to(device)
    optimizer = torch.optim.Adam(student_net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1

    student_train_loader = Pose_DataLoader(model='gcn_lstm', dataset=student_trainset, batch_size=32,
                                           sequence_length=student_sequence_length,
                                           frame_sample_hop=student_frame_sample_hop, drop_last=False, shuffle=False,
                                           num_workers=8)
    val_loader = Pose_DataLoader(model='gcn_lstm', dataset=student_valset, batch_size=128,
                                 sequence_length=student_sequence_length, frame_sample_hop=student_frame_sample_hop,
                                 drop_last=False, shuffle=False, num_workers=8)
    prev_losses = [1, 1]
    while epoch <= wandb.config.epochs:
        student_net.train()
        print('Training student model')
        progress_bar = tqdm(total=len(student_train_loader), desc='Progress')
        for teacher_data, student_data in zip(teacher_dataloader, student_train_loader):
            progress_bar.update(1)
            student_inputs, (int_labels, att_labels, act_labels) = student_data
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            teacher_int_logits, teacher_att_logits, teacher_act_logits = teacher_data
            student_int_outputs, student_att_outputs, student_act_outputs = student_net(student_inputs)
            # int_outputs, att_outputs, act_outputs, _ = net(inputs)
            loss_1 = F.cross_entropy(student_int_outputs, int_labels)
            loss_2 = F.cross_entropy(student_att_outputs, att_labels)
            loss_3 = F.cross_entropy(student_act_outputs, act_labels)
            loss_ce = loss_1 + loss_2 + loss_3

            teacher_int_soft = F.softmax(teacher_int_logits / T, dim=1)
            teacher_att_soft = F.softmax(teacher_att_logits / T, dim=1)
            teacher_act_soft = F.softmax(teacher_act_logits / T, dim=1)
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
                total_loss = loss_ce + 0.5 * loss_kd
            elif wandb.config.loss_type == 'dynamic':
                weights = 1.0 / (torch.tensor([loss_ce, loss_kd]) + 1e-8)
                weights = weights / weights.sum()
                total_loss = weights[0] * loss_ce + weights[1] * loss_kd
            elif wandb.config.loss_type == 'uncertain':
                total_loss = (torch.exp(-student_net.log_sigma1) * loss_ce + student_net.log_sigma1 + torch.exp(
                    -student_net.log_sigma2) * loss_kd + student_net.log_sigma2)
            elif wandb.config.loss_type == 'pareto':
                optimizer.zero_grad()
                loss_ce.backward(retain_graph=True)  # 保留计算图
                g1 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in student_net.parameters()]
                optimizer.zero_grad()
                loss_kd.backward(retain_graph=True)
                g2 = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in student_net.parameters()]
                # 合并梯度（例如使用简单加权或 MGDA）
                combined_grad = [g1[i] + g2[i] for i in range(len(g1))]
                for i, p in enumerate(student_net.parameters()):
                    p.grad = combined_grad[i]
                total_loss = loss_ce + loss_kd
            elif wandb.config.loss_type == 'dwa':
                weights = dynamic_weight_average(prev_losses, [loss_ce, loss_kd])
                total_loss = weights[0] * loss_ce + weights[1] * loss_kd
                prev_losses = [loss_ce.item(), loss_kd.item()]
            if wandb.config.loss_type != 'pareto':
                optimizer.zero_grad()
                total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        scheduler.step()
        progress_bar.close()
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
        result_str = 'student_model: %s, epoch: %d, ' % (student_model, epoch)
        wandb_log = {'epoch': epoch}
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
    print('Loading data for teacher')
    teacher_trainset = get_jpl_dataset('msgcn', [True, True, True], 1, 30, augment_method='mixed',
                                       subset='train', randnum=randnum)
    teacher_dataloader = TeacherDataloader('msgcn', teacher_trainset, 32)

    student_body_part = [True, False, False]
    student_frame_sample_hop = 1
    student_sequence_length = 30

    print('Loading data for student with body_part: %s, frame_sample_hop: %d' % (
        str(student_body_part), student_frame_sample_hop))
    student_trainset, student_valset, student_testset = get_jpl_dataset('gcn_lstm', student_body_part,
                                                                        student_frame_sample_hop,
                                                                        student_sequence_length,
                                                                        augment_method='mixed', randnum=randnum)


    def train():
        train_student(student_model='gcn_lstm', teacher_model='msgcn', teacher_dataloader=teacher_dataloader,
                      student_trainset=student_trainset, student_valset=student_valset,
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
            # 'epochs': {"values": [20, 30, 40]},
            'epochs': {"values": [1]},
            'loss_type': {"values": ['sum', 'weighted', 'uncertain']},
            # 'loss_type': {"values": ['sum']},
            'T': {'values': [2, 3, 4]},
            # 'T': {'values': [3]},
            'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
            # 'learning_rate': {'values': [1e-3]},
            'keypoint_hidden_dim': {'values': [16]},
            'time_hidden_dim': {'values': [4]},
            'fc_hidden1': {'values': [64]},
            'fc_hidden2': {'values': [16]},
            'student_body_part': {'values': [student_body_part]},
            'student_frame_sample_hop': {'values': [student_frame_sample_hop]},
            'student_sequence_length': {'values': [student_sequence_length]},
            # 'times': {'values': [ii for ii in range(5)]},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='MS-SEN_JPL_test')
    # wandb.agent(sweep_id, function=train, count=20)
    wandb.agent(sweep_id, function=train)
