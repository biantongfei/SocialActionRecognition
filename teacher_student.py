from DataLoader import Pose_DataLoader
from constants import intention_classes, attitude_classes, action_classes, device, msgcn_batch_size, gcn_batch_size, \
    learning_rate
from Models import GNN
from train_val import filter_not_interacting_sample
from Dataset import get_jpl_dataset

import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import time


def get_teacher_outputs(teacher_model, trainset, T):
    if teacher_model == 'msgcn':
        teacher_net = torch.load('models/jpl_msgcn_fps30.gt')
        batch_size = msgcn_batch_size
    teacher_net.to(device)
    teacher_net.eval()
    print('Getting training result for teacher model')
    train_loader = Pose_DataLoader(model=teacher_model, dataset=trainset, batch_size=batch_size,
                                   sequence_length=teacher_net.sequence_length,
                                   frame_sample_hop=teacher_net.frame_sample_hop, drop_last=False, shuffle=True,
                                   num_workers=1)
    train_int_outputs, train_att_outputs, train_act_outputs = torch.zeros(
        (len(trainset), len(intention_classes))), torch.zeros((len(trainset), len(attitude_classes))), torch.zeros(
        (len(trainset), len(action_classes)))
    for i, data in enumerate(train_loader):
        inputs, _ = data
        int_outputs, att_outputs, act_outputs = teacher_net(inputs)
        int_outputs = F.softmax(int_outputs / T, dim=1)
        att_outputs = F.softmax(att_outputs / T, dim=1)
        act_outputs = F.softmax(act_outputs / T, dim=1)
        train_int_outputs[i * batch_size:i * batch_size + inputs.shape[0]] = int_outputs
        train_att_outputs[i * batch_size:i * batch_size + inputs.shape[0]] = att_outputs
        train_act_outputs[i * batch_size:i * batch_size + inputs.shape[0]] = act_outputs
    return train_int_outputs, train_att_outputs, train_act_outputs


def train_student(wandb, student_model, teacher_model, trainset, valset, testset, teacher_outputs, T):
    if teacher_model == 'msgcn':
        teacher_net = torch.load('models/jpl_msgcn_fps30.gt')
    t_train_int_outputs, t_train_att_outputs, t_train_act_outputs = teacher_outputs
    num_workers = 8
    if 'gcn_' in student_model:
        batch_size = gcn_batch_size
        student_net = GNN(body_part=[True, True, True], framework='chain', model=student_model,
                          sequence_length=teacher_net.sequence_length, frame_sample_hop=teacher_net.frame_sample_hop)
    student_net.to(device)
    optimizer = torch.optim.Adam(student_net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    epoch = 1
    train_loader = Pose_DataLoader(model=student_model, dataset=trainset, batch_size=batch_size,
                                   sequence_length=teacher_net.sequence_length,
                                   frame_sample_hop=teacher_net.frame_sample_hop, drop_last=False, shuffle=True,
                                   num_workers=num_workers)
    val_loader = Pose_DataLoader(model=student_model, dataset=valset, sequence_length=teacher_net.sequence_length,
                                 frame_sample_hop=teacher_net.frame_sample_hop, drop_last=False, shuffle=True,
                                 num_workers=num_workers)
    while True:
        student_net.train()
        print('Training student model')
        progress_bar = tqdm(total=len(train_loader), desc='Progress')
        for data in train_loader:
            progress_bar.update(1)
            inputs, (int_labels, att_labels, act_labels) = data
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            int_outputs, att_outputs, act_outputs = student_net(inputs)
            # int_outputs, att_outputs, act_outputs, _ = net(inputs)
            loss_1 = functional.cross_entropy(int_outputs, int_labels)
            loss_2 = functional.cross_entropy(att_outputs, att_labels)
            loss_3 = functional.cross_entropy(act_outputs, act_labels)
            int_outputs = F.softmax(int_outputs / T, dim=1)
            att_outputs = F.softmax(att_outputs / T, dim=1)
            act_outputs = F.softmax(act_outputs / T, dim=1)
            loss_4 = F.kl_div(int_outputs.log(), t_train_int_outputs, reduction="batchmean")
            loss_5 = F.kl_div(att_outputs.log(), t_train_att_outputs, reduction="batchmean")
            loss_6 = F.kl_div(act_outputs.log(), t_train_act_outputs, reduction="batchmean")
            total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
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
        result_str += 'int_acc: %.2f, int_f1: %.4f, ' % (int_acc * 100, int_f1)
        wandb_log['val_int_acc'] = int_acc
        wandb_log['val_int_f1'] = int_f1
        att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
        att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
        att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
        result_str += 'att_acc: %.2f, att_f1: %.4f, ' % (att_acc * 100, att_f1)
        wandb_log['val_att_acc'] = att_acc
        wandb_log['val_att_f1'] = att_f1
        act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
        act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
        act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
        result_str += 'act_acc: %.2f%%, act_f1: %.4f, ' % (act_acc * 100, act_f1)
        wandb_log['val_act_acc'] = act_acc
        wandb_log['val_act_f1'] = act_f1
        print(result_str + 'loss: %.4f' % total_loss)
        wandb.log(wandb_log)
        torch.cuda.empty_cache()
        if epoch == wandb.config.epochs:
            break
        else:
            epoch += 1
            print('------------------------------------------')
            # break
    print('Testing student model')
    test_loader = Pose_DataLoader(model=student_model, dataset=testset, sequence_length=teacher_net.sequence_length,
                                  frame_sample_hop=teacher_net.frame_sample_hop, batch_size=batch_size, drop_last=False,
                                  num_workers=num_workers)
    int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
    process_time = 0
    student_net.eval()
    progress_bar = tqdm(total=len(test_loader), desc='Progress')
    for index, data in enumerate(test_loader):
        progress_bar.update(1)
        if index == 0:
            total_params = sum(p.numel() for p in student_net.parameters())
        start_time = time.time()
        inputs, (int_labels, att_labels, act_labels) = data
        int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
        int_outputs, att_outputs, act_outputs = student_net(inputs)
        process_time += time.time() - start_time
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
    result_str += 'int_acc: %.2f, int_f1: %.4f, ' % (int_acc * 100, int_f1)
    wandb_log['test_int_acc'] = int_acc
    wandb_log['test_int_f1'] = int_f1
    total_acc += int_acc
    total_f1 += int_f1
    att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
    att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
    att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
    result_str += 'att_acc: %.2f, att_f1: %.4f, ' % (att_acc * 100, att_f1)
    wandb_log['test_int_acc'] = att_acc
    wandb_log['test_int_f1'] = att_f1
    total_acc += att_acc
    total_f1 += att_f1
    act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
    act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
    act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
    result_str += 'act_acc: %.2f, act_f1: %.4f, ' % (act_acc * 100, act_f1)
    wandb_log['test_act_acc'] = act_acc
    wandb_log['test_act_f1'] = act_f1
    total_acc += act_acc
    total_f1 += act_f1
    print(result_str + 'Params: %d, process_time_pre_sample: %.2f ms' % (
        (total_params, process_time * 1000 / len(testset))))
    wandb_log['avg_f1'] = total_f1 / 3
    wandb_log['avg_acc'] = total_acc / 3
    wandb_log['params'] = total_params
    wandb_log['process_time'] = process_time * 1000 / len(testset)
    model_name = 'jpl_t-%s&s-%s_fps%d.pt' % (
        teacher_model, student_model, int(teacher_net.sequence_length / teacher_net.frame_sample_hop))
    torch.save(student_net, 'models/%s' % model_name)
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file("student_models/%s" % model_name)
    wandb.log_artifact(artifact)
    wandb.log(wandb_log)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


if __name__ == '__main__':
    trainset, valset, testset = get_jpl_dataset('gcn_lstm', [True, True, True], 1, 30,
                                                augment_method='mixed')
    T = 2.0
    teacher_outputs = get_teacher_outputs('msgcn', trainset, T)
    train_student(wandb, 'gcn_lstm', 'msgcn', trainset, valset, testset, teacher_outputs, T)
