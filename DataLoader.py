from torch.utils.data import DataLoader
import torch
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.utils import add_self_loops

from constants import coco_body_point_num, halpe_body_point_num, head_point_num, hands_point_num, coco_body_l_pair, \
    halpe_body_l_pair, coco_head_l_pair, coco_hand_l_pair, device


def rnn_collate_fn(data):
    data.sort(key=lambda feature: feature[0].shape[0], reverse=True)
    x, intention_labels, attitude_labels, action_labels = [], [], [], []
    for d in data:
        x.append(d[0])
        intention_labels.append(d[1][0])
        attitude_labels.append((d[1][1]))
        action_labels.append(d[1][2])
    data_length = [feature.shape[0] for feature in x]
    x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    return (
        x, (torch.Tensor(intention_labels), torch.Tensor(attitude_labels), torch.Tensor(action_labels))), data_length


class JPLDataLoader(DataLoader):
    def __init__(self, is_coco, model, dataset, batch_size, max_length, drop_last=True, shuffle=False):
        super(JPLDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                            drop_last=drop_last)
        if model in ['lstm', 'gru']:
            self.collate_fn = rnn_collate_fn
        elif model == 'conv1d':
            self.collate_fn = self.conv1d_collate_fn
        elif 'gcn_' in model:
            self.collate_fn = self.gcn_collate_fn
        elif model in ['stgcn', 'msgcn']:
            self.collate_fn = self.stgcn_collate_fn
        self.is_coco = is_coco
        self.max_length = max_length
        self.coco_body_l_pair_num = len(coco_body_l_pair)
        self.halpe_body_l_pair_num = len(halpe_body_l_pair)
        self.head_l_pair_num = len(coco_head_l_pair)
        self.hand_l_pair_num = len(coco_hand_l_pair)

    def conv1d_collate_fn(self, data):
        input, int_label, att_label, act_label = None, [], [], []
        for index, d in enumerate(data):
            x = d[0]
            while x.shape[0] < self.max_length:
                x = torch.cat((x, x[-1].reshape((1, x.shape[1]))), dim=0)
            input = x.reshape(1, x.shape[0], x.shape[1]) if index == 0 else torch.cat(
                (input, x.reshape(1, x.shape[0], x.shape[1])), dim=0)
            int_label.append(d[1][0])
            att_label.append(d[1][1])
            act_label.append(d[1][2])
        return input, (torch.Tensor(int_label), torch.Tensor(att_label), torch.Tensor(act_label))

    def gcn_collate_fn(self, data):
        x_tensors_list, edge_index_list, batch = [
            torch.zeros(
                (len(data) * self.max_length * (coco_body_point_num if self.is_coco else halpe_body_point_num), 3)),
            torch.zeros((len(data) * self.max_length * head_point_num, 3)),
            torch.zeros((len(data) * self.max_length * hands_point_num, 3))], [
            torch.zeros((2,
                         len(data) * self.max_length * self.coco_body_l_pair_num if self.is_coco else self.halpe_body_l_pair_num)).to(
                torch.int64), torch.zeros((2, len(data) * self.max_length * self.head_l_pair_num)).to(torch.int64),
            torch.zeros((2, len(data) * self.max_length * self.hand_l_pair_num)).to(torch.int64)], [torch.zeros(
            (len(data) * self.max_length * (coco_body_point_num if self.is_coco else halpe_body_point_num),)).to(
            torch.int64), torch.zeros(len(data) * self.max_length * head_point_num, ).to(torch.int64),
            torch.zeros((len(data) * self.max_length * hands_point_num,)).to(torch.int64)]
        point_nums = [coco_body_point_num if self.is_coco else halpe_body_point_num, head_point_num, hands_point_num]
        edge_nums = [
            2 * self.coco_body_l_pair_num if self.is_coco else self.halpe_body_l_pair_num + coco_body_point_num if self.is_coco else halpe_body_point_num,
            2 * self.head_l_pair_num + head_point_num, 2 * self.hand_l_pair_num + hands_point_num]
        int_label, att_label, act_label = [], [], []
        frame_num = 0
        for d in data:
            for ii in range(self.max_length):
                for i in range(len(d[0])):
                    if i == 0:
                        edge_index = torch.Tensor(coco_body_l_pair if self.is_coco else halpe_body_l_pair).t()
                        point_num = coco_body_point_num if self.is_coco else halpe_body_point_num
                    elif i == 1:
                        edge_index = torch.Tensor(coco_head_l_pair).t() - torch.full((2, len(coco_head_l_pair)),
                                                                                     fill_value=coco_body_point_num)
                        point_num = head_point_num
                    else:
                        edge_index = torch.Tensor(coco_hand_l_pair).t() - torch.full((2, len(coco_hand_l_pair)),
                                                                                     fill_value=head_point_num + coco_body_point_num)
                        point_num = hands_point_num
                    edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
                    edge_index, _ = add_self_loops(edge_index, num_nodes=point_num)
                    x_tensors_list[i][frame_num * point_nums[i]:(frame_num + 1) * point_nums[i]] = d[0][i][ii]
                    edge_index_list[i][:,
                    frame_num * edge_nums[i]:(frame_num + 1) * edge_nums[i]] = (edge_index + torch.full(
                        (2, edge_nums[i]), fill_value=frame_num * point_nums[i])).to(torch.int64)
                    batch[i][frame_num * point_nums[i]:(frame_num + 1) * point_nums[i]] = torch.full((point_nums[i],),
                                                                                                     fill_value=frame_num).to(
                        torch.int64)
                frame_num += 1
            int_label.append(d[1][0])
            att_label.append(d[1][1])
            act_label.append(d[1][2])
        return (x_tensors_list, edge_index_list, batch), (
            torch.Tensor(int_label), torch.Tensor(att_label), torch.Tensor(act_label))

    def stgcn_collate_fn(self, data):
        input, int_label, att_label, act_label = [], [], [], []
        for index, d in enumerate(data):
            if index == 0:
                for i in range(len(d[0])):
                    if type(d[0][i]) != int:
                        input.append(torch.zeros((len(data), 3, d[0][i].shape[1], d[0][i].shape[2], 1)))
                        input[i][0] = torch.Tensor(d[0][i])
            else:
                for i in range(len(d[0])):
                    if type(d[0][i]) != int:
                        input[i][index] = torch.Tensor(d[0][i])
            int_label.append(d[1][0])
            att_label.append(d[1][1])
            act_label.append(d[1][2])
        return input, (torch.Tensor(int_label), torch.Tensor(att_label), torch.Tensor(act_label))
