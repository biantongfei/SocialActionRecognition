import math
import numpy as np

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch_geometric.nn import GCN, GAT, GIN, global_mean_pool, TopKPooling, SAGPooling, ASAPooling

from Dataset import get_inputs_size, coco_body_point_num, halpe_body_point_num, head_point_num, hands_point_num
from graph import Graph, ConvTemporalGraphical
from constants import video_fps, intention_class, attitude_classes, action_classes, device, dtype

intention_class_num = len(intention_class)
attitude_class_num = len(attitude_classes)
action_class_num = len(action_classes)


class DNN(nn.Module):
    def __init__(self, is_coco, body_part, framework):
        super(DNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.framework = framework
        self.input_size = get_inputs_size(is_coco, body_part)
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(64, intention_class_num)
                                            )

        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64),
                                               nn.ReLU(),
                                               nn.Linear(64, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64),
                                             nn.ReLU(),
                                             nn.Linear(64, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num + attitude_class_num, action_class_num)
                                             )

    def forward(self, x):
        y = self.fc(x)
        if self.framework in ['intention', 'attitude', 'action']:
            if self.framework == 'intention':
                y = self.intention_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intention_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


class RNN(nn.Module):
    def __init__(self, is_coco, body_part, framework, gru=False):
        super(RNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.framework = framework
        self.input_size = get_inputs_size(is_coco, body_part)
        self.hidden_size = 256
        self.gru = gru
        if gru:
            self.rnn = nn.GRU(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=True,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=True,
                               batch_first=True)
            # self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3,
            #                    bidirectional=bidirectional, dropout=0.5, batch_first=True)
        self.lstm_attention = nn.Linear(self.hidden_size * 2, 1)
        # Readout layer
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(64, intention_class_num)
                                            )

        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(64, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(64, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(64 + intention_class_num),
                nn.ReLU(),
                nn.Linear(64 + intention_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(64 + intention_class_num),
                nn.ReLU(),
                nn.Linear(64 + intention_class_num, action_class_num)
            )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(64 + intention_class_num),
                nn.ReLU(),
                nn.Linear(64 + intention_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(64 + intention_class_num + attitude_class_num),
                nn.ReLU(),
                nn.Linear(64 + intention_class_num + attitude_class_num, action_class_num)
            )

    def forward(self, x):
        on, _ = self.rnn(x)
        on, _ = rnn_utils.pad_packed_sequence(on, batch_first=True)
        on = on.view(on.shape[0], on.shape[1], 2, -1)
        x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
        attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
        x = torch.sum(x * attention_weights, dim=1)
        y = self.fc(x)
        if self.framework in ['intention', 'attitude', 'action']:
            if self.framework == 'intention':
                y = self.intention_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intention_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


class Cnn1D(nn.Module):
    def __init__(self, is_coco, body_part, framework, max_length):
        super(Cnn1D, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.input_size = get_inputs_size(is_coco, body_part)
        self.framework = framework
        self.hidden_dim = 256 * math.ceil(math.ceil(math.ceil(max_length / 3) / 2) / 2)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.5),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(64, intention_class_num)
                                            )

        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(64, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(64, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num + attitude_class_num, action_class_num)
                                             )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.cnn(x)
        x = x.flatten(1)
        y = self.fc(x)
        if self.framework in ['intention', 'attitude', 'action']:
            if self.framework == 'intention':
                y = self.intention_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intention_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


class GNN(nn.Module):
    def __init__(self, is_coco, body_part, framework, model, max_length):
        super(GNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.body_part = body_part
        self.input_size = get_inputs_size(is_coco, body_part)
        self.framework = framework
        self.model = model
        self.max_length = max_length
        self.keypoint_hidden_dim = 64
        self.time_hidden_dim = self.keypoint_hidden_dim * 128
        self.pooling = False
        self.pooling_rate = 0.6 if self.pooling else 1
        if body_part[0]:
            # self.GCN_body = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            self.GCN_body = GAT(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_body = GIN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        if body_part[1]:
            # self.GCN_head = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            self.GCN_head = GAT(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_head = GIN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        if body_part[2]:
            # self.GCN_hand = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            self.GCN_hand = GAT(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_hand = GIN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        self.gcn_attention = nn.Linear(int(self.keypoint_hidden_dim * self.input_size / 3), 1)
        if self.model in ['gcn_lstm', 'gcn_gru']:
            self.time_model = nn.LSTM(math.ceil(self.pooling_rate * self.input_size / 3) * self.keypoint_hidden_dim,
                                      hidden_size=256, num_layers=3, bidirectional=True,
                                      batch_first=True) if self.model == 'gcn_lstm' else nn.GRU(
                math.ceil(self.pooling_rate * self.input_size / 3) * self.keypoint_hidden_dim,
                hidden_size=256, num_layers=3, bidirectional=True, batch_first=True)
            self.fc_input_size = 256 * 2
            self.lstm_attention = nn.Linear(self.fc_input_size, 1)
        elif self.model == 'gcn_conv1d':
            self.time_model = nn.Sequential(
                nn.Conv1d(math.ceil(self.pooling_rate * self.input_size / 3) * self.keypoint_hidden_dim, 64,
                          kernel_size=7, stride=3, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            )
            self.fc_input_size = 256 * math.ceil(math.ceil(math.ceil(max_length / 3) / 2) / 2)
        else:
            self.GCN_time = GCN(
                in_channels=math.ceil(self.pooling_rate * self.input_size / 3) * self.keypoint_hidden_dim,
                hidden_channels=self.keypoint_hidden_dim, num_layers=2)
            # self.GCN_time = GAT(in_channels=int(self.keypoint_hidden_dim * self.input_size / 3),
            #                     hidden_channels=self.keypoint_hidden_dim,
            #                     num_layers=2)
            # self.GCN_time = GIN(in_channels=int(self.keypoint_hidden_dim * self.input_size / 3),
            #                     hidden_channels=self.keypoint_hidden_dim,
            #                     num_layers=2)
            self.pool = TopKPooling(self.keypoint_hidden_dim, ratio=self.pooling_rate)
            self.fc_input_size = int(self.pooling_rate * self.keypoint_hidden_dim * max_length)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(64, intention_class_num)
                                            )
        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(64, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(64, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num + attitude_class_num, action_class_num)
                                             )

    def forward(self, data):
        x_list = []
        if self.body_part[0]:
            x_body, edge_index_body, batch_body = data[0][0].to(dtype=dtype, device=device), data[1][0].to(
                device=device), data[2][0].to(device)
            x_body = self.GCN_body(x=x_body, edge_index=edge_index_body, batch=batch_body).to(dtype=dtype,
                                                                                              device=device)
            if self.pooling:
                x_body, _, _, _, _, _ = self.pool(x_body, edge_index_body)
                # x_t, _, _, _, _ = self.pool(x_t, new_edge_index)
            # x_body = global_mean_pool(x_body, batch_body)
            x_body = x_body.view(-1, self.max_length, self.keypoint_hidden_dim * (
                coco_body_point_num if self.is_coco else halpe_body_point_num))
            x_list.append(x_body)
        if self.body_part[1]:
            x_head, edge_index_head, batch_head = data[0][1].to(dtype=dtype, device=device), data[1][1].to(device), \
                data[2][1].to(device)
            x_head = self.GCN_head(x=x_head, edge_index=edge_index_head, batch=batch_head).to(dtype=dtype,
                                                                                              device=device)
            if self.pooling:
                x_head, _, _, _, _, _ = self.pool(x_head, edge_index_head)
                # x_t, _, _, _, _ = self.pool(x_t, new_edge_index)
            # x_head = global_mean_pool(x_head, batch_head)
            x_head = x_head.view(-1, self.max_length, self.keypoint_hidden_dim * head_point_num)
            x_list.append(x_head)
        if self.body_part[2]:
            x_hand, edge_index_hand, batch_hand = data[0][2].to(dtype=dtype, device=device), data[1][2].to(device), \
                data[2][2].to(device)
            x_hand = self.GCN_hand(x=x_hand, edge_index=edge_index_hand, batch=batch_hand).to(dtype=dtype,
                                                                                              device=device)
            if self.pooling:
                x_hand, _, _, _, _, _ = self.pool(x_hand, edge_index_hand)
                # x_t, _, _, _, _ = self.pool(x_t, new_edge_index)
            # x_hand = global_mean_pool(x_hand, batch_hand)
            x_hand = x_hand.view(-1, self.max_length, self.keypoint_hidden_dim * hands_point_num)
            x_list.append(x_hand)
        x = torch.cat(x_list, dim=2)
        attention_weights = nn.Softmax(dim=1)(self.gcn_attention(x))
        x = x * attention_weights
        if self.model in ['gcn_lstm', 'gcn_gru']:
            on, _ = self.time_model(x)
            on = on.view(on.shape[0], on.shape[1], 2, -1)
            x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
            attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
            x = torch.sum(x * attention_weights, dim=1)
        elif self.model == 'gcn_conv1d':
            x = torch.transpose(x, 1, 2)
            x = self.time_model(x)
            x = x.flatten(1)
        else:
            time_edge_index = torch.tensor(np.array([[i, i + 1] for i in range(self.max_length - 1)]),
                                           dtype=torch.int32, device=device).t().contiguous()
            x_time = torch.zeros((x.shape[0], math.ceil(self.pooling_rate * x.shape[1] * self.keypoint_hidden_dim)),
                                 dtype=dtype, device=device)
            for i in range(x.shape[0]):
                x_t = x[i]
                x_t = self.GCN_time(x=x_t, edge_index=time_edge_index).to(dtype=dtype, device=device)
                # x_t, new_edge_index, _, _, _, _ = self.pool(x_t, new_edge_index)
                if self.pooling:
                    x_t, _, _, _, _, _ = self.pool(x_t, time_edge_index)
                x_time[i] = x_t.view(1, -1)[0]
            x = x_time.flatten(1)
        y = self.fc(x)
        if self.framework in ['intention', 'attitude', 'action']:
            if self.framework == 'intention':
                y = self.intention_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intention_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


def zero(x):
    return 0


def iden(x):
    return x


class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 is_coco,
                 body,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(is_coco=is_coco, body=body)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)).to(device) if data_bn else iden
        print(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0).to(device),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs).to(device),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs).to(device),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs).to(device),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs).to(device),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs).to(device),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs).to(device),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs).to(device),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs).to(device),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs).to(device),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        print(x.shape)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class STGCN(nn.Module):
    def __init__(self, is_coco, body_part, framework):
        super(STGCN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.body_part = body_part
        self.input_size = get_inputs_size(is_coco, body_part)
        self.framework = framework
        graph_cfg = ()
        self.stgcn_list = []
        if self.body_part[0]:
            self.stgcn_body = ST_GCN_18(3, is_coco, 0).to(device)
        if self.body_part[1]:
            self.stgcn_head = ST_GCN_18(3, is_coco, 1).to(device)
        if self.body_part[2]:
            self.stgcn_hand = ST_GCN_18(3, is_coco, 2).to(device)
        self.gcn_attention = nn.Linear(16 * 256, 1)
        # fcn for prediction
        self.fcn = nn.Conv2d(256, 64, kernel_size=1)
        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(64, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(64, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(64 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(64 + intention_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(64 + intention_class_num + attitude_class_num, action_class_num)
                                             )

    def forward(self, x):
        y_list = []
        if self.body_part[0]:
            print(x[0].shape, 'x')
            y = self.stgcn_body(x=x[0].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            print(y.shape, 'y')
            y_list.append(y)
        if self.body_part[1]:
            print(x[1].shape, 'x')
            y = self.stgcn_head(x=x[1].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            print(y.shape, 'y')
            y_list.append(y)
        if self.body_part[2]:
            print(x[2].shape, 'x')
            y = self.stgcn_hand(x=x[2].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            print(y.shape, 'y')
            y_list.append(y)
        y = torch.cat(y_list, dim=0)
        attention_weights = nn.Softmax(dim=1)(self.gcn_attention(x))
        x = x * attention_weights
        y = self.fcn(y).view(y.size(0), -1)
        if self.framework in ['intention', 'attitude', 'action']:
            if self.framework == 'intention':
                y = self.intention_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intention_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3
