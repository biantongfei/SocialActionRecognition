import math
import numpy as np

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import GCN, GAT, GIN, EdgeCNN, TopKPooling

from Dataset import get_inputs_size

box_feature_num = 4
intention_class_num = 3
attitude_class_num = 3
action_class_num = 10
fps = 30
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
dtype = torch.float


class DNN(nn.Module):
    def __init__(self, is_coco, body_part, framework):
        super(DNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.framework = framework
        self.input_size = get_inputs_size(is_coco, body_part)
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(16, intention_class_num)
                                            )

        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16),
                                               nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16),
                                             nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intention_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intention_class_num + attitude_class_num, action_class_num)
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
                y1 = self.intention_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intention_head(y)
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
        self.lstm_attention = nn.Linear(self.fc_input_size, 1)
        # Readout layer
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(16, intention_class_num)
                                            )

        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(16 + intention_class_num),
                nn.ReLU(),
                nn.Linear(16 + intention_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(16 + intention_class_num),
                nn.ReLU(),
                nn.Linear(16 + intention_class_num, action_class_num)
            )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(16 + intention_class_num),
                nn.ReLU(),
                nn.Linear(16 + intention_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(16 + intention_class_num + attitude_class_num),
                nn.ReLU(),
                nn.Linear(16 + intention_class_num + attitude_class_num, action_class_num)
            )

    def forward(self, x):
        on, _ = self.rnn(x)
        on = on.reshape(on.shape[0], on.shape[1], 2, -1)
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
                y1 = self.intention_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intention_head(y)
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
        self.hidden_dim = 16 * math.ceil(math.ceil(math.ceil(max_length / 3) / 2) / 2)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 128, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            # nn.Dropout(0.5),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(16, intention_class_num)
                                            )

        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intention_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intention_class_num + attitude_class_num, action_class_num)
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
                y1 = self.intention_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intention_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


class GNN(torch.nn.Module):
    def __init__(self, is_coco, body_part, framework, model, max_length):
        super(GNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.input_size = get_inputs_size(is_coco, body_part, True)
        self.framework = framework
        self.model = model
        self.max_length = max_length
        self.keypoint_hidden_dim = 16
        self.pooling = False
        self.pooling_rate = 0.8 if self.pooling else 1
        if self.model in ['gcn_lstm', 'gcn_conv1d', 'gcn_gcn']:
            # self.GCN_keypoints = GCN(in_channels=2, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_keypoints = GAT(in_channels=2,hidden_channels=self.keypoint_hidden_dim,num_layers=3)
            self.GCN_keypoints = GIN(in_channels=2,hidden_channels=self.keypoint_hidden_dim,num_layers=3)
            # self.GCN_keypoints = EdgeCNN(in_channels=2,hidden_channels=self.keypoint_hidden_dim,num_layers=3)
            self.pool = TopKPooling(self.keypoint_hidden_dim, ratio=self.pooling_rate)
            if self.model == 'gcn_lstm':
                self.time_model = nn.LSTM(math.ceil(self.pooling_rate * self.input_size / 2) * self.keypoint_hidden_dim,
                                          hidden_size=256, num_layers=3, bidirectional=True, batch_first=True)
                # self.time_model = nn.LSTM(69 * 4, hidden_size=256, num_layers=3, bidirectional=True, batch_first=True)
                self.fc_input_size = 256 * 2
                self.lstm_attention = nn.Linear(self.fc_input_size, 1)
            elif self.model == 'gcn_conv1d':
                self.time_model = nn.Sequential(
                    nn.Conv1d(math.ceil(self.pooling_rate * self.input_size / 2) * self.keypoint_hidden_dim, 256,
                              kernel_size=7, stride=3, padding=3),
                    # nn.Conv1d(69 * 4, 256, kernel_size=7, stride=3, padding=3),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                )
                self.fc_input_size = 64 * math.ceil(math.ceil(math.ceil(max_length / 3) / 2) / 2)
            else:
                # self.GCN_time = GCN(
                #     in_channels=math.ceil(self.pooling_rate * self.input_size / 2) * self.keypoint_hidden_dim,
                #     hidden_channels=self.keypoint_hidden_dim, num_layers=2)
                # self.GCN_time = GAT(in_channels=int(self.keypoint_hidden_dim * self.input_size / 2),
                #                     hidden_channels=self.keypoint_hidden_dim,
                #                     num_layers=2)
                self.GCN_time = GIN(in_channels=int(self.keypoint_hidden_dim * self.input_size / 2),
                                    hidden_channels=self.keypoint_hidden_dim,
                                    num_layers=2)
                self.pool = TopKPooling(self.keypoint_hidden_dim, ratio=self.pooling_rate)
                self.fc_input_size = int(self.pooling_rate * self.keypoint_hidden_dim * max_length)
        else:
            self.ST_GCN1 = GCN(in_channels=2, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.ST_GCN1 = GAT(in_channels=2, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.ST_GCN1 = GIN(in_channels=2, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.ST_GCN1 = EdgeCNN(in_channels=2, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.pool = TopKPooling(self.keypoint_hidden_dim, ratio=self.pooling_rate)
            self.fc_input_size = self.keypoint_hidden_dim * (self.input_size / 2) * max_length
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
        )
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(16, intention_class_num)
                                            )
        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intention_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intention_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intention_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intention_class_num + attitude_class_num, action_class_num)
                                             )

    def forward(self, data):
        x, edge_index, edge_attr = data[0], data[1], data[2]
        time_edge_index = torch.tensor(np.array([[i, i + 1] for i in range(self.max_length - 1)]),
                                       dtype=torch.int64).t().contiguous().to(device)
        if self.model != 'st-gcn':
            x_time = torch.zeros(
                (x.shape[0], x.shape[1],
                 math.ceil(self.pooling_rate * self.input_size / 2) * self.keypoint_hidden_dim)).to(dtype).to(device)
            # x_time = torch.zeros((x.shape[0], x.shape[1], 69 * 4)).to(dtype).to(device)
            for i in range(x.shape[0]):
                for ii in range(x.shape[1]):
                    x_t, new_edge_index, edge_attr_t = x[i][ii], edge_index[i][ii], edge_attr[i][ii]
                    x_t = self.GCN_keypoints(x=x_t, edge_index=new_edge_index, edge_attr=edge_attr_t).to(dtype).to(
                        device)
                    # x_t, new_edge_index, _, _, _, _ = self.pool(x_t, new_edge_index)
                    if self.pooling:
                        x_t, _, _, _, _, _ = self.pool(x_t, new_edge_index)
                    x_time[i][ii] = x_t.reshape(1, -1)[0]
            if self.model == 'gcn_lstm':
                on, _ = self.time_model(x_time)
                on = on.reshape(on.shape[0], on.shape[1], 2, -1)
                x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
                attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
                x = torch.sum(x * attention_weights, dim=1)
            elif self.model == 'gcn_conv1d':
                x = torch.transpose(x_time, 1, 2)
                x = self.time_model(x)
                x = x.flatten(1)
            else:
                x = self.GCN_time(x_time, time_edge_index)
                x = x.flatten(1)
        else:
            x = self.ST_GCN1(x=x, edge_index=edge_index, edge_attr=edge_attr)
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
                y1 = self.intention_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intention_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3
