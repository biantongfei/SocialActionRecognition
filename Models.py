import math
import numpy as np

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import GATConv, GCNConv

from Dataset import get_inputs_size

box_feature_num = 4
intent_class_num = 3
attitude_class_num = 3
action_class_num = 9
fps = 30
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
dtype = torch.float


class Conv1DAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1DAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # Compute attention scores
        scores = self.conv(x)
        # Apply softmax to compute attention weights
        weights = self.softmax(scores)
        # Apply attention weights to input features
        output = torch.mul(x, weights)
        return output


class FCAttention(nn.Module):
    def __init__(self, input_dim):
        super(FCAttention, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Compute attention scores
        scores = self.fc(x)
        # Apply softmax to compute attention weights
        weights = nn.Softmax(dim=1)(scores)
        # Apply attention weights to input features
        output = torch.mul(x, weights)
        return output


class DNN(nn.Module):
    def __init__(self, is_coco, body_part, data_format, framework):
        super(DNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.framework = framework
        self.input_size = get_inputs_size(is_coco, body_part, data_format)
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
        self.intent_head = nn.Sequential(nn.ReLU(),
                                         nn.Linear(16, intent_class_num)
                                         )

        if self.framework in ['parallel', 'intent', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16),
                                               nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16),
                                             nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intent_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intent_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intent_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + attitude_class_num, action_class_num)
                                             )

    def forward(self, x):
        y = self.fc(x)
        if self.framework in ['intent', 'attitude', 'action']:
            if self.framework == 'intent':
                y = self.intent_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intent_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


class RNN(nn.Module):
    def __init__(self, is_coco, body_part, data_format, framework, bidirectional=False, gru=False):
        super(RNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.framework = framework
        self.input_size = get_inputs_size(is_coco, body_part, data_format)
        self.hidden_size = 256
        self.bidirectional = bidirectional
        self.gru = gru
        if gru:
            self.rnn = nn.GRU(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=bidirectional,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=bidirectional,
                               batch_first=True)
            # self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3,
            #                    bidirectional=bidirectional, dropout=0.5, batch_first=True)

        # Readout layer
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * (2 if bidirectional else 1)),
            nn.Linear(self.hidden_size * (2 if bidirectional else 1), 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.intent_head = nn.Sequential(nn.ReLU(),
                                         nn.Linear(16, intent_class_num)
                                         )

        if self.framework in ['parallel', 'intent', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(16 + intent_class_num),
                nn.ReLU(),
                nn.Linear(16 + intent_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(16 + intent_class_num),
                nn.ReLU(),
                nn.Linear(16 + intent_class_num, action_class_num)
            )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(16 + intent_class_num),
                nn.ReLU(),
                nn.Linear(16 + intent_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(16 + intent_class_num + attitude_class_num),
                nn.ReLU(),
                nn.Linear(16 + intent_class_num + attitude_class_num, action_class_num)
            )

    def forward(self, x):
        if self.gru:
            on, hn = self.rnn(x)
        else:
            on, (hn, _) = self.rnn(x)
        out_pad, out_length = rnn_utils.pad_packed_sequence(on, batch_first=True)
        if self.bidirectional:
            out = torch.zeros(out_pad.data.shape[0], self.hidden_size * 2).to(device)
            out_pad = out_pad.reshape(out_pad.shape[0], out_pad.shape[0], 2, -1)
            for i in range(out_pad.data.shape[0]):
                index = out_length[i] - 1
                out[i] = torch.cat((out_pad.data[i, -1, 0, :], out_pad.data[i, 0, 1, :]), dim=0)
        else:
            out = torch.zeros(out_pad.data.shape[0], self.hidden_size).to(device)
            for i in range(out_pad.data.shape[0]):
                index = out_length[i] - 1
                out[i] = out_pad.data[i, index, :]
        y = self.fc(out)
        if self.framework in ['intent', 'attitude', 'action']:
            if self.framework == 'intent':
                y = self.intent_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intent_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


class Cnn1D(nn.Module):
    def __init__(self, is_coco, body_part, data_format, framework, max_length):
        super(Cnn1D, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.input_size = get_inputs_size(is_coco, body_part, data_format)
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
        self.intent_head = nn.Sequential(nn.ReLU(),
                                         nn.Linear(16, intent_class_num)
                                         )

        if self.framework in ['parallel', 'intent', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intent_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intent_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intent_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intent_class_num + attitude_class_num, action_class_num)
                                             )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.cnn(x)
        x = x.flatten(1)
        y = self.fc(x)
        if self.framework in ['intent', 'attitude', 'action']:
            if self.framework == 'intent':
                y = self.intent_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intent_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3


class GNN(torch.nn.Module):
    def __init__(self, is_coco, body_part, data_format, framework, model, max_length, attention):
        super(GNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        self.input_size = get_inputs_size(is_coco, body_part, data_format)
        self.framework = framework
        self.model = model
        self.max_length = max_length
        self.attention = attention
        self.num_heads = 4
        self.keypoint_hidden_dim = 16
        self.time_hidden_dim = 256
        self.out_channels = 4
        if self.model in ['gnn_keypoint_lstm', 'gnn_keypoint_conv1d']:
            if attention:
                self.GCN1_keypoints = GATConv(2, self.keypoint_hidden_dim, heads=self.num_heads)
                self.GCN2_keypoints = GATConv(self.keypoint_hidden_dim * self.num_heads, self.keypoint_hidden_dim,
                                              heads=self.num_heads)
                self.GCN3_keypoints = GATConv(self.keypoint_hidden_dim * self.num_heads, self.out_channels, heads=1)
            else:
                self.GCN1_keypoints = GCNConv(2, self.keypoint_hidden_dim)
                self.GCN2_keypoints = GCNConv(self.keypoint_hidden_dim, self.keypoint_hidden_dim)
                self.GCN3_keypoints = GCNConv(self.keypoint_hidden_dim, self.out_channels)
            if self.model == 'gnn_keypoint_lstm':
                self.time_model = nn.LSTM(int(self.input_size / 2 * self.out_channels), hidden_size=256, num_layers=3,
                                          bidirectional=True, batch_first=True)
                self.fc_input_size = 256 * 2
                self.lstm_attention = nn.Linear(self.fc_input_size, 1)
            else:
                self.time_model = nn.Sequential(
                    nn.Conv1d(int(self.input_size / 2 * self.out_channels), 256, kernel_size=7, stride=3, padding=3),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                )
                self.conv1d_attention = Conv1DAttention(64, 64)
                self.fc_input_size = 64 * math.ceil(math.ceil(math.ceil(max_length / 3) / 2) / 2)
        if self.model in ['gnn_time', 'gnn2+1d']:
            if attention:
                self.GCN1_time = GATConv(
                    16 if self.model == 'gnn_time' else int(self.input_size / 2 * self.out_channels),
                    self.time_hidden_dim, heads=self.num_heads)
                self.GCN2_time = GATConv(self.time_hidden_dim * self.num_heads, self.time_hidden_dim,
                                         heads=self.num_heads)
                self.GCN3_time = GATConv(self.time_hidden_dim * self.num_heads, self.out_channels, heads=1)
            else:
                self.GCN1_time = GCNConv(
                    16 if self.model == 'gnn_time' else int(self.input_size / 2 * self.out_channels),
                    self.time_hidden_dim)
                self.GCN2_time = GCNConv(self.time_hidden_dim, self.time_hidden_dim)
                self.GCN3_time = GCNConv(self.time_hidden_dim, self.out_channels)
            if self.model == 'gnn_time':
                self.keypoints_fc = nn.Sequential(
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
                self.fc_input_size = 16 * self.out_channels
            else:
                self.fc_input_size = self.max_length * self.out_channels
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
        )
        self.fc_attention = FCAttention(16)

        self.intent_head = nn.Sequential(nn.ReLU(),
                                         nn.Linear(16, intent_class_num)
                                         )
        if self.framework in ['parallel', 'intent', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(16, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(16, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intent_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intent_class_num, action_class_num)
                                             )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num),
                                               nn.ReLU(),
                                               nn.Linear(16 + intent_class_num, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + intent_class_num + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + intent_class_num + attitude_class_num, action_class_num)
                                             )

    def forward(self, data):
        x, edge_index, edge_attr = data[0], data[1], data[2]
        time_edge_index = torch.tensor(np.array([[i, i + 1] for i in range(self.max_length - 1)]),
                                       dtype=torch.long).t().contiguous()
        if self.model != 'gnn_time':
            x_time = torch.zeros((x.shape[0], x.shape[1], int(self.input_size / 2 * self.out_channels))).to(dtype).to(
                device)
            for i in range(x.shape[0]):
                for ii in range(x.shape[1]):
                    x_t, edge_attr_t = x[i][ii], edge_attr[i][ii]
                    x_t = self.GCN1_keypoints(x=x_t, edge_index=edge_index[i][ii]).to(dtype).to(device)
                    # x_t = self.GCN1_keypoints(x=x_t, edge_index=edge_index[i][ii], edge_attr=edge_attr_t).to(dtype).to(
                    #     device)
                    x_t = nn.ReLU()(
                        nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1)).to(device)(
                            x_t))
                    x_t = self.GCN2_keypoints(x=x_t, edge_index=edge_index[i][ii])
                    # x_t = self.GCN2_keypoints(x=x_t, edge_index=edge_index[i][ii], edge_attr=edge_attr_t)
                    x_t = nn.ReLU()(
                        nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1)).to(device)(
                            x_t))
                    x_t = self.GCN3_keypoints(x=x_t, edge_index=edge_index[i][ii])
                    # x_t = self.GCN3_keypoints(x=x_t, edge_index=edge_index[i][ii], edge_attr=edge_attr_t)
                    x_t = nn.ReLU()(nn.BatchNorm1d(self.out_channels).to(device)(x_t))
                    x_time[i][ii] = x_t.reshape(1, -1)[0]
            if self.model == 'gnn_keypoint_lstm':
                on, _ = self.time_model(x_time)
                on = on.reshape(on.shape[0], on.shape[1], 2, -1)
                x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
                if self.attention:
                    attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
                    x = torch.sum(x * attention_weights, dim=1)
            elif self.model == 'gnn_keypoint_conv1d':
                x = torch.transpose(x_time, 1, 2)
                x = self.time_model(x)
                if self.attention:
                    x = self.conv1d_attention(x)
                x = x.flatten(1)
            else:
                x = self.GCN1_time(x, time_edge_index)
                x = nn.ReLU()(nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1))(x))
                x = self.GCN2_time(x, time_edge_index)
                x = nn.ReLU()(nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1))(x))
                x = self.GCN3_time(x, time_edge_index)
                x = nn.ReLU()(nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1))(x))
        else:
            x = self.keypoints_fc(x)
            x = self.GCN1_time(x, time_edge_index)
            x = nn.ReLU()(nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1))(x))
            x = self.GCN2_time(x, time_edge_index)
            x = nn.ReLU()(nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1))(x))
            x = self.GCN3_time(x, time_edge_index)
            x = nn.ReLU()(nn.BatchNorm1d(self.keypoint_hidden_dim * (self.num_heads if self.attention else 1))(x))
        y = self.fc(x)
        if self.attention:
            y = self.fc_attention(y)
        if self.framework in ['intent', 'attitude', 'action']:
            if self.framework == 'intent':
                y = self.intent_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        else:
            y1 = self.intent_head(y)
            if self.framework == 'parallel':
                y2 = self.attitude_head(y)
                y3 = self.action_head(y)
            elif self.framework == 'tree':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1), dim=1))
            elif self.framework == 'chain':
                y1 = self.intent_head(y)
                y2 = self.attitude_head(torch.cat((y, y1), dim=1))
                y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            return y1, y2, y3
