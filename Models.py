import math
import numpy as np

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch_geometric.nn import GCN, TopKPooling
from torch_geometric.utils import add_self_loops
import torchvision.models.video as models

from Dataset import get_inputs_size, coco_body_point_num, head_point_num, hands_point_num, harper_body_point_num
from graph import Graph, ConvTemporalGraphical
from MSG3D.msg3d import Model as MsG3d
from DGSTGCN.dgstgcn import Model as DG_Model
from constants import intention_classes, attitude_classes, jpl_action_classes, device, dtype, attack_class

intention_class_num = len(intention_classes)
attitude_class_num = len(attitude_classes)
action_class_num = len(jpl_action_classes)
attack_class_num = len(attack_class)


class Classifier(nn.Module):
    def __init__(self, framework, in_feature_size=16):
        super(Classifier, self).__init__()
        super().__init__()
        self.framework = framework
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(in_feature_size, intention_class_num)
                                            )

        if self.framework in ['parallel', 'intention', 'attitude', 'action']:
            self.attitude_head = nn.Sequential(nn.ReLU(),
                                               nn.Linear(in_feature_size, attitude_class_num)
                                               )
            self.action_head = nn.Sequential(nn.ReLU(),
                                             nn.Linear(in_feature_size, action_class_num)
                                             )
        elif self.framework == 'tree':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(in_feature_size + intention_class_num),
                nn.ReLU(),
                nn.Linear(in_feature_size + intention_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(in_feature_size + intention_class_num),
                nn.ReLU(),
                nn.Linear(in_feature_size + intention_class_num, action_class_num)
            )
        elif self.framework == 'chain':
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(in_feature_size + intention_class_num),
                nn.ReLU(),
                nn.Linear(in_feature_size + intention_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(in_feature_size + intention_class_num + attitude_class_num),
                nn.ReLU(),
                nn.Linear(in_feature_size + intention_class_num + attitude_class_num, action_class_num)
            )
        elif self.framework == 'chain+contact':
            self.contact_head = nn.Sequential(nn.ReLU(), nn.Linear(in_feature_size, 2))
            self.attitude_head = nn.Sequential(
                nn.BatchNorm1d(in_feature_size + intention_class_num),
                nn.ReLU(),
                nn.Linear(in_feature_size + intention_class_num, attitude_class_num)
            )
            self.action_head = nn.Sequential(
                nn.BatchNorm1d(in_feature_size + intention_class_num + attitude_class_num),
                nn.ReLU(),
                nn.Linear(in_feature_size + intention_class_num + attitude_class_num, action_class_num)
            )

    def forward(self, y):
        if self.framework in ['intention', 'attitude', 'action']:
            if self.framework == 'intention':
                y = self.intention_head(y)
            elif self.framework == 'attitude':
                y = self.attitude_head(y)
            elif self.framework == 'chain':
                y = self.action_head(y)
            return y
        elif self.framework in ['parallel', 'tree', 'chain']:
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
        elif self.framework == 'chain+contact':
            y1 = self.intention_head(y)
            y2 = self.attitude_head(torch.cat((y, y1), dim=1))
            y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
            y4 = self.contact_head(y)
            return y1, y2, y3, y4


class Attack_Classifier(nn.Module):
    def __init__(self, framework, in_feature_size=16):
        super(Attack_Classifier, self).__init__()
        super().__init__()
        self.framework = framework
        self.attack_current_head = nn.Sequential(nn.ReLU(),
                                                 nn.Linear(in_feature_size, attack_class_num)
                                                 )
        if 'parallel' in framework:
            self.attack_future_head = nn.Sequential(nn.ReLU(),
                                                    nn.Linear(in_feature_size, attack_class_num)
                                                    )
        elif 'chain' in framework:
            self.attack_future_head = nn.Sequential(nn.ReLU(),
                                                    nn.Linear(in_feature_size + attack_class_num, attack_class_num)
                                                    )

    def forward(self, y):
        if 'attack' in self.framework:
            y1 = self.attack_current_head(y)
            if 'parallel' in self.framework:
                y2 = self.attack_future_head(y)
            elif 'chain' in self.framework:
                y2 = self.attack_future_head(torch.cat((y, y1), dim=1))
            return y1, y2


class DNN(nn.Module):
    def __init__(self, body_part, framework, train_classifier):
        super(DNN, self).__init__()
        super().__init__()
        self.framework = framework
        self.input_size = get_inputs_size(body_part)
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.classifier = Classifier(framework)

    def forward(self, x):
        y = self.fc(x)
        return self.classifier(y)


class RNN(nn.Module):
    def __init__(self, body_part, framework):
        super(RNN, self).__init__()
        super().__init__()
        self.framework = framework
        self.input_size = get_inputs_size(body_part)
        self.hidden_size = 128
        self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=True,
                           batch_first=True)
        self.lstm_attention = nn.Linear(self.hidden_size * 2, 1)
        # Readout layer
        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
        )
        self.classifier = Classifier(framework)

    def forward(self, x):
        on, _ = self.rnn(x)
        on, _ = rnn_utils.pad_packed_sequence(on, batch_first=True)
        on = on.view(on.shape[0], on.shape[1], 2, -1)
        x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
        attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
        x = torch.sum(x * attention_weights, dim=1)
        y = self.fc(x)
        return self.classifier(y)


class Cnn1D(nn.Module):
    def __init__(self, body_part, framework, sequence_length):
        super(Cnn1D, self).__init__()
        super().__init__()
        self.input_size = get_inputs_size(body_part)
        self.framework = framework
        self.hidden_dim = 256 * math.ceil(math.ceil(sequence_length / 3) / 2)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.BatchNorm1d(512),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            # nn.Dropout(0.5),
        )
        self.classifier = Classifier(framework)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.cnn(x)
        x = x.flatten(1)
        y = self.fc(x)
        return self.classifier(y)


class Transformer(nn.Module):
    def __init__(self, body_part, framework, sequence_length):
        super(Transformer, self).__init__()
        self.input_size = get_inputs_size(body_part)
        self.framework = framework
        model_dim, num_heads, num_layers, num_classes = 512, 8, 3, 16
        self.embedding = nn.Linear(self.input_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.classifier = Classifier(framework)

    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)

        transformer_output = self.transformer_encoder(src)
        transformer_output = transformer_output.mean(dim=0)  # Global average pooling
        y = self.fc(transformer_output)
        return self.classifier(y)


class GNN(nn.Module):
    def __init__(self, body_part, framework, model, sequence_length, frame_sample_hop, keypoint_hidden_dim,
                 time_hidden_dim, fc_hidden1, fc_hidden2, is_harper=False, is_attack=False, train_classifier=True):
        super(GNN, self).__init__()
        super().__init__()
        self.body_part = body_part
        self.input_size = harper_body_point_num * 3 if is_harper else get_inputs_size(body_part)
        self.framework = framework
        self.model = model
        self.sequence_length = sequence_length
        self.frame_sample_hop = frame_sample_hop
        self.keypoint_hidden_dim = keypoint_hidden_dim
        self.time_hidden_dim = self.keypoint_hidden_dim * time_hidden_dim
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.body_point_num = harper_body_point_num if is_harper else coco_body_point_num
        if body_part[0]:
            self.GCN_body = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_body = GAT(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_body = GIN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        if body_part[1]:
            self.GCN_head = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_head = GAT(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_head = GIN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        if body_part[2]:
            self.GCN_hand = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_hand = GAT(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
            # self.GCN_hand = GIN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        self.gcn_attention = nn.Linear(int(self.keypoint_hidden_dim * self.input_size / 3), 1)
        # self.gcn_attention = nn.MultiheadAttention(embed_dim=self.keypoint_hidden_dim, num_heads=1, batch_first=True)
        if self.model == 'gcn_lstm':
            self.time_model = nn.LSTM(math.ceil(self.input_size / 3) * self.keypoint_hidden_dim,
                                      hidden_size=self.time_hidden_dim, num_layers=3, bidirectional=True,
                                      batch_first=True)
            self.fc_input_size = self.time_hidden_dim * 2
            self.lstm_attention = nn.Linear(self.fc_input_size, 1)
        elif self.model == 'gcn_conv1d':
            self.time_model = nn.Sequential(
                nn.Conv1d(math.ceil(self.input_size / 3) * self.keypoint_hidden_dim, 64, kernel_size=7, stride=3,
                          padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5))
            # self.other_parameters += self.time_model.parameters()
            self.fc_input_size = 256 * math.ceil(math.ceil(sequence_length / frame_sample_hop / 3) / 2)
        elif model == 'gcn_tran':
            model_dim, num_heads, num_layers, num_classes = 256, 8, 3, 16
            self.embedding = nn.Linear(math.ceil(self.input_size / 3) * self.keypoint_hidden_dim, model_dim)
            self.positional_encoding = nn.Parameter(torch.zeros(1, int(sequence_length / frame_sample_hop), model_dim))

            encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # self.other_parameters += self.embedding.parameters() + self.positional_encoding + self.transformer_encoder.parameters()
            self.fc_input_size = model_dim
        else:
            self.time_edge_index = torch.tensor(
                np.array([[i, i + 1] for i in range(int(self.sequence_length / self.frame_sample_hop) - 1)]),
                dtype=torch.int32, device=device).t().contiguous()
            self.time_edge_index = torch.cat([self.time_edge_index, self.time_edge_index.flip([0])], dim=1)
            self.time_edge_index, _ = add_self_loops(self.time_edge_index,
                                                     num_nodes=int(self.sequence_length / self.frame_sample_hop))
            self.GCN_time = GCN(
                in_channels=math.ceil(self.input_size / 3) * self.keypoint_hidden_dim,
                hidden_channels=self.time_hidden_dim, num_layers=2)
            # self.GCN_time = GAT(in_channels=int(self.keypoint_hidden_dim * self.input_size / 3),
            #                     hidden_channels=self.keypoint_hidden_dim,
            #                     num_layers=2)
            # self.GCN_time = GIN(in_channels=int(self.keypoint_hidden_dim * self.input_size / 3),
            #                     hidden_channels=self.keypoint_hidden_dim,
            #                     num_layers=2)
            self.fc_input_size = int(self.time_hidden_dim * sequence_length / frame_sample_hop)
            # self.other_parameters += self.GCN_time.parameters()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_hidden1),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_hidden2),
        )
        self.classifier = Attack_Classifier(framework, self.fc_hidden2) if is_attack else Classifier(framework,
                                                                                                     self.fc_hidden2)
        self.train_classifier = train_classifier
        # self.other_parameters += self.attitude_head.parameters()
        # self.other_parameters += self.action_head.parameters()

    def forward(self, data):
        x_list = []
        if self.body_part[0]:
            x_body, edge_index_body, batch_body = data[0][0].to(dtype=dtype, device=device), data[1][0].to(
                device=device), data[2][0].to(device)
            x_body = self.GCN_body(x=x_body, edge_index=edge_index_body, batch=batch_body).to(dtype=dtype,
                                                                                              device=device)
            # x_body = global_mean_pool(x_body, batch_body)
            x_body = x_body.view(-1, int(self.sequence_length / self.frame_sample_hop), self.keypoint_hidden_dim * (
                self.body_point_num))
            x_list.append(x_body)
        if self.body_part[1]:
            x_head, edge_index_head, batch_head = data[0][1].to(dtype=dtype, device=device), data[1][1].to(device), \
                data[2][1].to(device)
            x_head = self.GCN_head(x=x_head, edge_index=edge_index_head, batch=batch_head).to(dtype=dtype,
                                                                                              device=device)
            # x_head = global_mean_pool(x_head, batch_head)
            x_head = x_head.view(-1, int(self.sequence_length / self.frame_sample_hop),
                                 self.keypoint_hidden_dim * head_point_num)
            x_list.append(x_head)
        if self.body_part[2]:
            x_hand, edge_index_hand, batch_hand = data[0][2].to(dtype=dtype, device=device), data[1][2].to(device), \
                data[2][2].to(device)
            x_hand = self.GCN_hand(x=x_hand, edge_index=edge_index_hand, batch=batch_hand).to(dtype=dtype,
                                                                                              device=device)
            # x_hand = global_mean_pool(x_hand, batch_hand)
            x_hand = x_hand.view(-1, int(self.sequence_length / self.frame_sample_hop),
                                 self.keypoint_hidden_dim * hands_point_num)
            x_list.append(x_hand)
        x = torch.cat(x_list, dim=2)
        x = x.view(-1, int(self.sequence_length / self.frame_sample_hop),
                   self.keypoint_hidden_dim * int(self.input_size / 3))
        gcn_attention_weights = nn.Softmax(dim=1)(self.gcn_attention(x))
        x = x * gcn_attention_weights

        # x = x.reshape(-1, int(self.input_size / 3), self.keypoint_hidden_dim)
        # x, gcn_attention_weights = self.gcn_attention(x, x, x)
        # x = x.reshape(-1, self.sequence_length, int(self.input_size / 3), self.keypoint_hidden_dim)
        # x = x.reshape(-1, self.sequence_length, self.keypoint_hidden_dim * int(self.input_size / 3))
        # gcn_attention_weights = torch.mean(gcn_attention_weights, dim=(0, 1))
        if self.model == 'gcn_lstm':
            on, _ = self.time_model(x)
            on = on.view(on.shape[0], on.shape[1], 2, -1)
            x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
            attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
            x = torch.sum(x * attention_weights, dim=1)
        elif self.model == 'gcn_conv1d':
            x = torch.transpose(x, 1, 2)
            x = self.time_model(x)
            x = x.flatten(1)
        elif self.model == 'gcn_tran':
            x = self.embedding(x) + self.positional_encoding
            x = x.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
            x = self.transformer_encoder(x)
            x = x.mean(dim=0)  # Global average pooling
        else:
            x_time = torch.zeros((x.shape[0], x.shape[1] * self.time_hidden_dim), dtype=dtype, device=device)
            for i in range(x.shape[0]):
                x_t = x[i]
                x_t = self.GCN_time(x=x_t, edge_index=self.time_edge_index).to(dtype=dtype, device=device)
                # x_t, new_edge_index, _, _, _, _ = self.pool(x_t, new_edge_index)
                x_time[i] = x_t.view(1, -1)[0]
            x = x_time.flatten(1)
        y = self.fc(x)
        return self.classifier(y) if self.train_classifier else y
        # return y1, y2, y3, gcn_attention_weights


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
                 body,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(body=body)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)).to(device) if data_bn else iden
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
    def __init__(self, body_part, framework):
        super(STGCN, self).__init__()
        super().__init__()
        self.body_part = body_part
        self.input_size = get_inputs_size(body_part)
        self.framework = framework
        graph_cfg = ()
        if self.body_part[0]:
            self.stgcn_body = ST_GCN_18(3, 0).to(device)
            self.fcn_body = nn.Conv2d(256, 16, kernel_size=1).to(device)
        if self.body_part[1]:
            self.stgcn_head = ST_GCN_18(3, 1).to(device)
            self.fcn_head = nn.Conv2d(256, 16, kernel_size=1).to(device)
        if self.body_part[2]:
            self.stgcn_hand = ST_GCN_18(3, 2).to(device)
            self.fcn_hand = nn.Conv2d(256, 16, kernel_size=1).to(device)
        self.classifier = Classifier(framework, 16 * self.body_part.count(True))

    def forward(self, x):
        y_list = []
        if self.body_part[0]:
            y = self.stgcn_body(x=x[0].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y = self.fcn_body(y).view(y.size(0), -1)
            y_list.append(y)
        if self.body_part[1]:
            y = self.stgcn_head(x=x[1].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y = self.fcn_head(y).view(y.size(0), -1)
            y_list.append(y)
        if self.body_part[2]:
            y = self.stgcn_hand(x=x[2].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y = self.fcn_hand(y).view(y.size(0), -1)
            y_list.append(y)
        y = torch.cat(y_list, dim=1)
        return self.classifier(y)


class MSGCN(nn.Module):
    def __init__(self, body_part, framework, keypoint_hidden_dim):
        super(MSGCN, self).__init__()
        super().__init__()
        self.body_part = body_part
        self.input_size = get_inputs_size(body_part)
        self.framework = framework
        if self.body_part[0]:
            self.MSGCN_body = MsG3d(0, keypoint_hidden_dim).to(device)
        if self.body_part[1]:
            self.MSGCN_head = MsG3d(1, keypoint_hidden_dim).to(device)
        if self.body_part[2]:
            self.MSGCN_hand = MsG3d(2, keypoint_hidden_dim).to(device)
        self.classifier = Classifier(framework, keypoint_hidden_dim * self.body_part.count(True))

    def forward(self, x):
        y_list = []
        if self.body_part[0]:
            y = self.MSGCN_body(x=x[0].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y_list.append(y)
        if self.body_part[1]:
            y = self.MSGCN_head(x=x[1].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y_list.append(y)
        if self.body_part[2]:
            y = self.MSGCN_hand(x=x[2].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y_list.append(y)
        y = torch.cat(y_list, dim=1)
        return self.classifier(y)


class DGSTGCN(nn.Module):
    def __init__(self, body_part, framework):
        super(DGSTGCN, self).__init__()
        super().__init__()
        self.body_part = body_part
        self.input_size = get_inputs_size(body_part)
        self.framework = framework
        if self.body_part[0]:
            self.DGSTGCN_body = DG_Model(0, 16).to(device)
        if self.body_part[1]:
            self.DGSTGCN_head = DG_Model(1, 16).to(device)
        if self.body_part[2]:
            self.DGSTGCN_hand = DG_Model(2, 16).to(device)
        self.classifier = Classifier(framework, 16 * self.body_part.count(True))

    def forward(self, x):
        y_list = []
        if self.body_part[0]:
            y = self.DGSTGCN_body(x=x[0].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y_list.append(y)
        if self.body_part[1]:
            y = self.DGSTGCN_head(x=x[1].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y_list.append(y)
        if self.body_part[2]:
            y = self.DGSTGCN_hand(x=x[2].to(dtype=dtype, device=device)).to(dtype=dtype, device=device)
            y_list.append(y)
        y = torch.cat(y_list, dim=1)
        return self.classifier(y)


class R3D(nn.Module):
    def __init__(self, framework):
        super(R3D, self).__init__()
        num_classes = 16
        self.framework = framework
        self.resnet3d = models.r3d_18(weights=None)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)
        self.classifier = Classifier(framework)

    def forward(self, x):
        y = self.resnet3d(x)
        return self.classifier(y)
