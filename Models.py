import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

coco_body_point_num = 23
halpe_body_point_num = 26
head_point_num = 68
hands_point_num = 42
box_feature_num = 4
ori_action_class_num = 7
action_class_num = 9
attitude_class_num = 3
fps = 30
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def get_points_num(is_coco, body_part):
    points_num = 0
    if body_part[0]:
        points_num += coco_body_point_num if is_coco else halpe_body_point_num
    if body_part[1]:
        points_num += head_point_num
    if body_part[2]:
        points_num += hands_point_num
    return points_num


class DNN(nn.Module):
    def __init__(self, is_coco, action_recognition, body_part, model):
        super(DNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        # self.input_size = 2 * points_num + box_feature_num
        self.input_size = 2 * points_num
        if action_recognition:
            self.output_size = ori_action_class_num if action_recognition == 1 else action_class_num
        else:
            self.output_size = attitude_class_num
        if model == 'avg':
            self.fc = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.BatchNorm1d(64),
                nn.Linear(64, 16),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.BatchNorm1d(16),
                nn.Linear(16, self.output_size),
            )
        elif model == 'perframe':
            self.fc = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.BatchNorm1d(128),
                # nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                # nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                # nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                # nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(16, self.output_size),
            )

    def forward(self, x):
        x = self.fc(x)
        # x = nn.Softmax(dim=1)(x)
        return x


class RNN(nn.Module):
    def __init__(self, is_coco, action_recognition, body_part, bidirectional=False, gru=False):
        super(RNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        self.input_size = 2 * points_num
        self.hidden_size = 512
        self.bidirectional = bidirectional
        if action_recognition:
            self.output_size = ori_action_class_num if action_recognition == 1 else action_class_num
        else:
            self.output_size = attitude_class_num

        if gru:
            self.rnn = nn.GRU(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=bidirectional,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=bidirectional,
                               batch_first=True)
            # self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3,
            #                    bidirectional=bidirectional, dropout=0.5, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(self.hidden_size * (2 if bidirectional else 1), self.output_size)
        self.dropout = nn.Dropout(0.5)
        self.BatchNorm1d = nn.BatchNorm1d(self.hidden_size * (2 if bidirectional else 1))

    def forward(self, x):
        on, (hn, _) = self.rnn(x)
        out_pad, out_length = rnn_utils.pad_packed_sequence(on, batch_first=True)
        # print(out_pad.data.shape)
        if self.bidirectional:
            out = torch.zeros(out_pad.data.shape[0], self.hidden_size * 2).to(device)
            for i in range(out_pad.data.shape[0]):
                index = out_length[i] - 1
                out[i] = torch.cat((out_pad.data[i, index, :self.hidden_size], out_pad.data[i, 0, self.hidden_size:]),
                                   dim=0)
        else:
            out = torch.zeros(out_pad.data.shape[0], self.hidden_size).to(device)
            for i in range(out_pad.data.shape[0]):
                index = out_length[i] - 1
                out[i] = out_pad.data[i, index, :]
        out = nn.ReLU(out)
        # out = self.dropout(out)
        out = self.BatchNorm1d(out)
        out = self.fc(out)
        return out


class Cnn1D(nn.Module):
    def __init__(self, is_coco, action_recognition, body_part):
        super(Cnn1D, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        self.input_size = 2 * points_num
        if action_recognition:
            self.output_size = ori_action_class_num if action_recognition == 1 else action_class_num
        else:
            self.output_size = attitude_class_num
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, self.input_size, kernel_size=3),
            # nn.BatchNorm2d(self.input_size),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv1d(self.input_size, self.input_size, kernel_size=3),
            nn.BatchNorm2d(self.input_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv1d(self.input_size, self.input_size, kernel_size=3),
            nn.BatchNorm2d(self.input_size),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * self.input_size, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            # nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            # nn.Dropout(0.5),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.cnn(x)
        x = x.view(-1, x.size)
        x = self.fc(x)
        return x
