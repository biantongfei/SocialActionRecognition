import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

coco_body_point_num = 23
halpe_body_point_num = 26
head_point_num = 68
hands_point_num = 42
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
    def __init__(self, is_coco, body_part, framework):
        super(DNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        self.framework = framework
        self.input_size = 2 * points_num
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

        if self.framework == 'parallel':
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
            self.action_head = nn.Sequential(nn.BatchNorm1d(16 + attitude_class_num),
                                             nn.ReLU(),
                                             nn.Linear(16 + attitude_class_num, action_class_num)
                                             )

    def forward(self, x):
        y = self.fc(x)
        y1 = self.intent_head(y)
        if self.framework == 'parallel':
            y2 = self.attitude_head(y)
            y3 = self.action_head(y)
        elif self.framework == 'tree':
            y2 = self.attitude_head(torch.cat((y, y1), dim=1))
            y3 = self.action_head(torch.cat((y, y1), dim=1))
        elif self.framework == 'chain':
            y2 = self.attitude_head(torch.cat((y, y1), dim=1))
            y3 = self.action_head(torch.cat((y, y2), dim=1))
        return y1, y2, y3


class RNN(nn.Module):
    def __init__(self, is_coco, body_part, framework, bidirectional=False, gru=False):
        super(RNN, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        self.input_size = 2 * points_num
        self.hidden_size = 512
        self.bidirectional = bidirectional
        if gru:
            self.rnn = nn.GRU(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=bidirectional,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=bidirectional,
                               batch_first=True)
            # self.rnn = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3,
            #                    bidirectional=bidirectional, dropout=0.5, batch_first=True)

        # Readout layer
        self.dropout = nn.Dropout(0.5)
        self.BatchNorm1d = nn.BatchNorm1d(self.hidden_size * (2 if bidirectional else 1))
        self.attitude_head = nn.Sequential(nn.ReLU(),
                                           nn.Linear(self.hidden_size * (2 if bidirectional else 1),
                                                     attitude_class_num))
        self.action_head = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * (2 if bidirectional else 1) + attitude_class_num),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size * (2 if bidirectional else 1) + attitude_class_num, action_class_num))

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
        y = self.BatchNorm1d(out)
        # out = self.dropout(out)
        y1 = self.attitude_head(y)
        y2 = self.action_head(torch.cat((y, y1), dim=1))
        return y1, y2


class Cnn1D(nn.Module):
    def __init__(self, is_coco, body_part, framework):
        super(Cnn1D, self).__init__()
        super().__init__()
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        self.input_size = 2 * points_num
        self.framework = framework
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 64, kernel_size=7, stride=3, padding=3),
            nn.MaxPool1d(2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv1d(64, 32, kernel_size=7, stride=3, padding=3),
            nn.MaxPool1d(2, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Conv1d(32, 16, kernel_size=7, stride=3, padding=3),
            nn.MaxPool1d(2, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, 32),
            nn.BatchNorm1d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            # nn.Dropout(0.5),
        )
        self.intent_head = nn.Sequential(nn.ReLU(),
                                         nn.Linear(16, intent_class_num)
                                         )

        if self.framework == 'parallel':
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
        y1 = self.intent_head(y)
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
