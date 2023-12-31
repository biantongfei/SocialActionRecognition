import torch
from torch import nn

coco_body_point_num = 23
halpe_body_point_num = 26
head_point_num = 68
hands_point_num = 42
box_feature_num = 4
ori_action_class_num = 7
action_class_num = 9
attitude_class_num = 3
batch_size = 128
fps = 30


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
    def __init__(self, is_coco, action_recognition, body_part):
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
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc(x)

        return x


class RNN(nn.Module):
    def __init__(self, is_coco, action_recognition, body_part, video_len, bidirectional=False, gru=False):
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
            self.rnn = nn.GRU(self.input_size, hidden_size=self.hidden_size, num_layers=3, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(self.input_size, hidden_size=self.hidden_size, num_layers=3,
                                bidirectional=bidirectional)

        # Readout layer
        self.fc = nn.Linear(self.hidden_size * (2 if bidirectional else 1), self.output_size)
        self.dropout = nn.Dropout(0.5)
        self.BatchNorm1d = nn.BatchNorm1d(self.hidden_size * (2 if bidirectional else 1))

    def forward(self, x):
        # x = self.dropout(x)
        _, (hidden, _) = self.rnn(x)
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        # hidden = self.dropout(hidden)
        # hidden = self.BatchNorm1d(hidden)
        out = self.fc(hidden)
        return out
