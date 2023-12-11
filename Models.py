from torch import nn
import math

coco_body_point_num = 23
halpe_body_point_num = 26
head_point_num = 68
hands_point_num = 42
box_feature_num = 4
ori_action_class_num = 7
action_class_num = 9
attitude_class_num = 3
batch_size = 128


def get_points_num(is_coco, body_part):
    if body_part == 1:
        points_num = coco_body_point_num if is_coco else halpe_body_point_num
    elif body_part == 2:
        points_num = head_point_num + (coco_body_point_num if is_coco else halpe_body_point_num)
    elif body_part == 3:
        points_num = hands_point_num + (coco_body_point_num if is_coco else halpe_body_point_num)
    else:
        points_num = head_point_num + hands_point_num + (coco_body_point_num if is_coco else halpe_body_point_num)
    return points_num


class FCNN(nn.Module):
    def __init__(self, is_coco, action_recognition, body_part=4, *args, **kwargs):
        super(FCNN, self).__init__()
        super().__init__(*args, **kwargs)
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        self.input_size = 2 * points_num + box_feature_num
        if action_recognition:
            self.output_size = ori_action_class_num if action_recognition == 1 else action_class_num
        else:
            self.output_size = attitude_class_num
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size)
        )

    def forward(self, x):
        x = self.fc(x)

        return x


class CNN(nn.Module):
    def __init__(self, is_coco, action_recognition=False, body_part=4, *args, **kwargs):
        super(CNN, self).__init__()
        super().__init__(*args, **kwargs)
        self.is_coco = is_coco
        points_num = get_points_num(is_coco, body_part)
        if action_recognition:
            self.output_size = ori_action_class_num if action_recognition == 0 else action_class_num
        else:
            self.output_size = attitude_class_num

        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 3), padding=(2, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 2), padding=(2, 0), stride=(2, 1)),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.ceil((points_num + box_feature_num / 2) / 4) * 64, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size)
        )

    def forward(self, x):
        print(x.shape)
        print(x.shape)

        return x
