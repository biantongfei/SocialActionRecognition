from torch import nn

coco_point_num = 133
halpe_point_num = 136
box_feature_num = 4
action_class_num = 9
attitude_class_num = 3


class FCNN(nn.Module):
    def __init__(self, is_coco, action_recognition, *args, **kwargs):
        super(FCNN, self).__init__()
        super().__init__(*args, **kwargs)
        self.is_coco = is_coco
        points_num = coco_point_num if self.is_coco else halpe_point_num
        self.input_size = 2 * points_num + box_feature_num
        self.output_size = action_class_num if action_recognition else attitude_class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(16, self.output_size)
        )

    def forward(self, x):
        x = self.fc(x)

        return x


class CNN(nn.Module):
    def __init__(self, is_coco, action_recognition, *args, **kwargs):
        super(CNN, self).__init__()
        super().__init__(*args, **kwargs)
        self.is_coco = is_coco
        points_num = coco_point_num if self.is_coco else halpe_point_num
        self.input_size = (points_num + box_feature_num / 2, 2)
        self.output_size = action_class_num if action_recognition else attitude_class_num
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 9, kernel_size=(3, 2), padding=(1, 0)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        x = self.Conv(x)
        x = self.fc(x)

        return x
