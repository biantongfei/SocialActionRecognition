import random

import torch
import numpy as np

from Models import RNN
from Dataset import rnn_collate_fn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional
import torch.nn.utils.rnn as rnn_utils

device = torch.device('mps')


class DataSet(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.x = []
        self.y = []
        for i in range(1024):
            times = i % 7
            self.x.append(np.zeros((10, 266)))
            self.y.append(times)
            for ii in range(10):
                if ii == 0:
                    self.x[i][ii] = np.array([random.randint(1, 30) for _ in range(266)])
                else:
                    self.x[i][ii] = self.x[i][ii - 1] * times
            index = random.randint(1, 10)
            self.x[i] = torch.from_numpy(self.x[i][:index])

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return 1024


net = RNN(is_coco=True, action_recognition=1, body_part=[True, True, True], bidirectional=False).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
Data_train = DataSet()
Data_test = DataSet()
train_loader = DataLoader(dataset=Data_train, batch_size=32, collate_fn=rnn_collate_fn)
val_loader = DataLoader(dataset=Data_test, batch_size=32, collate_fn=rnn_collate_fn)
acc = 0
while acc < 0.8:
    for data in train_loader:
        (inputs, labels), data_length = data
        inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True)
        inputs = inputs.to(torch.float).to(device)
        outputs = net(inputs)
        loss = functional.cross_entropy(outputs, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_true, y_pred = [], []
    for data in val_loader:
        (inputs, labels), data_length = data
        inputs = rnn_utils.pack_padded_sequence(inputs, data_length, batch_first=True).to(torch.float).to(device)
        outputs = net(inputs)
        pred = outputs.argmax(dim=1)
        y_true += labels.tolist()
        y_pred += pred.tolist()
    print(y_true)
    print(y_pred)
    y_true, y_pred = torch.Tensor(y_true), torch.Tensor(y_pred)
    acc = y_pred.eq(y_true).sum().float().item() / y_pred.size(dim=0)
    print(acc)
