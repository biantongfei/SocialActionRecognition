from torch.utils.data import DataLoader
import torch
import torch.nn.utils.rnn as rnn_utils


def rnn_collate_fn(data):
    data.sort(key=lambda feature: feature[0].shape[0], reverse=True)
    x, y = [], []
    for d in data:
        x.append(d[0])
        y.append(d[1])
    data_length = [feature.shape[0] for feature in x]
    x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    return (x, torch.Tensor(y).long()), data_length


class JPLDataLoader(DataLoader):
    def __init__(self, model, dataset, batch_size, max_length, shuffle=False):
        super(JPLDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        if model in ['lstm', 'gru']:
            self.collate_fn = rnn_collate_fn
        elif model == 'conv1d':
            self.collate_fn = self.conv1d_collate_fn
        self.max_length = max_length

    def conv1d_collate_fn(self, data):
        padding = 'zero'
        # padding = 'same'
        input, label = None, []
        for index, d in enumerate(data):
            x = d[0]
            while x.shape[0] < self.max_length:
                x = torch.cat(
                    (x, torch.zeros((1, x.shape[1])) if padding == 'zero' else x[-1].reshape((1, x.shape[1]))),
                    dim=0)
            input = x.reshape(1, x.shape[0], x.shape[1]) if index == 0 else torch.cat(
                (input, x.reshape(1, x.shape[0], x.shape[1])), dim=0)
            label.append(d[1])
        return input, torch.Tensor(label).long()
