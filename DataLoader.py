from torch.utils.data import DataLoader
import torch
import torch.nn.utils.rnn as rnn_utils


def rnn_collate_fn(data):
    data.sort(key=lambda feature: feature[0].shape[0], reverse=True)
    x, intention_labels, attitude_labels, action_labels = [], [], [], []
    for d in data:
        x.append(d[0])
        intention_labels.append(d[1][0])
        attitude_labels.append((d[1][1]))
        action_labels.append(d[1][2])
    data_length = [feature.shape[0] for feature in x]
    x = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    return (x, (torch.Tensor(intention_labels).long(), torch.Tensor(attitude_labels).long(),
                torch.Tensor(action_labels).long())), data_length


class JPLDataLoader(DataLoader):
    def __init__(self, model, dataset, batch_size, max_length, shuffle=False, empty_frame=False):
        super(JPLDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        if model in ['lstm', 'gru']:
            self.collate_fn = rnn_collate_fn
        elif model == 'conv1d':
            self.collate_fn = self.conv1d_collate_fn
        self.max_length = max_length
        self.empty_frame = empty_frame

    def conv1d_collate_fn(self, data):
        padding = 'same' if self.empty_frame == 'same' else 'zero'
        input, int_label, att_label, act_label = None, [], [], []
        for index, d in enumerate(data):
            x = d[0]
            while x.shape[0] < self.max_length:
                x = torch.cat(
                    (x, torch.zeros((1, x.shape[1])) if padding == 'zero' else x[-1].reshape((1, x.shape[1]))),
                    dim=0)
            input = x.reshape(1, x.shape[0], x.shape[1]) if index == 0 else torch.cat(
                (input, x.reshape(1, x.shape[0], x.shape[1])), dim=0)
            int_label.append(d[1][0])
            att_label.append(d[1][1])
            act_label.append(d[1][2])
        return input, (torch.Tensor(int_label).long(), torch.Tensor(att_label).long(), torch.Tensor(act_label).long())
