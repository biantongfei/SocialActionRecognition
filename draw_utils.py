from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor


def draw_training_process(training_process):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(training_process.keys())))
    for index, key in enumerate(training_process.keys()):
        acc = [100 * a for a in training_process[key]['accuracy']]
        plt.plot(range(0, len(training_process[key]['accuracy'])), acc, color=colors[index])
        plt.legend(training_process.keys())
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
    plt.savefig('plots/accuracy.png')
    plt.close()
    for index, key in enumerate(training_process.keys()):
        f1 = [f for f in training_process[key]['f1']]
        plt.plot(range(0, len(training_process[key]['f1'])), f1, color=colors[index])
        plt.legend(training_process.keys())
        plt.xlabel('epoch')
        plt.ylabel('f1')
    plt.savefig('plots/f1.png')
    plt.close()
    for index, key in enumerate(training_process.keys()):
        loss = [l for l in training_process[key]['loss']]
        plt.plot(range(0, len(training_process[key]['loss'])), loss, color=colors[index])
        plt.legend(training_process.keys())
        plt.xlabel('epoch')
        plt.ylabel('loss')
    plt.savefig('plots/loss.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, sub_name):
    y_true, y_pred = Tensor.cpu(y_true), Tensor.cpu(y_pred)
    plt.rc('font', size='8')  # 设置字体样式、大小
    cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=None,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('plots/%s.jpg' % sub_name, dpi=300)


def draw_observe_window_plots():
    windows = [5, 10, 15, 20, 25, 30, 35, 40]
    # windows = [5, 10, 15, 20]
    int_f1 = [0.858, 0.890, 0.892, 0.901, 0.913, 0.919, 0.911, 0.923]
    att_f1 = [0.733, 0.783, 0.812, 0.834, 0.843, 0.867, 0.878, 0.883]
    act_f1 = [0.558, 0.611, 0.646, 0.681, 0.686, 0.749, 0.762, 0.761]
    # int_f1 = [0.5, 0.64, 0.79, 0.89]
    # att_f1 = [0.4, 0.56, 0.68, 0.81]
    # act_f1 = [0.3, 0.45, 0.58, 0.71]

    l1 = plt.plot(windows, int_f1, 'r--', label='Interest')
    l2 = plt.plot(windows, att_f1, 'g--', label='Attitude')
    l3 = plt.plot(windows, act_f1, 'b--', label='Action')
    plt.plot(windows, int_f1, 'ro-', windows, att_f1, 'g+-', windows, act_f1, 'b^-')
    plt.xlabel('Observation Window Size (Frame)')
    plt.ylabel('Confidence score')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    draw_observe_window_plots()
