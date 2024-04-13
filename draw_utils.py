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
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小
    cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    str_cm = cm.astype(np.str_).tolist()
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
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


if __name__ == '__main__':
    pass
