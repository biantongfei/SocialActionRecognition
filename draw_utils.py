from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor, tensor


def draw_performance(accuracy_loss_dict, sub_name):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(accuracy_loss_dict.keys())))
    for index, key in enumerate(accuracy_loss_dict.keys()):
        acc = [100 * a for a in accuracy_loss_dict[key][0]]
        plt.plot(range(0, len(accuracy_loss_dict[key][0])), acc, color=colors[index])
    plt.legend(accuracy_loss_dict.keys())
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('accuracy_%s_%s.png' % (key, sub_name))
    plt.close()
    for index, key in enumerate(accuracy_loss_dict.keys()):
        loss = [float(l) for l in accuracy_loss_dict[key][1]]
        plt.plot(range(0, len(accuracy_loss_dict[key][1])), loss, color=colors[index])
        plt.legend(accuracy_loss_dict.keys())
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('loss_%s_%s.png' % (key, sub_name))
        plt.close()
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, sub_name):
    y_true, y_pred = Tensor.cpu(y_true), Tensor.cpu(y_pred)
    print(y_true.shape, y_pred.shape)
    print(y_true)
    print(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    plt.imshow(cm)  # 在特定的窗口上显示图像
    plt.title('confusion_matrix')  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(classes)))
    plt.xticks(num_local, classes, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, classes)  # 将标签印在y轴坐标上
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Results')
    plt.savefig('cm_%s.png' % sub_name, format='png')
    # plt.show()


if __name__ == '__main__':
    pass
