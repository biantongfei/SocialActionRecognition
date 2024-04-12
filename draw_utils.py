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
    cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('plots/cm_%s.png' % sub_name, format='png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    pass
