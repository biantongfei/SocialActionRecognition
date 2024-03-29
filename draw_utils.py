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




    # y_true, y_pred = Tensor.cpu(y_true), Tensor.cpu(y_pred)
    # cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    # FP = sum(cm.sum(axis=0)) - sum(np.diag(cm))  # 假正样本数
    # FN = sum(cm.sum(axis=1)) - sum(np.diag(cm))  # 假负样本数
    # TP = sum(np.diag(cm))  # 真正样本数
    # TN = sum(cm.sum().flatten()) - (FP + FN + TP)  # 真负样本数
    # SUM = TP + FP
    # PRECISION = TP / (TP + FP)  # 查准率，又名准确率
    # RECALL = TP / (TP + FN)  # 查全率，又名召回率
    # plt.figure(figsize=(12, 12), dpi=100)
    # np.set_printoptions(precision=2)
    # # 在混淆矩阵中每格的概率值
    # ind_array = np.arange(len(classes) + 1)
    # x, y = np.meshgrid(ind_array, ind_array)  # 生成坐标矩阵
    # diags = np.diag(cm)  # 对角TP值
    # TP_FNs, TP_FPs = [], []
    # for x_val, y_val in zip(x.flatten(), y.flatten()):  # 并行遍历
    #     max_index = len(classes)
    #     if x_val != max_index and y_val != max_index:  # 绘制混淆矩阵各格数值
    #         c = cm[y_val][x_val]
    #         plt.text(x_val, y_val, c, color='black', fontsize=15, va='center', ha='center')
    #     elif x_val == max_index and y_val != max_index:  # 绘制最右列即各数据类别的查全率
    #         TP = diags[y_val]
    #         TP_FN = cm.sum(axis=1)[y_val]
    #         recall = TP / (TP_FN)
    #         if recall != 0.0 and recall > 0.01:
    #             recall = str('%.2f' % (recall * 100,)) + '%'
    #         elif recall == 0.0:
    #             recall = '0'
    #         TP_FNs.append(TP_FN)
    #         plt.text(x_val, y_val, str(TP_FN) + '\n' + str(recall), color='black', va='center', ha='center')
    #     elif x_val != max_index and y_val == max_index:
    #         # 绘制最下行即各数据类别的查准率
    #         TP = diags[x_val]
    #         TP_FP = cm.sum(axis=0)[x_val]
    #         precision = TP / (TP_FP)
    #         if precision != 0.0 and precision > 0.01:
    #             precision = str('%.2f' % (precision * 100,)) + '%'
    #         elif precision == 0.0:
    #             precision = '0'
    #         TP_FPs.append(TP_FP)
    #         plt.text(x_val, y_val, str(TP_FP) + '\n' + str(precision), color='black', va='center', ha='center')
    # cm = np.insert(cm, max_index, TP_FNs, 1)
    # cm = np.insert(cm, max_index, np.append(TP_FPs, SUM), 0)
    # plt.text(max_index, max_index, str(SUM) + '\n' + str('%.2f' % (PRECISION * 100,)) + '%', color='white', va='center',
    #          ha='center')
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('confusion_matrix')
    # plt.colorbar()
    # xlocations = np.array(range(len(classes)))
    # plt.xticks(xlocations, classes, rotation=90)
    # plt.yticks(xlocations, classes)
    # plt.ylabel('actual label')
    # plt.xlabel('predict label')
    # # offset the tick
    # tick_marks = np.array(range(len(classes))) + 0.5
    # plt.gca().set_xticks(tick_marks, minor=True)
    # plt.gca().set_yticks(tick_marks, minor=True)
    # plt.gca().xaxis.set_ticks_position('none')
    # plt.gca().yaxis.set_ticks_position('none')
    # plt.grid(True, which='minor', linestyle='-')
    # # plt.gcf().subplots_adjust(bottom=0.15)
    # # show confusion matrix
    # plt.savefig('plots/cm_%s.png' % sub_name, format='png')
    # # plt.show()
    # plt.close()


if __name__ == '__main__':
    pass
