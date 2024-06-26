from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torch import Tensor
import cv2


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
    # int_f1 = [0.868, 0.889, 0.894, 0.904]
    # att_f1 = [0.757, 0.795, 0.804, 0.810]
    # act_f1 = [0.558, 0.603, 0.652, 0.682]

    l1 = plt.plot(windows, int_f1, 'r--', label='Intent')
    l2 = plt.plot(windows, att_f1, 'g--', label='Attitude')
    l3 = plt.plot(windows, act_f1, 'b--', label='Action')
    plt.plot(windows, int_f1, 'ro-', windows, att_f1, 'g+-', windows, act_f1, 'b^-')
    plt.xlabel('Observation Window Size (Frame)')
    plt.ylabel('F1 score')
    ax = plt.gca()
    x_major_locator = plt.MultipleLocator(5)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.show()


def draw_attention_weight():
    width, height = 350, 650
    width, height = 300, 286
    width, height = 500, 316
    heatmap_data = np.zeros((height, width))
    points = [[175, 80], [190, 66], [166, 66], [207, 76], [149, 75], [246, 148], [108, 150], [283, 243], [74, 244],
              [316, 318], [43, 319], [213, 355], [146, 354], [217, 477], [137, 476], [222, 555], [133, 554], [225, 613],
              [130, 613], [243, 606], [110, 605], [212, 575], [140, 575]]
    points = [[17, 60], [18, 95], [22, 130], [30, 165], [43, 197], [65, 226], [91, 250], [119, 270], [152, 275],
              [185, 269], [213, 249], [239, 224], [260, 195], [273, 162], [280, 126], [282, 91], [282, 55], [42, 33],
              [58, 20], [81, 16], [103, 19], [125, 28], [169, 26], [190, 17], [213, 13], [236, 16], [253, 29],
              [147, 53], [147, 76], [148, 100], [148, 123], [123, 139], [135, 143], [148, 148], [162, 143], [175, 138],
              [69, 57], [82, 49], [99, 49], [113, 59], [98, 63], [82, 63], [182, 59], [196, 47], [212, 47], [227, 55],
              [214, 60], [198, 61], [99, 181], [117, 174], [136, 170], [149, 172], [163, 169], [183, 172], [202, 179],
              [184, 199], [166, 208], [150, 210], [136, 209], [117, 201], [107, 182], [136, 181], [149, 182],
              [163, 180], [194, 181], [164, 191], [150, 192], [136, 191]]
    points = [[106, 257], [144, 232], [180, 202], [209, 168], [243, 141], [146, 121], [152, 89], [158, 59], [161, 21],
              [115, 119], [115, 80], [115, 42], [116, 5], [85, 127], [74, 91], [66, 57], [57, 23], [56, 146], [40, 126],
              [27, 103], [6, 70], [392, 257], [353, 232], [318, 203], [289, 168], [255, 142], [352, 121], [346, 89],
              [340, 59], [338, 21], [383, 120], [382, 80], [383, 42], [382, 5], [413, 126], [424, 91], [432, 56],
              [440, 22], [442, 145], [458, 127], [471, 103], [491, 70]]
    edges = [[0, 1], [0, 2], [1, 3], [2, 4],  # Head
             [5, 7], [7, 9], [6, 8], [8, 10],  # Body
             [5, 6], [11, 12], [5, 11], [6, 12],
             [11, 13], [12, 14], [13, 15], [14, 16],
             [15, 21], [21, 19], [21, 17], [16, 22], [22, 20], [22, 18]]
    edges = [[23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
             [32, 33], [33, 34], [34, 35],
             [35, 36], [36, 37], [37, 38], [38, 39], [40, 41], [41, 42], [42, 43], [43, 44], [45, 46],
             [46, 47], [47, 48], [48, 49],
             [50, 51], [51, 52], [52, 53], [54, 55], [55, 56], [56, 57], [57, 58], [59, 60], [60, 61],
             [61, 62], [62, 63], [63, 64], [64, 59],
             [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 65], [71, 72], [72, 73], [73, 74], [74, 75],
             [75, 76], [76, 77], [77, 78],
             [78, 79], [79, 80], [80, 81], [81, 82], [71, 83], [83, 84], [84, 85], [85, 86], [86, 87],
             [87, 88], [88, 89], [89, 90], [23, 40], [39, 49], [50, 44], [50, 45], [50, 62], [50, 65], [53, 56],
             [71, 82], [87, 77], [83, 90]]
    edges = [[91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
             [100, 101], [101, 102],
             [102, 103], [91, 104], [104, 105], [105, 106], [106, 107], [91, 108], [108, 109], [109, 110],
             [110, 111], [112, 113],
             [113, 114], [114, 115], [115, 116], [112, 117], [117, 118], [118, 119], [119, 120], [112, 121],
             [121, 122], [122, 123],
             [123, 124], [112, 125], [125, 126], [126, 127], [127, 128], [112, 129], [129, 130], [130, 131],
             [131, 132]]
    # weights = [0.3, 0.5, 0.2, 0.1, 0.4, 0.3, 0.5, 0.2, 0.1, 0.4, 0.3, 0.5, 0.2, 0.1, 0.4, 0.3, 0.5, 0.2, 0.1, 0.4, 0.3,
    #            0.5, 0.2]
    x, y = [], []
    for i, p in enumerate(points):
        x.append(p[0])
        y.append(p[1])
        # heatmap_data[p[1], p[0]] = weights[i]
    heatmap_data = cv2.GaussianBlur(heatmap_data, (75, 75), sigmaX=0, sigmaY=0)
    # plt.imshow(heatmap_data, cmap='YlOrRd', alpha=0.5)  # Use alpha to make the heatmap semi-transparent
    # plt.colorbar()
    plt.scatter(x, y, marker='.', color='black')
    for e in edges:
        plt.plot((points[e[0] - 23 - 68][0], points[e[1] - 23 - 68][0]),
                 (points[e[0] - 23 - 68][1], points[e[1] - 23 - 68][1]),
                 linewidth=1, color='black')
    plt.axis('equal')
    plt.show()


def draw_pie_chart():
    sizes = [28, 25, 12, 30, 30, 31, 23, 24, 17, 70]
    labels = ['Hand Shake', 'Hug', 'Pet', 'Wave', 'Punch', 'Throw', 'Point-Converse', 'Gaze', 'Leave', 'No Response']
    colors = [(np.random.random(), np.random.random(), np.random.random()) for _ in labels]

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            # 同时显示数值和占比的饼图
            return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct

    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct=make_autopct(sizes))
    plt.show()


if __name__ == '__main__':
    draw_observe_window_plots()
    # draw_attention_weight()
    # draw_pie_chart()
