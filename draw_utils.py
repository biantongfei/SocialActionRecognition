from matplotlib import pyplot as plt
import numpy as np


def draw_performance(accuracy_dict):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(accuracy_dict.keys())))
    for index, key in enumerate(accuracy_dict.keys()):
        acc = [100 * a for a in accuracy_dict[key]]
        plt.plot(range(0, len(accuracy_dict[key])), acc, color=colors[index])
    plt.legend(accuracy_dict.keys)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('accuracy.png')
    plt.show()





if __name__ == '__main__':
    pass
