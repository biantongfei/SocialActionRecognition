from matplotlib import pyplot as plt
import numpy as np


def draw_performance(hyperparam_dict):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(hyperparam_dict.keys())))
    for index, key in enumerate(hyperparam_dict.keys()):
        acc = [100 * a for a in hyperparam_dict[key]]
        plt.plot(range(0, len(hyperparam_dict[key])), acc, color=colors[index])
    plt.legend(['accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('accuracy.png')
    plt.show()





if __name__ == '__main__':
    pass
