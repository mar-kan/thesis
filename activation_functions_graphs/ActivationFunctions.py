import matplotlib.pyplot as plt
import numpy as np
import math


def main():
    afs = ['sigmoid', 'tanh', 'relu', 'lrelu', 'elu']

    t = np.arange(-5, 5, 0.1)
    for af in afs:
        f, name = plotAF(af, t)
        plotAxes(t)
        plt.plot(t, f)
        plt.ylabel(name)
        plt.show()


def plotAxes(t):
    z = np.zeros(100)
    plt.plot(t, z, 'black')

    z = np.zeros(12)
    t1 = np.arange(-1, 5, 0.5)
    plt.plot(z, t1, 'black')


def plotAF(name, t):
    if name == 'sigmoid':
        return 1 / (1 + np.exp(-t)), 'Sigmoid Function'
    elif name == 'tanh':
        return (np.exp(t) - np.exp(-t)) / (np.exp(t) + np.exp(-t)), "Hyperbolic Tangent Function"
    elif name == 'relu':
        return [max(0, t1) for t1 in t], "ReLU Function"
    elif name == 'lrelu':
        return [max(0.05 * t1, t1) for t1 in t], "LReLU Function"  # a = 0.05
    elif name == 'elu':
        x = []
        for t1 in t:
            if t1 > 0:
                x.append(t1)
            else:
                x.append(np.exp(t1) - 1)  # a = 1

        return x, "ELU Function"


if __name__ == '__main__':
    main()
