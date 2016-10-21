# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt

__author__ = 'Henry Cagnini'


def main(file_name):
    X = loadtxt(file_name, delimiter=',')

    N = 5

    plt.boxplot(X[:5])

    plt.xlabel(u'geração')
    plt.ylabel('fitness')

    plt.show()

if __name__ == '__main__':
    filename = '/home/henry/Projects/forrestTemp/metadata/some_metadata.csv'
    main(filename)
