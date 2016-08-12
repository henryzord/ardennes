# coding=utf-8

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
import pandas as pd
import itertools as it

from genotype import Individual

__author__ = 'Henry Cagnini'


def get_pmf(n_attributes, n_internal):
    # checks if n_nodes is a valid value for the tree
    if n_internal < 1 or not np.log2(n_internal + 1).is_integer():  # TODO verify integrity!
        raise ValueError(
            'Invalid value for n_nodes! Must be a power of 2, minus 1 (e.g, 1, 3, 7).'
        )

    pmf = np.ones((n_internal, n_attributes), dtype=np.float32) / n_attributes
    return pmf


def set_pmf(pmf, fittest):
    nodes = np.array(map(lambda x: x.nodes, fittest))
    counts = map(lambda v: Counter(v), nodes.T)

    n_internal, n_vals = pmf.shape
    n_fittest = fittest.shape[0]

    for i in xrange(n_internal):  # for each node index
        for j in xrange(n_vals):
            try:
                pmf[i, j] = counts[i][j] / float(n_fittest)
            except KeyError:
                pmf[i, j] = 0.

    return pmf


def init_pop(n_individuals, n_leaf, n_internal, pmf, sets):
    pop = np.array(
        map(
            lambda x: Individual(
                n_leaf=n_leaf,
                n_internal=n_internal,
                pmf=pmf,
                sets=sets
            ),
            xrange(n_individuals)
        )
    )
    return pop


def get_folds(df, n_folds=10, random_state=None):
    from sklearn.cross_validation import StratifiedKFold

    Y = df[df.columns[-1]]

    folds = StratifiedKFold(Y, n_folds=n_folds, shuffle=True, random_state=random_state)
    return folds


def early_stop(iteration, mean, median, past, n_past):
    past[iteration % n_past] = mean
    if median == mean and np.all(past == mean):
        raise StopIteration('you should stop.')
    else:
        return past


def get_node_count(n_nodes):
    n_leaf = (n_nodes + 1) / 2
    n_internal = n_nodes - n_leaf
    return n_internal, n_leaf


def main_loop(sets, n_nodes, n_individuals, n_iterations=100, threshold=0.9, verbose=True):
    n_pred_attr = sets['train'].columns.shape[0] - 1  # number of predictive attributes

    n_internal, n_leaf = get_node_count(n_nodes)

    pmf = get_pmf(n_pred_attr, n_internal)  # pmf has one distribution for each node
    population = init_pop(
        n_individuals=n_individuals,
        n_leaf=n_leaf,
        n_internal=n_internal,
        pmf=pmf,
        sets=sets
    )

    fitness = np.array(map(lambda x: x.fitness, population))

    integer_threshold = int(threshold * n_individuals)

    n_past = 3
    past = np.random.rand(n_past)

    iteration = 0
    while iteration < n_iterations:  # evolutionary process
        mean = np.mean(fitness)
        median = np.median(fitness)
        _max = np.max(fitness)

        if verbose:
            print 'mean: %+0.6f\tmedian: %+0.6f\tmax: %+0.6f' % (mean, median, _max)

        try:
            past = early_stop(iteration, mean, median, past, n_past)
        except StopIteration:
            break

        borderline = np.partition(fitness, integer_threshold)[integer_threshold]
        fittest = population[np.flatnonzero(fitness >= borderline)]
        pmf = set_pmf(pmf, fittest)

        to_replace = population[np.flatnonzero(fitness < borderline)]
        for ind in to_replace:
            ind.sample(pmf)

        fitness = np.array(map(lambda x: x.fitness, population))

        iteration += 1

    fittest = population[np.argmax(fitness)]
    return fittest


def get_iris(n_folds=10, random_state=None):
    data = load_iris()

    X, Y = data['data'], data['target_names'][data['target']]

    df = pd.DataFrame(X, columns=data['feature_names'])
    df['class'] = pd.Series(Y, index=df.index)

    folds = get_folds(df, n_folds=n_folds, random_state=random_state)

    return df, folds


def get_bank(n_folds=10, random_state=None):
    df = pd.read_csv('/home/henryzord/Projects/forrestTemp/datasets/bank-full_no_missing.csv', sep=',')
    folds = get_folds(df, n_folds=n_folds, random_state=random_state)
    return df, folds


def main():
    import warnings
    import random
    warnings.warn('WARNING: deterministic approach!')
    random.seed(1)
    np.random.seed(1)

    n_folds = 10
    df, folds = get_iris(n_folds=n_folds)
    # df, folds = get_bank(n_folds=2)

    acc = 0.

    for i, (arg_train, arg_test) in enumerate(folds):
        sets = {'train': df.iloc[arg_train], 'test': df.iloc[arg_test]}

        fittest = main_loop(sets=sets, n_nodes=15, n_individuals=100, threshold=0.9, verbose=True)

        test_acc = fittest.__validate__(sets['test'])
        print '%d accuracy: %+0.2f' % (i, test_acc)

        acc += test_acc

        fittest.plot()

    print 'mean acc: %.2f' % (acc / float(n_folds))

    from matplotlib import pyplot as plt
    plt.show()

if __name__ == '__main__':
    main()
