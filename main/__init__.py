# coding=utf-8

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
import pandas as pd

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


def init_pop(n_individuals, n_leaf, n_internal, pmf, classes, sets, attributes):
    pop = np.array(
        map(
            lambda x: Individual(
                n_leaf=n_leaf,
                n_internal=n_internal,
                pmf=pmf,
                classes=classes,
                sets=sets,
                attributes=attributes
            ),
            xrange(n_individuals)
        )
    )
    return pop


def get_sets(X, Y, share, random_state=None):

    from sklearn.cross_validation import train_test_split
    x_train, _x, y_train, _y = \
        train_test_split(X, Y, train_size=share['train'], random_state=random_state) \
        if random_state is not None else \
        train_test_split(X, Y, train_size=share['train'])

    x_test, x_val, y_test, y_val = \
        train_test_split(_x, _y, train_size=share['test'] / (share['test'] + share['val']), random_state=random_state) \
        if random_state is not None else \
        train_test_split(_x, _y, train_size=share['test'] / (share['test'] + share['val']))

    return {
        'train': np.hstack((x_train, y_train[:, np.newaxis])),
        'test': np.hstack((x_test, y_test[:, np.newaxis])),
        'val': np.hstack((x_val, y_val[:, np.newaxis]))
    }


def early_stop(iteration, mean, median, past, n_past):
    past[iteration % n_past] = mean
    if median == mean and np.all(past == mean):
        raise StopIteration('you should stop.')
    else:
        return past


def get_node_distribution(n_nodes):
    n_leaf = (n_nodes + 1) / 2
    n_internal = n_nodes - n_leaf
    return n_internal, n_leaf


def main_loop(n_individuals, n_nodes, sets, classes, attributes, n_iterations=100, threshold=0.9, verbose=True):
    n_attributes = sets['train'].shape[1] - 1  # discards target attribute

    n_internal, n_leaf = get_node_distribution(n_nodes)

    pmf = get_pmf(n_attributes, n_internal)  # pmf has one distribution for each node
    population = init_pop(n_individuals, n_leaf, n_internal, pmf, classes, sets)
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


def main():
    # import random
    # random.seed(1)
    # np.random.seed(1)

    share = {'train': 0.8, 'test': 0.1, 'val': 0.1}
    data = load_iris()

    attributes = {i: {'name': x, 'type': 'numeral'} for i, x in enumerate(data['feature_names'])}
    class_names = {i: x for i, x in enumerate(data['target_names'])}

    X, Y = data['data'], data['target']

    classes = np.unique(X)
    sets = get_sets(X, Y, share)

    # print class_names
    # df = pd.DataFrame(sets['test'], columns=column_names + ['class'], index=None)
    # df['class'] = df['class'].apply(lambda x: class_names[x])
    # print df[['petal width (cm)', 'class']].sort('petal width (cm)')

    fittest = main_loop(n_individuals=100, n_nodes=7, sets=sets, classes=classes, attributes=attributes, threshold=0.9,
                        verbose=True)

    for name, set in sets.iteritems():
        acc = fittest.__validate__(set)
        print '%s accuracy: %+0.2f' % (name, acc)

    fittest.plot(attributes, class_names)

if __name__ == '__main__':
    main()
