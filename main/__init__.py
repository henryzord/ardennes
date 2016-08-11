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
    population = init_pop(
        n_individuals=n_individuals,
        n_leaf=n_leaf,
        n_internal=n_internal,
        pmf=pmf,
        classes=classes,
        sets=sets,
        attributes=attributes
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


def get_iris(share):
    data = load_iris()
    attributes = {i: {'name': x, 'type': 'numerical'} for i, x in enumerate(data['feature_names'])}
    class_names = {i: x for i, x in enumerate(data['target_names'])}

    X, Y = data['data'], data['target']

    classes = np.unique(Y)
    sets = get_sets(X, Y, share)

    return attributes, class_names, classes, sets


def get_bank(share):
    dataset = pd.read_csv('/home/henryzord/Projects/forrestTemp/datasets/bank-full_no_missing.csv', sep=',')

    column_names = dataset.columns[:-1].tolist()
    # column_names = [u'age', u'job', u'marital', u'education', u'default', u'balance', \
    # u'housing', u'loan', u'contact', u'day', u'month', u'duration', \
    # u'campaign', u'pdays', u'previous', u'poutcome']
    types = {0: 'categorical', 1: 'categorical', 2: 'categorical', 3: 'categorical',
             4: 'categorical', 5: 'numerical', 6: 'numerical', 7: 'numerical',
             8: 'numerical', 9: 'categorical', 10: 'categorical', 11: 'numerical',
             12: 'numerical', 13: 'numerical', 14: 'numerical', 15: 'categorical',
             16: 'numerical', 17: 'numerical', 18: 'numerical', 19: 'numerical',
             20: 'numerical'}

    attributes = {i: {'name': x, 'type': t} for i, (x, t) in enumerate(it.izip(column_names, types.itervalues()))}
    X = dataset[column_names]
    Y = dataset[dataset.columns[-1]]

    exit(-1)


def main():
    share = {'train': 0.8, 'test': 0.1, 'val': 0.1}

    # attributes, class_names, classes, sets = get_iris(share)
    attributes, class_names, classes, sets = get_bank(share)

    fittest = main_loop(
        n_individuals=100,
        n_nodes=7,
        sets=sets,
        classes=classes,
        attributes=attributes,
        threshold=0.9,
        verbose=True
    )

    for name, jset in sets.iteritems():
        acc = fittest.__validate__(jset)
        print '%s accuracy: %+0.2f' % (name, acc)

    fittest.plot(class_names)

if __name__ == '__main__':
    main()
