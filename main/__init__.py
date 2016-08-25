# coding=utf-8

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import pandas as pd
import itertools as it
from matplotlib import pyplot as plt


from genotype import Individual
import networkx as nx

__author__ = 'Henry Cagnini'


def get_pmf(pred_attr, target, target_prob):
    n_pred = pred_attr.shape[0]
    pred_prob = (1. - target_prob) / n_pred

    pmf = nx.DiGraph()

    pmf.add_node(target, prob=target_prob)

    for at in pred_attr:
        pmf.add_node(at, prob=pred_prob)

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


def get_raw_pmf(pmf):
    raw_pmf = pmf.nodes(data=True)
    raw_pmf = {x: y['prob'] for x, y in raw_pmf}
    return raw_pmf


def init_pop(n_individuals, pmf, sets):
    # TODO implement with threading.

    raw_pmf = get_raw_pmf(pmf)

    pop = np.array(
        map(
            lambda x: Individual(
                id=x,
                raw_pmf=raw_pmf,
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


def early_stop(pmf, diff=0.01):
    """

    :type pmf: networkx.DiGraph
    :param pmf:
    :return:
    """
    # TODO implement!
    return False


def get_node_count(n_nodes):
    n_leaf = (n_nodes + 1) / 2
    n_internal = n_nodes - n_leaf
    return n_internal, n_leaf


def main_loop(sets, n_individuals, target_prob, n_iterations=100, inf_thres=0.9, diff=0.01, verbose=True):
    pred_attr = sets['train'].columns[:-1]
    target = sets['train'].columns[-1]

    pmf = get_pmf(pred_attr, target, target_prob)  # pmf has one distribution for each node

    population = init_pop(
        n_individuals=n_individuals,
        pmf=pmf,
        sets=sets
    )

    fitness = np.array(map(lambda x: x.fitness, population))

    # threshold where individuals will be picked for PMF updatting/replacing
    integer_threshold = int(inf_thres * n_individuals)

    n_past = 15
    past = np.random.rand(n_past)

    iteration = 0
    while iteration < n_iterations:  # evolutionary process
        mean = np.mean(fitness)  # type: float
        median = np.median(fitness)  # type: float
        _max = np.max(fitness)  # type: float

        if verbose:
            print 'mean: %+0.6f\tmedian: %+0.6f\tmax: %+0.6f' % (mean, median, _max)

        try:
            past = early_stop(pmf, diff)
        except StopIteration:
            break

        borderline = np.partition(fitness, integer_threshold)[integer_threshold]  # TODO slow. test other implementation!
        fittest = population[np.flatnonzero(fitness >= borderline)]  # TODO slow. test other implementation!
        pmf = set_pmf(pmf, fittest)

        to_replace = population[np.flatnonzero(fitness < borderline)]  # TODO slow. test other implementation!
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

    random_state = 1

    random.seed(random_state)
    np.random.seed(random_state)

    n_individuals = 10
    n_folds = 10
    n_run = 1
    diff = 0.01
    df, folds = get_iris(n_folds=n_folds)  # iris dataset
    # df, folds = get_bank(n_folds=2)  # bank dataset

    overall_acc = 0.

    for i, (arg_train, arg_test) in enumerate(folds):
        fold_acc = 0.
        for j in xrange(n_run):

            x_train, x_val, y_train, y_val = train_test_split(
                df.iloc[arg_train][df.columns[:-1]],
                df.iloc[arg_train][df.columns[-1]],
                test_size=1./(n_folds - 1.),
                random_state=random_state
            )

            train_set = x_train.join(y_train)
            val_set = x_val.join(y_val)

            sets = {'train': train_set, 'val': val_set, 'test': df.iloc[arg_test]}

            fittest = main_loop(sets=sets, n_individuals=n_individuals, target_prob=0.55, inf_thres=0.9, diff=diff, verbose=False)

            test_acc = fittest.__validate__(sets['test'])
            print 'fold: %d run: %d accuracy: %0.2f' % (i, j, test_acc)

            fold_acc += test_acc
            overall_acc += test_acc

        print '%d-th fold mean accuracy: %0.2f' % (i, fold_acc / float(n_run))

    print 'overall mean acc: %.2f' % (overall_acc / float(n_folds))

    # plt.show()

if __name__ == '__main__':
    main()
