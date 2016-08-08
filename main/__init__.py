# coding=utf-8
from collections import Counter

from sklearn.datasets import load_iris
import numpy as np


class Individual(object):

    _root = 0

    def __init__(self, pmf, sets, classes):
        self._n_nodes = pmf.shape[1]
        self._nodes = None
        self._threshold = np.empty(self._n_nodes, dtype=np.float32)
        self._sets = sets
        self._classes = classes

        self.sample(pmf)

    def sample(self, pmf):
        self._nodes = map(lambda x: np.random.choice(pmf.shape[1], p=x), pmf)
        self.set_threshold(sets['train'], self._root)

    def has_children(self, pos):
        return pos is not None and ((2*pos) + 2) < self._n_nodes

    @staticmethod
    def argchildren(pos):
        return (2*pos) + 1, (2*pos) + 2

    def children(self, pos):
        left, right = self.argchildren(pos)
        return pos[left], pos[right]

    def set_threshold(self, subset, node):
        """
        Sets the threshold for the whole tree.

        :param subset: Subset of objects that reach the current node.
        :param node:
        :return:
        """

        # TODO make method also predict for training!

        arg_attr = self._nodes[node]

        # TODO make verification of values more smart!
        # TODO verify only values where the class of adjacent objects changes!

        unique_vals = np.sort(np.unique(subset[:, arg_attr]))

        t_counter = 0
        max_t = unique_vals.shape[0]

        if max_t <= 0:
            return  # TODO no objects reaching this node!

        best_threshold = unique_vals[t_counter]

        entropy_left = self.entropy(subset[subset[:, arg_attr] < best_threshold])
        entropy_right = self.entropy(subset[subset[:, arg_attr] >= best_threshold])

        best_entropy = entropy_left + entropy_right

        while t_counter < max_t - 1:
            t_counter += 1
            threshold = unique_vals[t_counter]

            entropy_left = self.entropy(subset[subset[:, arg_attr] < threshold])
            entropy_right = self.entropy(subset[subset[:, arg_attr] >= threshold])
            entropy = entropy_left + entropy_right

            # print 'entropy: %f\tthreshold:%f\targ:%d\t' % (entropy, threshold, arg)

            if entropy < best_entropy:
                best_entropy = entropy
                best_threshold = threshold

        self._threshold[node] = best_threshold
        if self.has_children(node):
            arg_left, arg_right = self.argchildren(node)

            subset_left = subset[subset[:, arg_attr] < best_threshold]
            subset_right = subset[subset[:, arg_attr] >= best_threshold]

            self.set_threshold(subset_left, arg_left)
            self.set_threshold(subset_right, arg_right)

    def __str__(self):
        return ' ' + str(self._nodes)

    # the smaller, the better

    @staticmethod
    def entropy(subset):
        size = float(subset.shape[0])

        counter = Counter(subset[:, 1])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy


def get_pmf(n_attributes, n_nodes):
    # checks if n_nodes is a valid value for the tree
    if n_nodes < 1 or not np.log2(n_nodes + 1).is_integer():  # TODO verify integrity!
        raise ValueError(
            'Invalid value for n_nodes! Must be a power of 2, minus 1 (e.g, 1, 3, 7).'
        )

    pmf = np.ones((n_nodes, n_attributes), dtype=np.float32) / n_attributes
    return pmf


def set_pmf(pmf, fittest):
    pass


def sample_population(pmf, n_individuals, sets, classes):
    pop = map(lambda x: Individual(pmf, sets, classes), xrange(n_individuals))
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


def main_loop(sets, classes, n_nodes, n_individuals, n_iterations=100):
    n_attributes = sets['train'].shape[1]

    pmf = get_pmf(n_attributes, n_nodes)  # pmf has one distribution for each node
    population = sample_population(pmf, n_individuals, sets, classes)

    iteration = 0
    while iteration < n_iterations:
        for ind in population:
            for node in ind:
                pass
                # entropy()

        iteration += 1


if __name__ == '__main__':
    share = {'train': 0.8, 'test': 0.1, 'val': 0.1}
    data = load_iris()
    X, Y = data['data'], data['target']

    classes = np.unique(X)
    sets = get_sets(X, Y, share)

    main_loop(sets=sets, classes=classes, n_nodes=7, n_individuals=100)
