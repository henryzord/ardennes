# coding=utf-8

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris


class Individual(object):

    _root = 0

    def __init__(self, pmf, n_leaf, sets, classes):
        self._n_internal = pmf.shape[0]
        self._n_leaf = n_leaf
        self._nodes = None
        self._threshold = np.empty(self._n_internal + self._n_leaf, dtype=np.float32)
        self._sets = sets
        self._classes = classes
        self._train_error = 0
        self._val_acc = 0

        self.sample(pmf)

    def sample(self, pmf):
        self._nodes = map(lambda x: np.random.choice(pmf.shape[1], p=x), pmf)
        self._train_error = self.__set_internal__(self._sets['train'], self._root)
        self._val_acc = self.validate(self._sets['val'])

    def validate(self, set):
        # TODO make code faster!

        hit_count = 0
        for obj in set:
            arg_node = 0
            while not self.is_leaf(arg_node):
                attr = self._nodes[arg_node]

                arg_left, arg_right = self.__argchildren__(arg_node)
                go_left = obj[attr] < self._threshold[arg_node]
                arg_node = arg_left if go_left else arg_right

            hit_count += obj[-1] == self._threshold[arg_node]

        acc = hit_count / float(set.shape[0])
        return acc

    def is_internal(self, pos):
        return pos is not None and ((2*pos) + 2) < self._n_internal

    def is_leaf(self, pos):
        return not self.is_internal(pos)

    @staticmethod
    def __argchildren__(pos):
        return (2*pos) + 1, (2*pos) + 2

    def __children__(self, pos):
        left, right = self.__argchildren__(pos)
        return pos[left], pos[right]

    def __set_internal__(self, subset, node):
        """
        Sets the threshold for the whole tree.

        :param subset: Subset of objects that reach the current node.
        :param node:
        :return:
        """
        if subset.shape[0] <= 0:
            raise StandardError('empty subset!')

        if self.is_internal(node):
            arg_left, arg_right = self.__argchildren__(node)

            arg_attr = self._nodes[node]  # arg_attr is the attribute chosen for split for the given node

            # TODO make verification of values more smart!
            # TODO verify only values where the class of adjacent objects changes!

            unique_vals = np.sort(np.unique(subset[:, arg_attr]))

            t_counter = 1
            max_t = unique_vals.shape[0]

            best_entropy = np.inf

            if unique_vals.shape[0] < 3:
                error = self.__set_node__(node, subset)
                return error

            while t_counter < max_t - 1:  # may not pick the limitrophe values, since it would generate empty sets
                threshold = unique_vals[t_counter]

                subset_left = subset[subset[:, arg_attr] < threshold]
                subset_right = subset[subset[:, arg_attr] >= threshold]

                entropy = \
                    self.entropy(subset_left) + \
                    self.entropy(subset_right)

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_threshold = threshold
                    best_subset_left = subset_left
                    best_subset_right = subset_right

                t_counter += 1

            self._threshold[node] = best_threshold

            # TODO todo!
            # TODO make method also predict for training!

            error_left = self.__set_internal__(best_subset_left, arg_left)
            error_right = self.__set_internal__(best_subset_right, arg_right)
            return error_left + error_right
        else:
            error = self.__set_node__(node, subset)
            return error

    def __set_node__(self, node, subset):
        count = Counter(subset[:, -1])

        f_key = None
        f_val = -np.inf
        for key, val in count.iteritems():
            if val > f_val:
                f_key = key
                f_val = val

        self._threshold[node] = f_key  # not a threshold, but a class instead!
        error = (subset.shape[0] - f_val) / float(subset.shape[0])
        return error

    def __str__(self):
        return 'val accuracy: %0.2f attributes: %s' % (self._val_acc, str(self._nodes))

    # the smaller, the better

    @staticmethod
    def gini(subset):
        raise NotImplementedError('not implemented yet!')

    @staticmethod
    def entropy(subset):
        size = float(subset.shape[0])

        counter = Counter(subset[:, -1])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy


def get_pmf(n_attributes, n_internal):
    # checks if n_nodes is a valid value for the tree
    if n_internal < 1 or not np.log2(n_internal + 1).is_integer():  # TODO verify integrity!
        raise ValueError(
            'Invalid value for n_nodes! Must be a power of 2, minus 1 (e.g, 1, 3, 7).'
        )

    pmf = np.ones((n_internal, n_attributes), dtype=np.float32) / n_attributes
    return pmf


def set_pmf(pmf, fittest):
    raise NotImplementedError('not implemented  yet!')


def sample_population(pmf, to_replace):
    raise NotImplementedError('not implemented yet!')
    to_replace = map(lambda x: x.sample(pmf), to_replace)
    return to_replace


def init_pop(pmf, n_leaf, n_individuals, sets, classes):
    # TODO leaf nodes do not require this position in the array!

    pop = map(lambda x: Individual(pmf, n_leaf, sets, classes), xrange(n_individuals))
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


def get_node_distribution(n_nodes):
    n_leaf = (n_nodes + 1) / 2
    n_internal = n_nodes - n_leaf
    return n_internal, n_leaf


def main_loop(sets, classes, n_nodes, n_individuals, n_iterations=100):
    n_attributes = sets['train'].shape[1] - 1  # discards target attribute

    n_internal, n_leaf = get_node_distribution(n_nodes)

    pmf = get_pmf(n_attributes, n_internal)  # pmf has one distribution for each node
    population = init_pop(pmf, n_leaf, n_individuals, sets, classes)

    iteration = 0
    while iteration < n_iterations:
        # evolutionary process
        iteration += 1

if __name__ == '__main__':
    import random
    random.seed(1)
    np.random.seed(1)

    share = {'train': 0.8, 'test': 0.1, 'val': 0.1}
    data = load_iris()
    X, Y = data['data'], data['target']

    classes = np.unique(X)
    sets = get_sets(X, Y, share)

    main_loop(sets=sets, classes=classes, n_nodes=7, n_individuals=100)
