# coding=utf-8

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris


class Individual(object):

    _root = 0

    def __init__(self, pmf, n_leaf, sets, classes):
        self._n_internal = pmf.shape[0]
        self._n_leaf = n_leaf
        self._internal_nodes = None
        self._threshold = np.empty(self._n_internal + self._n_leaf, dtype=np.float32)
        self._sets = sets
        self._classes = classes
        self._train_acc = 0
        self._val_acc = 0

        self.sample(pmf)

    @property
    def nodes(self):
        return self._internal_nodes

    @property
    def fitness(self):
        return self._val_acc

    def sample(self, pmf):
        self._internal_nodes = map(lambda x: np.random.choice(pmf.shape[1], p=x), pmf)
        self._train_acc = self.__set_internal__(self._sets['train'], self._root)
        self._val_acc = self.__validate__(self._sets['val'])

    def __validate__(self, set):
        # TODO make code faster!

        hit_count = 0
        for obj in set:
            arg_node = 0
            while not self.is_leaf(arg_node):
                attr = self._internal_nodes[arg_node]

                arg_left, arg_right = self.__argchildren__(arg_node)
                go_left = obj[attr] < self._threshold[arg_node]
                arg_node = arg_left if go_left else arg_right

            hit_count += obj[-1] == self._threshold[arg_node]

        acc = hit_count / float(set.shape[0])
        return acc

    def predict(self):
        raise NotImplementedError('not implemented yet!')

    def is_internal(self, pos):
        return pos is not None and ((2*pos) + 2) < self._n_internal + self._n_leaf

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

            arg_attr = self._internal_nodes[node]  # arg_attr is the attribute chosen for split for the given node

            # TODO make verification of values more smart!
            # TODO verify only values where the class of adjacent objects changes!

            unique_vals = np.sort(np.unique(subset[:, arg_attr]))

            t_counter = 1
            max_t = unique_vals.shape[0]

            best_entropy = np.inf

            if unique_vals.shape[0] < 3:
                acc = self.__set_node__(node, subset)
                return acc

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

            acc_left = self.__set_internal__(best_subset_left, arg_left)
            acc_right = self.__set_internal__(best_subset_right, arg_right)
            return acc_left + acc_right
        else:
            acc = self.__set_node__(node, subset)
            return acc

    def __set_node__(self, node, subset):
        count = Counter(subset[:, -1])

        f_key = None
        f_val = -np.inf
        for key, val in count.iteritems():
            if val > f_val:
                f_key = key
                f_val = val

        self._threshold[node] = f_key  # not a threshold, but a class instead!
        acc = f_val / float(subset.shape[0])
        return acc

    def plot(self, column_names, class_names):
        from matplotlib import pyplot as plt
        import networkx as nx

        G = nx.Graph()

        for i, node in enumerate(self._internal_nodes):
            left, right = self.__argchildren__(i)

            # G.add_edge(i, left)
            # G.add_edge(i, right)

            left_name = \
                ('%d: ' % left) + (
                    column_names[self._internal_nodes[left]]
                    if self.is_internal(left)
                    else class_names[self._threshold[left]]
                )
            right_name = \
                ('%d: ' % right) + (
                    column_names[self._internal_nodes[right]]
                    if self.is_internal(right)
                    else class_names[self._threshold[right]]
                )

            self_name = ('%d: ' % node) + column_names[node]

            G.add_edge(self_name, left_name)
            G.add_edge(self_name, right_name)

        edges = [(u, v) for (u, v, d) in G.edges(data=True)]

        pos = nx.spring_layout(G)  # positions for all nodes

        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='#CCFFFF')  # nodes
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3)  # edges
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')  # labels

        plt.axis('off')
        # plt.savefig("weighted_graph.png")  # save as png
        plt.show()  # display

    def __str__(self):
        return 'val accuracy: %0.2f attributes: %s' % (self._val_acc, str(self._internal_nodes))

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


def init_pop(pmf, n_leaf, n_individuals, sets, classes):
    pop = np.array(map(lambda x: Individual(pmf, n_leaf, sets, classes), xrange(n_individuals)))
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


def main_loop(sets, classes, n_nodes, n_individuals, threshold=0.9, n_iterations=100, verbose=True):
    n_attributes = sets['train'].shape[1] - 1  # discards target attribute

    n_internal, n_leaf = get_node_distribution(n_nodes)

    pmf = get_pmf(n_attributes, n_internal)  # pmf has one distribution for each node
    population = init_pop(pmf, n_leaf, n_individuals, sets, classes)
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
    import random
    random.seed(1)
    np.random.seed(1)

    share = {'train': 0.8, 'test': 0.1, 'val': 0.1}
    data = load_iris()

    class_names = data['target_names']
    column_names = data['feature_names']

    X, Y = data['data'], data['target']

    classes = np.unique(X)
    sets = get_sets(X, Y, share)

    fittest = main_loop(sets=sets, classes=classes, n_nodes=7, threshold=0.9, n_individuals=100, verbose=True)
    acc = fittest.__validate__(sets['test'])
    print 'test accuracy: %+0.2f' % acc
    fittest.plot(column_names, class_names)

if __name__ == '__main__':
    main()
