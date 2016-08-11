# coding=utf-8

import numpy as np
from collections import Counter

__author__ = 'Henry Cagnini'


class Individual(object):

    _root = 0

    def __init__(self, n_leaf, n_internal, pmf, classes, sets, attributes):
        self._n_internal = n_internal
        self._n_leaf = n_leaf
        self._classes = classes
        self._sets = sets
        self._attributes = attributes

        self._train_acc = 0
        self._val_acc = 0

        self._threshold = np.empty(self._n_internal + self._n_leaf, dtype=np.float32)
        self._internal_nodes = None

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

    @staticmethod
    def __set_categorical__(attr, subset):
        pass

    @staticmethod
    def __set_numerical__(attr, subset):
        pass

    def __set_internal__(self, subset, node):
        """
        Sets the threshold for the whole tree.

        :param subset: Subset of objects that reach the current node.
        :param node:
        :return:
        """

        # TODO only deals with real-valued attributes that keep an order between itself!
        # TODO must threat other attribute types, such as categorical

        if subset.shape[0] <= 0:
            raise StandardError('empty subset!')

        if self.is_internal(node):
            arg_left, arg_right = self.__argchildren__(node)

            arg_attr = self._internal_nodes[node]  # arg_attr is the attribute chosen for split for the given node

            # TODO dictionary for attr!

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
                    Individual.entropy(subset_left) + \
                    Individual.entropy(subset_right)

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_threshold = threshold
                    best_subset_left = subset_left
                    best_subset_right = subset_right

                t_counter += 1

            self._threshold[node] = best_threshold

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

    def plot(self, attributes, class_names):
        from matplotlib import pyplot as plt
        import networkx as nx

        G = nx.DiGraph()

        edges = dict()
        node_labels = dict()
        edge_labels = dict()
        colors = dict()

        for i in xrange(self._n_internal):
            left, right = self.__argchildren__(i)
            edges[left] = i
            edges[right] = i

            edge_labels[(i, left)] = '< %.2f' % self._threshold[i]
            edge_labels[(i, right)] = '>= %.2f' % self._threshold[i]

            node_labels[i] = attributes[self.nodes[i]]['name']

            colors[i] = '#CCFFFF'

            if self.is_leaf(left):
                node_labels[left] = class_names[int(self._threshold[left])]
                colors[left] = '#CCFF99'

            if self.is_leaf(right):
                node_labels[right] = class_names[int(self._threshold[right])]
                colors[right] = '#CCFF99'

        colors[0] = '#FFFFFF'
        edges = map(lambda x: x[::-1], edges.iteritems())

        G.add_nodes_from(xrange(self._n_internal + self._n_leaf))

        G.add_edges_from(edges)

        pos = nx.spectral_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=colors.values())  # nodes
        nx.draw_networkx_edges(G, pos, edgelist=edges, style='dashed')
        nx.draw_networkx_labels(G, pos, node_labels, font_size=16)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=16)

        plt.axis('off')
        plt.show()

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

    handler_dict = {
        'categorical': __set_categorical__,
        'numerical': __set_numerical__
    }
