# coding=utf-8

import numpy as np
from collections import Counter
import networkx as nx

__author__ = 'Henry Cagnini'


class Individual(object):

    _root = 0

    def __init__(self, n_leaf, n_internal, pmf, sets):
        self._n_internal = n_internal
        self._n_leaf = n_leaf
        self._sets = sets

        self._graph = None

        self._train_acc = 0.
        self._val_acc = 0.

        self._column_names = self._sets['train'].columns
        self._column_types = map(
            lambda x: self.type_handler_dict[str(self._sets['train'][x].dtype)],
            self._sets['train']
        )

        self._threshold = np.empty(self._n_internal + self._n_leaf, dtype=np.object)
        self._internal_nodes = None

        self.sample(pmf)

    @property
    def nodes(self):
        return self._internal_nodes

    @property
    def fitness(self):
        return self._val_acc

    def sample(self, pmf):
        self._graph = nx.DiGraph()

        root = np.random.choice(pmf, pmf.iloc[0])

        # self._graph.add_node()

        raise NotImplementedError('not implemented yet!')

        self._internal_nodes = map(lambda x: np.random.choice(pmf.shape[1], p=x), pmf)
        self._train_acc = self.__set_internal__(self._sets['train'], self._root)
        self._val_acc = self.__validate__(self._sets['val'])

    def __validate__(self, set):
        def val_func(obj):
            arg_node = 0
            while not self.is_leaf(arg_node):
                attr = self._internal_nodes[arg_node]

                arg_left, arg_right = self.__argchildren__(arg_node)
                go_left = obj[attr] < self._threshold[arg_node]
                arg_node = arg_left if go_left else arg_right

            return obj[-1] == self._threshold[arg_node]

        hit_count = set.apply(val_func, axis=1).sum()
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

    def __set_error__(self, node, attr_name, subset):
        raise TypeError('Unsupported data type for column %s!' % attr_name)

    def __set_categorical__(self, node, attr_name, subset):
        raise NotImplemented('not implemented yet!')

        arg_left, arg_right = self.__argchildren__(node)

        unique_vals = np.sort(np.unique(subset[:, [-1, arg_attr]]))

    def __set_numerical__(self, node, attr_name, subset):
        arg_left, arg_right = self.__argchildren__(node)

        unique_vals = subset[attr_name].sort_values().unique()

        t_counter = 1
        max_t = unique_vals.shape[0]

        best_entropy = np.inf

        if unique_vals.shape[0] < 3:
            acc = self.__set_node__(node, subset)
            return acc

        best_threshold = None
        best_subset_left = None
        best_subset_right = None

        while t_counter < max_t - 1:  # should not pick limitrophe values, since it generates empty sets
            threshold = unique_vals[t_counter]

            subset_left = subset.loc[subset[attr_name] < threshold]  # TODO replace by pandas!
            subset_right = subset.loc[subset[attr_name] >= threshold]  # TODO replace by pandas!

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

        acc_left = self.__set_internal__(best_subset_left, arg_left)
        acc_right = self.__set_internal__(best_subset_right, arg_right)
        return (acc_left + acc_right) / 2.

    def __set_internal__(self, subset, node):
        """
        Sets the threshold for the whole tree.

        :param subset: Subset of objects that reach the current node.
        :param node:
        :return:
        """

        # TODO make verification of values more smart!
        # TODO verify only values where the class of adjacent objects changes!

        if subset.shape[0] <= 0:
            raise StandardError('empty subset!')

        if self.is_internal(node):
            arg_attr = self._internal_nodes[node]  # arg_attr is the attribute chosen for split for the given node

            attr_name = self._column_names[arg_attr]
            attr_type = self._column_types[arg_attr]

            acc = self.attr_handler_dict[attr_type](self, node, attr_name, subset)
        else:
            acc = self.__set_node__(node, subset)

        return acc

    def __set_node__(self, node, subset):
        count = Counter(subset[self._column_names[-1]])

        f_key = None
        f_val = -np.inf
        for key, val in count.iteritems():
            if val > f_val:
                f_key = key
                f_val = val

        self._threshold[node] = f_key  # not a threshold, but a class instead!
        acc = f_val / float(subset.shape[0])
        return acc

    def plot(self):
        from matplotlib import pyplot as plt
        import networkx as nx

        plt.figure()

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

            node_labels[i] = self._column_names[self.nodes[i]]

            colors[i] = '#CCFFFF'

            if self.is_leaf(left):
                node_labels[left] = self._threshold[left]
                colors[left] = '#CCFF99'

            if self.is_leaf(right):
                node_labels[right] = self._threshold[right]
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
        # plt.show()

    def __str__(self):
        return 'train accuracy: %0.2f attributes: %s' % (self._train_acc, str(self._internal_nodes))

    # the smaller, the better

    def entropy(self, subset):
        size = float(subset.shape[0])

        counter = Counter(subset[self._column_names[-1]])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy

    type_handler_dict = {
        'bool': 'bool',
        'bool_': 'bool',
        'int': 'int',
        'int_': 'int',
        'intc': 'int',
        'intp': 'int',
        'int8': 'int',
        'int16': 'int',
        'int32': 'int',
        'int64': 'int',
        'uint8': 'int',
        'uint16': 'int',
        'uint32': 'int',
        'uint64': 'int',
        'float': 'float',
        'float_': 'float',
        'float16': 'float',
        'float32': 'float',
        'float64': 'float',
        'complex_': 'complex',
        'complex64': 'complex',
        'complex128': 'complex',
        'str': 'str',
        'object': 'object'
    }

    attr_handler_dict = {
        'object': __set_categorical__,
        'str': __set_categorical__,
        'int': __set_numerical__,
        'float': __set_numerical__,
        'bool': __set_categorical__,
        'complex': __set_error__

    }
