# coding=utf-8

import numpy as np
from collections import Counter
import networkx as nx

__author__ = 'Henry Cagnini'


class Individual(object):

    _root = 0

    def __init__(self, raw_pmf, sets):
        """

        :type raw_pmf: dict
        :param raw_pmf:
        :type sets: dict
        :param sets:
        """

        self._sets = sets  # type: dict

        self._class_name = sets['train'].columns[-1]  # type: str

        self._column_types = {
            x: self.type_handler_dict[str(self._sets['train'][x].dtype)] for x in self._sets['train'].columns
        }  # type: dict

        self._tree = None  # type: nx.DiGraph
        self._train_acc = 0.  # type: float
        self._val_acc = 0.  # type: float

        self.sample(raw_pmf)

    @property
    def fitness(self):
        """
        :rtype: float
        :return: Fitness of this individual.
        """

        return self._val_acc

    def sample(self, raw_pmf):
        """

        :type raw_pmf: dict
        :param raw_pmf:
        :return:
        """

        tree = nx.DiGraph()

        tree = self.sample_node(
            raw_pmf=raw_pmf,
            tree=tree,
            id_current=0,
            id_parent=None,
            force_predictive=True
        )

        # self._tree = tree
        # self.plot()

        dict_threshold = self.__set_internal__(
            id_node=Individual._root,
            dict_threshold={},
            tree=tree,
            subset=self._sets['train']
        )

        nx.set_node_attributes(tree, name='threshold', values=dict_threshold)
        self._tree = tree
        self.plot()

        raise NotImplementedError('implement!')

        # self._train_acc = self.__set_internal__(self._sets['train'], self._root)
        # self._val_acc = self.__validate__(self._sets['val'])

    def sample_node(self, raw_pmf, tree, id_current, id_parent=None, force_predictive=False):
        """

        :param id_parent:
        :type raw_pmf: dict
        :type tree: networkx.DiGraph
        :type id_current: int
        :type force_predictive: bool
        :return:
        """

        label_current = np.random.choice(a=raw_pmf.keys(), p=raw_pmf.values())
        if force_predictive:
            while label_current == self._class_name:
                label_current = np.random.choice(a=raw_pmf.keys(), p=raw_pmf.values())

        # color setting
        if label_current == self._class_name:
            node_color = '#98FB98'
        elif id_parent is not None:
            node_color = '#AEEAFF'
        else:
            node_color = '#FFFFFF'

        tree.add_node(id_current, label=label_current, color=node_color)
        if id_parent is not None:
            tree.add_edge(id_parent, id_current, attr_dict={'threshold': None})

        if label_current != self._class_name:
            id_left = (id_current * 2) + 1
            id_right = (id_current * 2) + 2
            tree = self.sample_node(raw_pmf, tree, id_left, id_current, force_predictive=False)
            tree = self.sample_node(raw_pmf, tree, id_right, id_current, force_predictive=False)

        return tree

    def __validate__(self, set):
        raise NotImplementedError('not implemented yet!')

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

    def __set_internal__(self, id_node, dict_threshold, tree, subset):
        # TODO make verification of values more smart!
        # TODO verify only values where the class of adjacent objects changes!

        if subset.shape[0] <= 0:
            raise StandardError('empty subset!')
            # TODO treat this case! clear all subtrees from this one! make this one a class!

        attr_name = tree.node[id_node]['label']

        if attr_name != self._class_name:
            attr_type = self._column_types[attr_name]
            dict_threshold = self.attr_handler_dict[attr_type](self, id_node, attr_name, dict_threshold, tree, subset)
        else:
            dict_threshold = self.__set_terminal__(id_node, dict_threshold, subset)

        return dict_threshold

    def __set_categorical__(self, id_node, attr_name, dict_threshold, tree, subset):
        raise NotImplemented('not implemented yet!')

    def __set_numerical__(self, id_node, attr_name, dict_threshold, tree, subset):
        """

        :type tree: networkx.DiGraph
        :param tree:
        :param dict_threshold:
        :param id_node:
        :param attr_name:
        :param subset:
        :return:
        """

        unique_vals = subset[attr_name].sort_values().unique()

        t_counter = 1
        max_t = unique_vals.shape[0]

        best_entropy = np.inf

        if unique_vals.shape[0] < 3:
            acc = self.__set_terminal__(id_node, dict_threshold, subset)
            return acc

        best_threshold = None
        best_subset_left = None
        best_subset_right = None

        while t_counter < max_t - 1:  # should not pick limitrophe values, since it generates empty sets
            threshold = unique_vals[t_counter]

            subset_left = subset.loc[subset[attr_name] < threshold]
            subset_right = subset.loc[subset[attr_name] >= threshold]

            entropy = \
                self.entropy(subset_left) + \
                self.entropy(subset_right)

            if entropy < best_entropy:
                best_entropy = entropy
                best_threshold = threshold
                best_subset_left = subset_left
                best_subset_right = subset_right

            t_counter += 1

        dict_threshold[id_node] = best_threshold

        if tree.out_degree(id_node) > 0:
            children = tree.successors(id_node)
            id_left = min(children)
            id_right = max(children)

            dict_threshold = self.__set_internal__(id_left, dict_threshold, tree, best_subset_left)
            dict_threshold = self.__set_internal__(id_right, dict_threshold, tree, best_subset_right)

        return dict_threshold

    def __set_error__(self, id_node, attr_name, dict_threshold, tree, subset):
        raise TypeError('Unsupported data type for column %s!' % attr_name)

    def __set_terminal__(self, id_node, dict_threshold, subset):
        count = Counter(subset[self._class_name])

        f_key = None
        f_val = -np.inf
        for key, val in count.iteritems():
            if val > f_val:
                f_key = key
                f_val = val

        dict_threshold[id_node] = f_key  # not a threshold, but a class instead!
        return dict_threshold

    def entropy(self, subset):
        # the smaller, the better
        size = float(subset.shape[0])

        counter = Counter(subset[self._class_name])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy

    def plot(self):
        from matplotlib import pyplot as plt

        plt.figure()

        tree = self._tree  # type: nx.DiGraph
        pos = nx.spectral_layout(tree)

        node_list = tree.nodes(data=True)
        edge_list = tree.edges()

        node_labels = {x[0]: x[1]['label'] for x in node_list}
        node_colors = [x[1]['color'] for x in node_list]
        edge_labels = {}

        nx.draw_networkx_nodes(tree, pos, node_size=1000, node_color=node_colors)  # nodes
        nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
        nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)

        plt.axis('off')
        plt.show()  # TODO remove later!

    def __str__(self):
        return 'fitness: %0.2f' % self._val_acc

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
