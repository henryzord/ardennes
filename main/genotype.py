# coding=utf-8

import numpy as np
from collections import Counter
import networkx as nx
import pandas as pd

__author__ = 'Henry Cagnini'


class Node(object):
    _root = 0

    @property
    def root(self):
        return self._root

    @staticmethod
    def get_left_child(id_node):
        return (id_node * 2) + 1

    @staticmethod
    def get_right_child(id_node):
        return (id_node * 2) + 2

    @staticmethod
    def get_parent(id_node):
        if id_node > 0:
            return int((id_node - 1) / 2.)
        return None

    @staticmethod
    def get_depth(id_node):
        """
        Gets depth of node in a binary heap.
        :param id_node: ID of the node in the binary heap.
        :return: The depth of the node.
        """
        return int(np.log2(id_node + 1))


class Individual(object):
    target_attr = None
    target_values = None
    column_types = None

    def __init__(self, ind_id, initial_pmf, sets):
        """

        :type initial_pmf: dict
        :param initial_pmf:
        :type sets: dict
        :param sets:
        """

        self._id = ind_id

        # common values to any Individual
        if any(map(lambda x: x is None, [Individual.target_attr, Individual.target_values, Individual.column_types])):
            Individual._sets = sets  # type: dict

            Individual.target_attr = sets['train'].columns[-1]  # type: str
            Individual.target_values = sets['train'][sets['train'].columns[-1]].unique()
            Individual.column_types = {
                x: self.type_handler_dict[str(Individual._sets['train'][x].dtype)] for x in Individual._sets['train'].columns
            }  # type: dict

        # only for pycharm variable resolving; won't result in error if removed
        self._sets = Individual._sets
        self.target_attr = Individual.target_attr
        self.target_values = Individual.target_values
        self.column_types = Individual.column_types

        self._tree = None  # type: nx.DiGraph

        self._val_acc = 0.  # type: float

        self._sampler = Individual.Sampler(sets=self._sets)
        self.sample(initial_pmf)

    def sample(self, pmf):
        self._tree = self._sampler.sample(pmf)
        self._val_acc = self.__validate__(self._sets['val'])

    @property
    def fitness(self):
        """
        :rtype: float
        :return: Fitness of this individual.
        """
        return self._val_acc

    @property
    def tree(self):
        return self._tree

    @staticmethod
    def entropy(subset, target_attr):
        # the smaller, the better
        size = float(subset.shape[0])

        counter = Counter(subset[target_attr])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy

    def __val_func__(self, obj):
        arg_node = 0

        node = self._tree.node[arg_node]
        while not node['terminal']:
            go_left = obj[node['label']] < node['threshold']
            arg_node = (int(go_left) * node['left']) + (int(not go_left) * node['right'])
            node = self._tree.node[arg_node]

        return obj[-1] == node['label']

    def __validate__(self, _set):
        hit_count = _set.apply(self.__val_func__, axis=1).sum()
        acc = hit_count / float(_set.shape[0])
        return acc

    def plot(self):
        from matplotlib import pyplot as plt

        fig = plt.figure()

        tree = self._tree  # type: nx.DiGraph
        pos = nx.spectral_layout(tree)

        node_list = tree.nodes(data=True)
        edge_list = tree.edges(data=True)

        node_labels = {x[0]: x[1]['label'] for x in node_list}
        node_colors = [x[1]['color'] for x in node_list]
        edge_labels = {(x1, x2): d['threshold'] for x1, x2, d in edge_list}

        nx.draw_networkx_nodes(tree, pos, node_size=1000, node_color=node_colors)  # nodes
        nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
        nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)

        plt.text(
            0.9,
            0.9,
            '%0.4f' % self._val_acc,
            fontsize=15,
            horizontalalignment='center',
            verticalalignment='center',
            transform=fig.transFigure
        )

        plt.axis('off')
        # plt.show()

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

    class Sampler(object):

        _sets = None

        def __init__(self, sets):
            if Individual.Sampler._sets is None:
                Individual.Sampler._sets = sets

            self._sets = Individual.Sampler._sets

        def sample(self, pmf):
            tree = nx.DiGraph()

            subset = self._sets['train']

            tree = self.sample_node(
                pmf=pmf,
                tree=tree,
                id_current=0,
                subset=subset
            )

            return tree

        def sample_node(self, pmf, tree, id_current, subset):
            node_label = pmf.sample(id_node=id_current)
            if id_current == Node.root:
                while node_label == Individual.target_attr:
                    node_label = pmf.sample(id_node=id_current)

            if subset.shape[0] <= 0:
                raise ValueError('empty subset!')

            if node_label != Individual.target_attr:
                meta, subset_left, subset_right = self.__set_internal__(
                    node_label=node_label,
                    subset=subset
                )

                id_left = (id_current * 2) + 1
                id_right = (id_current * 2) + 2

                try:  # if one of the subsets is empty, then the node is terminal
                    tree = self.sample_node(pmf=pmf, tree=tree, id_current=id_left, subset=subset_left)
                    tree = self.sample_node(pmf=pmf, tree=tree, id_current=id_right, subset=subset_right)

                    if tree.node[id_left]['label'] == tree.node[id_right]['label'] and tree.node[id_left]['label'] in \
                            Individual.target_values:

                        tree.remove_node(id_left)
                        tree.remove_node(id_right)
                        raise ValueError('same class for terminal nodes!')

                    terminal = False
                    threshold = meta['value']

                    if id_current == Node.root:
                        node_color = '#FFFFFF'  # root color
                    else:
                        node_color = '#AEEAFF'  # inner node color

                    tree.add_edge(id_current, id_left, attr_dict={'threshold': '< %0.2f' % meta['value']})
                    tree.add_edge(id_current, id_right, attr_dict={'threshold': '>= %0.2f' % meta['value']})

                except ValueError:
                    meta = self.__set_terminal__(subset, Individual.target_attr)
                    terminal = True
                    threshold = None
                    node_label = meta['value']
                    id_left = None
                    id_right = None
                    node_color = '#98FB98'
            else:
                meta = self.__set_terminal__(subset, Individual.target_attr)
                terminal = True
                threshold = None
                node_label = meta['value']
                id_left = None
                id_right = None
                node_color = '#98FB98'

            tree.add_node(
                id_current,
                attr_dict={
                    'label': node_label,
                    'color': node_color,
                    'terminal': terminal,
                    'threshold': threshold,
                    'left': id_left,
                    'right': id_right
                }
            )
            return tree

        def __set_internal__(self, node_label, subset):
            if node_label != Individual.target_attr:
                attr_type = Individual.column_types[node_label]
                meta, subset_left, subset_right = self.attr_handler_dict[attr_type](self, node_label, subset)
            else:
                meta = self.__set_terminal__(subset, Individual.target_attr)
                subset_left = pd.DataFrame([])
                subset_right = pd.DataFrame([])

            return meta, subset_left, subset_right

        def __set_categorical__(self, node_label, subset):
            raise NotImplemented('not implemented yet!')

        # self, id_node, node_label, thresholds, tree, subset
        def __set_numerical__(self, node_label, subset):
            # TODO make verification of values more smart!
            # TODO verify only values where the class of adjacent objects changes!

            unique_vals = subset[node_label].sort_values().unique()

            t_counter = 1
            max_t = unique_vals.shape[0]

            best_entropy = np.inf

            if unique_vals.shape[0] < 3:
                meta = self.__set_terminal__(subset, Individual.target_attr)
                return meta, pd.DataFrame([]), pd.DataFrame([])

            best_threshold = None
            best_subset_left = None
            best_subset_right = None

            while t_counter < max_t - 1:  # should not pick limitrophe values, since it generates empty sets
                threshold = unique_vals[t_counter]

                subset_left = subset.loc[subset[node_label] < threshold]
                subset_right = subset.loc[subset[node_label] >= threshold]

                entropy = \
                    Individual.entropy(subset_left, Individual.target_attr) + \
                    Individual.entropy(subset_right, Individual.target_attr)

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_threshold = threshold
                    best_subset_left = subset_left
                    best_subset_right = subset_right

                t_counter += 1

            meta = {'value': best_threshold, 'terminal': False}
            return meta, best_subset_left, best_subset_right

        @staticmethod
        def __set_terminal__(subset, target_attr):
            count = Counter(subset[target_attr])

            f_key = None
            f_val = -np.inf
            for key, val in count.iteritems():
                if val > f_val:
                    f_key = key
                    f_val = val

            meta = {'value': f_key, 'terminal': True}  # not a threshold, but a class label
            return meta

        @staticmethod
        def __set_error__(id_node, attr_name, dict_threshold, tree, subset):
            raise TypeError('Unsupported data type for column %s!' % attr_name)

        attr_handler_dict = {
            'object': __set_categorical__,
            'str': __set_categorical__,
            'int': __set_numerical__,
            'float': __set_numerical__,
            'bool': __set_categorical__,
            'complex': __set_error__
        }
