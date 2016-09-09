# coding=utf-8

import numpy as np
from collections import Counter
import networkx as nx
import pandas as pd
import itertools as it

from heap import Node

__author__ = 'Henry Cagnini'


class Individual(object):
    target_attr = None
    target_values = None
    column_types = None

    def __init__(self, initial_pmf, sets, **kwargs):
        """
        
        :param id: ID of the individual. For debugging purposes.
        :param initial_pmf:
        :param sets:
        """

        if 'id' in kwargs:
            self._id = kwargs['id']

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
        self._height = self.__get_height__()

    def sample(self, pmf):
        self._tree = self._sampler.sample(pmf)
        self._val_acc = self.__validate__(self._sets['val'])

    def __get_height__(self):
        node_dict = self._tree.node
        last = Node.get_depth(max(node_dict.keys()))
        return last

    @property
    def height(self):
        return self._height

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
            0.8,
            0.9,
            'Fitness: %0.4f' % self._val_acc,
            fontsize=15,
            horizontalalignment='center',
            verticalalignment='center',
            transform=fig.transFigure
        )

        plt.text(
            0.1,
            0.1,
            'ID: %03.d' % self._id,
            fontsize=15,
            horizontalalignment='center',
            verticalalignment='center',
            transform=fig.transFigure
        )

        plt.axis('off')

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

        def sample(self, gm):
            tree = nx.DiGraph()

            subset = self._sets['train']

            tree = self.sample_node(
                gm=gm,
                tree=tree,
                subset=subset,
                id_current=0,
                parent_label=None
            )

            return tree

        def sample_node(self, gm, tree, subset, id_current, parent_label):
            def node_attr(subset, target_attr):
                meta = self.__set_terminal__(subset, target_attr)
        
                attr_dict = {
                    'label': meta['value'],
                    'color': '#98FB98',
                    'terminal': True,
                    'threshold': None,
                    'left': None,
                    'right': None
                }
                return attr_dict
            
            if subset.shape[0] <= 0:
                raise ValueError('empty subset!')
            
            node_label = gm.sample(id_node=id_current, parent_label=parent_label)
            if id_current == Node.root:  # enforces sampling of non-terminal attribute
                while node_label == Individual.target_attr:
                    node_label = gm.sample(id_node=id_current, parent_label=parent_label)

            if node_label != Individual.target_attr:
                try:
                    meta, subset_left, subset_right = self.__set_internal__(
                        node_label=node_label,
                        subset=subset
                    )

                    id_left, id_right = (Node.get_left_child(id_current), Node.get_right_child(id_current))

                    try:  # if one of the subsets is empty, then the node is terminal
                        for (id_child, child_subset) in it.izip([id_left, id_right], [subset_left, subset_right]):
                            tree = self.sample_node(
                                gm=gm,
                                tree=tree,
                                subset=child_subset,
                                id_current=id_child,
                                parent_label=node_label
                            )
    
                        if tree.node[id_left]['label'] == tree.node[id_right]['label'] and tree.node[id_left][
                            'label'] in Individual.target_values:  # both children have the same class label
                            tree.remove_node(id_left)
                            tree.remove_node(id_right)
        
                            _node_attr = node_attr(subset, Individual.target_attr)
                        else:
                            tree.add_edge(id_current, id_left, attr_dict={'threshold': '< %0.2f' % meta['value']})
                            tree.add_edge(id_current, id_right, attr_dict={'threshold': '>= %0.2f' % meta['value']})
        
                            _node_attr = {
                                'label': node_label,
                                'color': '#FFFFFF' if id_current == Node.root else '#AEEAFF',
                                'terminal': False,
                                'threshold': meta['value'],
                                'left': id_left,
                                'right': id_right
                            }
                    except ValueError as ve:
                        _node_attr = node_attr(subset, Individual.target_attr)
                except ValueError as ve:
                    _node_attr = node_attr(subset, Individual.target_attr)
            else:
                _node_attr = node_attr(subset, Individual.target_attr)

            tree.add_node(id_current, attr_dict=_node_attr)
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
            def slide_filter(x):
                first = ((x.name - 1) * (x.name > 0)) + (x.name * (x.name <= 0))
                second = x.name
                column = Individual.target_attr
                
                return ss[column].iloc[first] == ss[column].iloc[second]

            def get_entropy(threshold):
                
                subset_left = subset.loc[subset[node_label] < threshold]
                subset_right = subset.loc[subset[node_label] >= threshold]
                
                entropy = \
                    Individual.entropy(subset_left, Individual.target_attr) + \
                    Individual.entropy(subset_right, Individual.target_attr)
    
                return entropy

            ss = subset[[node_label, Individual.target_attr]]  # type: pd.DataFrame
            ss = ss.sort_values(by=node_label).reset_index()
            
            ss['change'] = ss.apply(slide_filter, axis=1)
            unique_vals = ss[ss['change'] == False]
            
            if unique_vals.empty:
                raise ValueError('no valid threshold values!')
            
            unique_vals['entropy'] = unique_vals[node_label].apply(get_entropy)
            best_entropy = unique_vals['entropy'].min()
            
            best_threshold = (unique_vals[unique_vals['entropy'] == best_entropy])[node_label].values[0]
            
            best_subset_left = subset.loc[subset[node_label] < best_threshold]
            best_subset_right = subset.loc[subset[node_label] >= best_threshold]

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
