# coding=utf-8
import json

import numpy as np
from collections import Counter
import networkx as nx
import pandas as pd
import itertools as it

from heap import Node
from treelib.graphical_models import AbstractTree

from treelib.utils import SetterClass

__author__ = 'Henry Cagnini'


class Individual(AbstractTree):
    # target_attr = None  # type: str
    # target_labels = None  # type: list
    column_types = None  # type: dict
    sets = None  # type: dict
    tree = None
    val_acc = None
    
    def __init__(self, graphical_model, sets, **kwargs):
        """
        
        :type graphical_model: treelib.graphical_models.GraphicalModel
        :param graphical_model:
        :type sets: dict
        :param sets:
        :type kwargs: dict
        :param kwargs:
        """
        super(Individual, self).__init__(**kwargs)

        if Individual.column_types is None:
            Individual.column_types = {
                x: self.type_handler_dict[str(sets['train'][x].dtype)] for x in sets['train'].columns
            }  # type: dict
            Individual.column_types['class'] = 'class'
        self.column_types = Individual.column_types
        
        self.sets = sets
        self.sample(graphical_model, sets)

    def __str__(self):
        return 'fitness: %0.2f' % self._val_acc

    def plot(self):
        """
        Plots this individual.
        """
    
        raise NotImplementedError('not implemented yet!')
    
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

    @property
    def fitness(self):
        """
        :rtype: float
        :return: Fitness of this individual.
        """
        return self._val_acc

    # ############################ #
    # sampling and related methods #
    # ############################ #

    def sample(self, graphical_model, sets):
        sess = graphical_model.sample()

        self.tree = self.__set_thresholds__(sess, sets['train'])
        self.val_acc = self.__validate__(self.sets['val'])

    def __set_thresholds__(self, sess, train_set):
        tree = nx.DiGraph()
    
        subset = train_set
    
        tree = self.__set_node_threshold__(
            sess=sess,
            tree=tree,
            subset=subset,
            variable_name=0,
            parent_name=None
        )
        return tree

    def __set_node_threshold__(self, sess, tree, subset, variable_name, parent_name):
        # ############################ #
        # definition of inline methods #
        # ############################ #
        
        def node_attr(subset, target_attr):
            meta = self.__set_terminal__(target_attr, subset)
        
            attr_dict = {
                'label': meta['value'],
                'color': '#98FB98',
                'terminal': True,
                'threshold': None,
                'left': None,
                'right': None
            }
            return attr_dict
        
        def twin_nodes(tree, id_left, id_right):
            """
            Checks if two children nodes have the same class.
            :type tree: networkx.DiGraph
            :param tree: A DiGraph object representing a binary tree.
            :param id_left: ID of the left child.
            :param id_right: ID of the right child.
            :return: Whether right and left children are twins.
            """
            return tree.node[id_left]['label'] == tree.node[id_right]['label'] and \
                   tree.node[id_left]['label'] in Individual.target_labels
    
        # ##################### #
        # end of inline methods #
        # ##################### #
    
        if subset.shape[0] <= 0:
            raise ValueError('empty subset!')  # TODO see what must be done!
    
        try:
            meta, subset_left, subset_right = self.__set_node__(
                variable_name=variable_name,
                subset=subset,
                sess=sess
            )
            id_left, id_right = (Node.get_left_child(variable_name), Node.get_right_child(variable_name))
        
            try:  # if one of the subsets is empty, then the node is terminal
                for (id_child, child_subset) in it.izip([id_left, id_right], [subset_left, subset_right]):
                    tree = self.__set_node_threshold__(
                        sess=sess,
                        tree=tree,
                        subset=child_subset,
                        variable_name=id_child,
                        parent_name=variable_name
                    )
            
                if twin_nodes(tree, id_left, id_right):
                    tree.remove_node(id_left)
                    tree.remove_node(id_right)
                
                    _node_attr = node_attr(subset, Individual.target_attr)
                else:
                    tree.add_edge(variable_name, id_left, attr_dict={'threshold': '< %0.2f' % meta['value']})
                    tree.add_edge(variable_name, id_right, attr_dict={'threshold': '>= %0.2f' % meta['value']})
                
                    _node_attr = {
                        'label': variable_name,
                        'color': '#FFFFFF' if variable_name == Node.root else '#AEEAFF',
                        'terminal': False,
                        'threshold': meta['value'],
                        'left': id_left,
                        'right': id_right
                    }
            except ValueError as ve:
                _node_attr = node_attr(subset, Individual.target_attr)
        except ValueError as ve:
            _node_attr = node_attr(subset, Individual.target_attr)
    
        tree.add_node(variable_name, attr_dict=_node_attr)
        return tree
        
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

        node = self.tree.node[arg_node]
        while not node['terminal']:
            go_left = obj[node['label']] < node['threshold']
            arg_node = (int(go_left) * node['left']) + (int(not go_left) * node['right'])
            node = self.tree.node[arg_node]

        return obj[-1] == node['label']

    def __validate__(self, test_set):
        """
        Assess the accuracy of this Individual against the provided set.
        
        :type test_set: pandas.DataFrame
        :param test_set: a matrix with the class attribute in the last position (i.e, column).
        :return: The accuracy of this model when testing with test_set.
        """
        
        hit_count = test_set.apply(self.__val_func__, axis=1).sum()
        acc = hit_count / float(test_set.shape[0])
        return acc

    def __set_node__(self, variable_name, subset, sess, **kwargs):
        
        def set_terminal():
            meta = self.__set_terminal__(Individual.target_attr, subset, **kwargs)
            subset_left = pd.DataFrame([0.])
            subset_right = pd.DataFrame([0.])
            
            return meta, subset_left, subset_right
        
        if variable_name != Individual.target_attr:
            attr_type = Individual.column_types[sess[variable_name]]
            try:
                out = self.attr_handler_dict[attr_type](self, sess[variable_name], subset, **kwargs)
            except ValueError as ve:
                out = set_terminal()
        else:
            out = set_terminal()
        return out

    def __set_numerical__(self, node_label, subset, **kwargs):
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

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return best_subset_left, best_subset_right
        else:
            meta = {'value': best_threshold, 'terminal': False}
            return meta, best_subset_left, best_subset_right
    
    def __set_terminal__(self, node_label, subset, **kwargs):
        count = Counter(subset[node_label])

        f_key = None
        f_val = -np.inf
        for key, val in count.iteritems():
            if val > f_val:
                f_key = key
                f_val = val

        meta = {'value': f_key, 'terminal': True}  # not a threshold, but a class label
        return meta

    def __set_categorical__(self, node_label, subset):
        raise NotImplemented('not implemented yet!')

    @staticmethod
    def __set_error__(id_node, attr_name, dict_threshold, tree, subset):
        raise TypeError('Unsupported data type for column %s!' % attr_name)

    attr_handler_dict = {
        'object': __set_categorical__,
        'str': __set_categorical__,
        'int': __set_numerical__,
        'float': __set_numerical__,
        'bool': __set_categorical__,
        'complex': __set_error__,
        'class': __set_terminal__
    }

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
