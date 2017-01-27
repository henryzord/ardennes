# coding=utf-8
from collections import Counter

import numpy as np
import pandas as pd

from treelib.variable import Variable
from node import *
import warnings
import operator as op
import itertools as it

__author__ = 'Henry Cagnini'


class GraphicalModel(object):
    """
        A graphical model is a tree itself.
    """
    
    attributes = None  # tensor is a dependency graph
    
    def __init__(
            self, pred_attr, target_attr, class_labels, D, distribution='multivariate'
    ):

        self.class_labels = class_labels
        self.pred_attr = pred_attr
        self.target_attr = target_attr
        self.distribution = distribution
        self.D = D

        self.attributes = self.__init_attributes__(D, distribution)

    @classmethod
    def clean(cls):
        cls.pred_attr = None
        cls.target_attr = None
        cls.class_labels = None

    def __init_attributes__(self, D, distribution='univariate'):
        def get_parents(_id, _distribution):
            if _distribution == 'multivariate':
                raise NotImplementedError('not implemented yet!')
                # parents = range(_id) if _id > 0 else []
            elif _distribution == 'bivariate':
                raise NotImplementedError('not implemented yet!')
                # parents = [_id - 1] if _id > 0 else []
            elif _distribution == 'univariate':
                parents = []
            else:
                raise ValueError('Distribution must be either \'multivariate\', \'bivariate\' or \'univariate\'!')
            return parents

        def set_probability(column):
            # class attribute is the last one
            n_attributes = column.index.shape[0]

            d = get_depth(column.name)

            class_prob = (1. / (D + 1)) * float(d)  # linear progression
            # class_prob = 2. ** d / 2. ** (D + 1)  # power of 2 progression
            pred_prob = (1. - class_prob) / (n_attributes - 1.)

            column[-1] = class_prob
            column[:-1] = pred_prob

            rest = abs(column.values.sum() - 1.)
            column[np.random.randint(0, n_attributes)] += rest

            return column

        n_variables = get_total_nodes(D - 1)  # since the probability of generating the class at D is 100%

        attributes = pd.DataFrame(
            index=self.pred_attr + [self.target_attr], columns=range(n_variables)
        ).apply(set_probability, axis=0)

        return attributes
    
    def update(self, fittest):
        """
        Update attributes' probabilities.

        :type fittest: pandas.Series
        :param fittest:
        :return:
        """
        # TODO update using trace to each one of the instances!

        def get_label(_fit, _node_id):
            if _node_id not in _fit.tree.node:
                return None

            label = _fit.tree.node[_node_id]['label'] \
                if _fit.tree.node[_node_id]['label'] not in self.class_labels \
                else self.target_attr
            return label

        if self.distribution == 'univariate':

            def local_update(column):
                labels = [get_label(fit, column.name) for fit in fittest]
                n_unsampled = labels.count(None)
                labels = [x for x in labels if x is not None]  # removes none from unsampled
                label_count = Counter(labels)

                ommited = False
                for attribute in column.index:
                    if attribute not in label_count:
                        ommited = True
                        label_count.update({attribute: 0})

                if ommited is True:
                    label_count = {k: v + n_unsampled for k, v in label_count.iteritems()}

                column[:] = 0.
                for k, v in label_count.iteritems():
                    column[column.index == k] = v

                column /= float(column.sum())
                rest = abs(column.sum() - 1.)
                column[np.random.choice(column.index)] += rest

                return column

            self.attributes = self.attributes.apply(local_update, axis=0)

        elif self.distribution == 'multivariate':
            raise NotImplementedError('not implemented yet!')
        elif self.distribution == 'bivariate':
            raise NotImplementedError('not implemented yet!')

    # def update(self, fittest):
    #     # updates by instance
    #
    #     """
    #     Update attributes' probabilities.
    #
    #     :type fittest: pandas.Series
    #     :param fittest:
    #     :return:
    #     """
    #
    #     def __reset__(attr):
    #         attr[:] = 0.
    #         return attr
    #
    #     def __normalize__(attr):
    #         _sum = attr[:].sum()
    #         if _sum > 0:
    #             attr[:] /= _sum
    #         else:
    #             # uniform distribution
    #             attr[:-1] = 1. / attr.shape[0]
    #
    #         rest = abs(1. - attr[:].sum())
    #         attr[np.random.choice(attr.index)] += rest
    #
    #         return attr
    #
    #     def get_label(_fit, _node_id, _total_instances):
    #         rate_correct = _fit.tree.node[_node_id]['inst_correct']
    #         return rate_correct / float(_total_instances)
    #
    #     if self.distribution == 'univariate':
    #         self.attributes = self.attributes.apply(__reset__, axis=0)
    #
    #         _temp2 = [x for x in fittest[0].tree.nodes_iter() if fittest[0].tree.out_degree(x) == 0]
    #         total_instances = reduce(op.add, [fittest[0].tree.node[x]['inst_total'] for x in _temp2])
    #
    #         for fit in fittest:
    #             # gets only leafs
    #             _temp = [x for x in fit.tree.nodes_iter() if fit.tree.out_degree(x) == 0]
    #             # node_id, accuracy of the leaf, path to the node
    #             paths = [fit.parents_of(x) for x in _temp]
    #             path_labels = [[fit.tree.node[x]['label'] for x in path] for path in paths]
    #             leafs = list(it.izip(_temp, [get_label(fit, x, total_instances) for x in _temp], paths, path_labels))
    #
    #             for node_id, acc, path, path_label in leafs:
    #                 _len_path = len(path)
    #                 for mid_id, mid_label in it.izip(path, path_label):
    #                     self.attributes.loc[mid_label, mid_id] += acc / float(_len_path)
    #
    #         self.attributes.apply(__normalize__, axis=0)
    #
    #     elif self.distribution == 'multivariate':
    #         raise NotImplementedError('not implemented yet!')
    #     elif self.distribution == 'bivariate':
    #         raise NotImplementedError('not implemented yet!')

    def sample(self, node_id, level, parent_labels=None, enforce_nonterminal=False):
        warnings.filterwarnings('error')

        value = np.random.choice(a=self.attributes[node_id].index, p=self.attributes[node_id])
        if enforce_nonterminal:
            while value == self.target_attr:
                value = np.random.choice(a=self.attributes[node_id].index, p=self.attributes[node_id])

        return value
