# coding=utf-8
from collections import Counter

import numpy as np
import pandas as pd

from treelib.variable import Variable
from node import *
import warnings

__author__ = 'Henry Cagnini'


class GraphicalModel(object):
    """
        A graphical model is a tree itself.
    """
    
    attributes = None  # tensor is a dependency graph
    
    def __init__(
            self, pred_attr, target_attr, class_labels, max_depth=3, distribution='multivariate'
    ):

        self.class_labels = class_labels
        self.pred_attr = pred_attr
        self.target_attr = target_attr
        self.distribution = distribution
        self.max_depth = max_depth

        self.attributes = self.__init_attributes__(max_depth, distribution)

    @classmethod
    def clean(cls):
        cls.pred_attr = None
        cls.target_attr = None
        cls.class_labels = None

    def __init_attributes__(self, max_depth, distribution='univariate'):
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

            depth = get_depth(column.name)

            # class_prob = (1. / (max_depth + 1)) * float(depth)  # linear progression
            class_prob = 2. ** depth / 2. ** (max_depth + 1)  # power of 2 progression
            pred_prob = (1. - class_prob) / (n_attributes - 1.)

            column[-1] = class_prob
            column[:-1] = pred_prob

            rest = abs(column.values.sum() - 1.)
            column[np.random.randint(0, n_attributes)] += rest

            return column

        n_variables = get_total_nodes(max_depth)

        warnings.warn('WARNING: using a single pandas.DataFrame for attributes!')

        attributes = pd.DataFrame(
            index=self.pred_attr + [self.target_attr], columns=range(n_variables)
        ).apply(set_probability, axis=0)

        # sample_values = self.pred_attr + [self.target_attr]
        #
        # # TODO reduce!
        #
        # warnings.warn('WARNING: not using a linear progression!')
        # class_prob = 2. ** depth / 2. ** (max_depth + 1)  # power of 2 progression

        # attributes = map(
        #     lambda i: Variable(
        #         name=i,
        #         values=sample_values,
        #         parents=get_parents(i, distribution),
        #         max_depth=max_depth,
        #         target_attr=self.target_attr  # kwargs
        #     ),
        #     xrange(n_variables)
        # )
        
        return attributes
    
    def update(self, fittest):
        """
        Update attributes' probabilities.

        :type fittest: pandas.Series
        :param fittest:
        :return:
        """

        def get_label(_fit, _node_id):
            if _node_id not in _fit.tree.node:
                return None

            label = _fit.tree.node[_node_id]['label'] \
                if _fit.tree.node[_node_id]['label'] not in self.class_labels \
                else self.target_attr
            return label

        if self.distribution == 'univariate':
            for attr in self.attributes:

                weights = attr.weights

                labels = [get_label(fit, attr.name) for fit in fittest]
                labels = [x for x in labels if x is not None]
                # TODO remove all nones!

                if len(labels) > 0:
                    weights['probability'] = 0.

                    count = Counter(labels)
                    for k, v in count.iteritems():
                        weights.loc[weights[attr.name] == k, 'probability'] = v

                    weights['probability'] /= float(weights['probability'].sum())
                    rest = abs(weights['probability'].sum() - 1.)
                    weights.loc[np.random.choice(weights.index), 'probability'] += rest

                    attr.weights = weights

        elif self.distribution == 'multivariate':
            raise NotImplementedError('not implemented yet!')
        elif self.distribution == 'bivariate':
            raise NotImplementedError('not implemented yet!')

        # for attr in self.attributes:
        #     if attr.name == 29:
        #         print attr.name, attr.weights.values.ravel()

    def sample(self, node_id, level, parent_labels=None, enforce_nonterminal=False):
        value = np.random.choice(a=self.attributes[node_id].index, p=self.attributes[node_id])
        if enforce_nonterminal:
            while value == self.target_attr:
                value = np.random.choice(a=self.attributes[node_id].index, p=self.attributes[node_id])

        return value
