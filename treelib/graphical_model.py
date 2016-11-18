# coding=utf-8
from collections import Counter

import numpy as np
import pandas as pd

from treelib.classes import AbstractTree
from treelib.variable import Variable
from node import *

__author__ = 'Henry Cagnini'


class GraphicalModel(AbstractTree):
    """
        A graphical model is a tree itself.
    """
    
    variables = None  # tensor is a dependency graph
    
    def __init__(
            self, max_depth=3, distribution='multivariate', **kwargs
    ):
        super(GraphicalModel, self).__init__(**kwargs)

        self.distribution = distribution
        self.max_depth = max_depth

        self.variables = self.__init_variables__(max_depth, distribution)
    
    def __init_variables__(self, max_depth, distribution='univariate'):
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

        sample_values = self.pred_attr + [self.target_attr]

        n_variables = get_total_nodes(max_depth)

        variables = map(
            lambda i: Variable(
                name=i,
                values=sample_values,
                parents=get_parents(i, distribution),
                max_depth=max_depth,
                target_attr=self.target_attr  # kwargs
            ),
            xrange(n_variables)
        )
        
        return variables
    
    def update(self, fittest):
        """
        Updates graphical model.

        :type fittest: pandas.Series
        :param fittest:
        :return:
        """

        # pd.options.mode.chained_assignment = None  # why not throw some stuff into the fan?

        def get_label(_fit, _node_id):
            if _node_id not in _fit.tree.node:
                return None

            label = _fit.tree.node[_node_id]['label'] \
                if _fit.tree.node[_node_id]['label'] not in self.class_labels \
                else self.target_attr
            return label

        if self.distribution == 'univariate':
            for i, variable in enumerate(self.variables):
                weights = self.variables[i].weights

                labels = [get_label(fit, variable.name) for fit in fittest]

                weights['probability'] = 0.

                count = Counter(labels)
                for k, v in count.iteritems():
                    weights.loc[weights[variable.name] == k, 'probability'] = v

                weights['probability'] /= float(weights['probability'].sum())
                rest = abs(weights['probability'].sum() - 1.)
                weights.loc[np.random.choice(weights.index), 'probability'] += rest

                self.variables[i].weights = weights

        elif self.distribution == 'multivariate':
            raise NotImplementedError('not implemented yet!')

            for height in xrange(self.max_height):  # for each variable in the GM
                c_weights = self.variables[height].weights.copy()  # type: pd.DataFrame
                c_weights['probability'] = 0.

                for ind in fittest:  # for each individual in the fittest population
                    nodes_at_depth = ind.nodes_at_depth(height)
                    for node in nodes_at_depth:
                        parent_labels = ind.height_and_label_to(node['node_id'])

                        node_label = (node['label'] if not node['terminal'] else self.target_attr)
                        ind_weights = c_weights.loc[c_weights[node['level']] == node_label].index

                        if len(parent_labels) > 0:
                            str_ = '&'.join(['(c_weights[%d] == \'%s\')' % (p, l) for (p, l) in parent_labels.iteritems()])
                            p_ind_weights = c_weights[eval(str_)].index
                            ind_weights = set(p_ind_weights) & set(ind_weights)

                        c_weights.loc[ind_weights, 'probability'] += 1

                c_weights['probability'] /= float(c_weights['probability'].sum())
                rest = abs(c_weights['probability'].sum() - 1.)
                c_weights.loc[np.random.choice(c_weights.shape[0]), 'probability'] += rest
                self.variables[height].weights = c_weights
                # print c_weights
        elif self.distribution == 'bivariate':
            raise NotImplementedError('not implemented yet!')

        # pd.options.mode.chained_assignment = 'warn'  # ok, I promise I'll behave
    
    def sample(self, node_id, level, parent_labels=None, enforce_nonterminal=False):
        value = self.variables[node_id].get_value(parent_labels=parent_labels)
        if enforce_nonterminal:
            while value == self.target_attr:
                value = self.variables[node_id].get_value(parent_labels=parent_labels)

        return value
