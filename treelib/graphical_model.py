# coding=utf-8
from collections import Counter

import numpy as np
import pandas as pd

from treelib.classes import AbstractTree
from treelib.variable import Variable

__author__ = 'Henry Cagnini'


class GraphicalModel(AbstractTree):
    """
        A graphical model is a tree itself.
    """
    
    variables = None  # tensor is a dependency graph
    
    def __init__(
            self, max_height=3, distribution='multivariate', **kwargs
    ):
        super(GraphicalModel, self).__init__(**kwargs)

        self.distribution = distribution
        self.max_height = max_height

        self.variables = self.__init_variables__(max_height, distribution)
    
    def __init_variables__(self, max_height, distribution='univariate'):
        def get_parents(_id, _distribution):
            if _distribution == 'multivariate':
                parents = range(_id) if _id > 0 else []
            elif _distribution == 'bivariate':
                parents = [_id - 1] if _id > 0 else []
            elif _distribution == 'univariate':
                parents = []
            else:
                raise ValueError('Distribution must be either multivariate, bivariate or univariate!')
            return parents

        sample_values = self.pred_attr + [self.target_attr]

        variables = map(
            lambda i: Variable(
                name=i,
                values=sample_values,
                parents=get_parents(i, distribution),
                max_height=max_height,
                target_attr=self.target_attr  # kwargs
            ),
            xrange(max_height)
        )
        
        return variables
    
    def update(self, fittest):
        """
        Updates graphical model.

        :type fittest: pandas.Series
        :param fittest:
        :return:
        """

        pd.options.mode.chained_assignment = None  # why not throw some stuff into the fan?

        if self.distribution == 'univariate':
            for height in xrange(self.max_height):
                at_height = []
                for fit in fittest:
                    x = fit.nodes_at_depth(height)
                    if len(x) > 0:
                        at_height.extend(x)

                weights = self.variables[height].weights
                weights['probability'] = 0.

                count = Counter(x['label'] if x['label'] in (self.pred_attr + [self.target_attr]) else self.target_attr for x in at_height)

                for k, v in count.iteritems():
                    weights.loc[weights[height] == k, 'probability'] = v

                weights['probability'] /= float(sum(count.values()))
                self.variables[height].weights[height] = weights

        elif self.distribution == 'multivariate':
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

        pd.options.mode.chained_assignment = 'warn'  # ok, I promise I'll behave
    
    def sample(self, level, parent_labels=None, enforce_nonterminal=False):
        value = self.variables[level].get_value(parent_labels=parent_labels)
        if enforce_nonterminal:
            while value == self.target_attr:
                value = self.variables[level].get_value(parent_labels=parent_labels)

        return value
