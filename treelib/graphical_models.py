# coding=utf-8
from collections import Counter

import numpy as np
import pandas as pd

import itertools as it
from treelib.classes import SetterClass, AbstractTree
# from treelib.node import *

__author__ = 'Henry Cagnini'


class GraphicalModel(AbstractTree):
    """
        A graphical model is a tree itself.
    """
    
    tensors = None  # tensor is a dependency graph
    
    def __init__(
            self, max_height=3, distribution='multivariate', class_probability='declining', **kwargs
    ):
        super(GraphicalModel, self).__init__(**kwargs)

        self.distribution = distribution
        self.class_probability = class_probability
        self.max_height = max_height

        self.tensors = self.__init_tensors__(max_height, distribution, class_probability)
    
    def __init_tensors__(self, max_height, distribution='uniform', class_probability='declining'):
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

        tensors = map(
            lambda i: Tensor(
                i,
                parents=get_parents(i, distribution),
                values=sample_values,
                class_probability=class_probability,
                max_height=max_height,
                target_attr=self.target_attr
            ),
            xrange(max_height)
        )
        
        return tensors
    
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

                weights = self.tensors[height].weights
                weights['probability'] = 0.

                count = Counter(x['label'] if x['label'] in (self.pred_attr + [self.target_attr]) else self.target_attr for x in at_height)

                for k, v in count.iteritems():
                    weights.loc[weights[height] == k, 'probability'] = v

                weights['probability'] /= float(sum(count.values()))
                self.tensors[height].weights[height] = weights

        elif self.distribution == 'multivariate':
            raise NotImplementedError('not implemented yet!')
        elif self.distribution == 'bivariate':
            raise NotImplementedError('not implemented yet!')

        pd.options.mode.chained_assignment = 'warn'  # ok, I promise I'll behave
    
    def sample(self, level, parent_labels=None, enforce_nonterminal=False):
        value = self.tensors[level].get_value(parent_labels=parent_labels)
        if enforce_nonterminal:
            while value == self.target_attr:
                value = self.tensors[level].get_value(parent_labels=parent_labels)

        return value


class Tensor(SetterClass):
    target_attr = None

    def __init__(
            self, name, values, parents=None, class_probability='declining',
            max_height=None, probability='uniform', **kwargs
    ):
        super(Tensor, self).__init__(**kwargs)

        self.name = name  # type: int
        self.values = values  # type: list of str
        self.parents = parents if (len(parents) > 0 or parents is not None) else []  # type: list of int

        self.weights = self.__init_probabilities__(
            values,
            probability,
            class_probability if len(self.values) > 1 else 'same',
            max_height=max_height
        )  # type: pd.DataFrame

    @property
    def n_parents(self):
        return len(self.parents)

    @property
    def n_values(self):
        return len(self.values)

    def __init_probabilities__(self, values, probability='uniform', class_probability='declining', max_height=None):
        if probability == 'uniform':
            vec_vals = [self.values] + [self.values for p in self.parents]

            combs = list(it.product(*vec_vals))
            columns = [self.name] + [p for p in self.parents]

            df = pd.DataFrame(
                data=combs,
                columns=columns
            )

            if class_probability == 'same':
                df['probability'] = 1. / df.shape[0]
            elif class_probability == 'decreased':
                _slice = df.loc[df[self.name] == self.target_attr]
                df['probability'] = 1. / (df.shape[0] - _slice.shape[0])
                df.loc[_slice.index, 'probability'] = 0.
            elif class_probability == 'declining':
                _slice = df.loc[df[self.name] == self.target_attr]
                if len(self.parents) > 0:
                    depth = get_depth(self.name)
                    class_prob = (1. / (max_height - 1.)) * float(depth)
                    slice_prob = class_prob / _slice.shape[0]

                    pred_prob = (1. - class_prob) / (df.shape[0] - _slice.shape[0])

                    df['probability'] = pred_prob
                    df.loc[_slice.index, 'probability'] = slice_prob
                elif len(values) > 1:
                    _slice = df.loc[df[self.name] == self.target_attr]
                    df['probability'] = 1. / (df.shape[0] - _slice.shape[0])
                    df.loc[_slice.index, 'probability'] = 0.
                else:
                    df['probability'] = 1. / df.shape[0]

            else:
                raise ValueError('class probability must be either \'same\', \'decreased\' or \'declining\'!')

            return df
        elif isinstance(probability, list):
            if len(probability) != len(values):
                raise IndexError('number of weights must be the same as the number of values!')
            else:
                raise NotImplementedError('not implemented yet!')
        else:
            raise TypeError('probability must be either a string or a list!')

    def get_value(self, parent_labels=None):

        if len(self.parents) == 0:
            a, p = (self.weights[self.name], self.weights['probability'])
            try:
                value = np.random.choice(a=a, p=p)
            except ValueError as ve:
                print p
                print p.sum()
                raise ve

        else:
            raise NotImplementedError('not implemented yet!')

            # print samples
            grouped = samples.groupby(by=self.parents, axis=0)
            groups = grouped.groups

            # iterates over values
            for group_name, group_index in groups.iteritems():
                if not isinstance(group_name, str):
                    try:
                        group_name = iter(group_name)
                    except TypeError, te:
                        group_name = [group_name]
                else:
                    group_name = [group_name]

                group_size = group_index.shape[0]

                # localize probabilities accordingly to parent values
                df = self.weights.copy()  # type: pd.DataFrame
                for i_p, p in it.izip(self.parents, group_name):
                    df = df.loc[df[i_p] == p]

                # update probabilities
                _sum = df['probability'].sum()
                df['probability'] = df['probability'].apply(lambda x: x / _sum)
                a, p = (df[self.name], df['probability'])

                repeat = (2 ** get_depth(self.name))  # TODO see if this is correct!
                size = group_size * repeat

                # proper sample
                value = np.random.choice(a=a, p=p, replace=True, size=size)
                _index = index_at_level(get_depth(self.name))
                samples.loc[group_index, _index] = value.reshape(group_size, repeat)

        # ----------------------- #
        return value