# coding = utf-8

import numpy as np
import pandas as pd
import itertools as it

from treelib.classes import SetterClass
from treelib.node import *
import warnings

__author__ = 'Henry Cagnini'


class Variable(SetterClass):
    target_attr = None

    # i,
    # parents = get_parents(i, distribution),
    # values = sample_values,
    # max_height = max_height,
    # target_attr = self.target_attr

    def __init__(self, name, values, parents=None, max_depth=None, **kwargs):
        super(Variable, self).__init__(**kwargs)

        self.name = name  # type: int
        self.values = values  # type: list of str
        self.parents = parents if (len(parents) > 0 or parents is not None) else []  # type: list of int

        self.weights = self.__init_probabilities__(max_depth)  # type: pd.DataFrame

    @property
    def n_parents(self):
        return len(self.parents)

    @property
    def n_values(self):
        return len(self.values)

    def __init_probabilities__(self, max_depth=None):

        vec_vals = [self.values] + [self.values for p in self.parents]

        combs = list(it.product(*vec_vals))
        columns = [self.name] + [p for p in self.parents]

        df = pd.DataFrame(
            data=combs,
            columns=columns
        )

        # raise NotImplementedError('name of variable is not its height!')

        _slice = df.loc[df[self.name] == self.target_attr]
        depth = get_depth(self.name)
        class_prob = (1. / (max_depth + 1)) * float(depth)
        slice_prob = class_prob / _slice.shape[0]

        pred_prob = (1. - class_prob) / (df.shape[0] - _slice.shape[0])

        df['probability'] = pred_prob
        df.loc[_slice.index, 'probability'] = slice_prob

        rest = abs(df['probability'].sum() - 1.)
        df.loc[np.random.randint(0, df.shape[0]), 'probability'] += rest
        return df

    def get_value(self, parent_labels=None):
        if len(self.parents) > 0:
            weights = self.weights.copy()
            # parent_labels are sorted, from the most distance
            for parent, label in it.izip(self.parents, parent_labels):
                weights = weights.loc[weights[parent] == label]

            weights['probability'] /= weights['probability'].sum()
            rest = abs(weights['probability'].sum() - 1.)
            weights.loc[np.random.choice(weights.index), 'probability'] += rest
        else:
            weights = self.weights

        a, p = (weights[self.name], weights['probability'])

        warnings.filterwarnings('error')

        try:
            value = np.random.choice(a=a, p=p)
        except RuntimeWarning:
            print 'values:', a
            print 'probs:', p
            value = np.random.choice(a=a, p=p)
        # ----------------------- #

        warnings.filterwarnings('warn')

        return value
