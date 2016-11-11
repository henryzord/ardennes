# coding = utf-8

import numpy as np
import pandas as pd
import itertools as it

from treelib.classes import SetterClass

__author__ = 'Henry Cagnini'


class Variable(SetterClass):
    target_attr = None

    # i,
    # parents = get_parents(i, distribution),
    # values = sample_values,
    # max_height = max_height,
    # target_attr = self.target_attr

    def __init__(self, name, values, parents=None, max_height=None, **kwargs):
        super(Variable, self).__init__(**kwargs)

        self.name = name  # type: int
        self.values = values  # type: list of str
        self.parents = parents if (len(parents) > 0 or parents is not None) else []  # type: list of int

        self.weights = self.__init_probabilities__(max_height)  # type: pd.DataFrame

    @property
    def n_parents(self):
        return len(self.parents)

    @property
    def n_values(self):
        return len(self.values)

    def __init_probabilities__(self, max_height=None):

        vec_vals = [self.values] + [self.values for p in self.parents]

        combs = list(it.product(*vec_vals))
        columns = [self.name] + [p for p in self.parents]

        df = pd.DataFrame(
            data=combs,
            columns=columns
        )

        _slice = df.loc[df[self.name] == self.target_attr]
        depth = self.name
        class_prob = (1. / (max_height - 1.)) * float(depth)
        slice_prob = class_prob / _slice.shape[0]

        pred_prob = (1. - class_prob) / (df.shape[0] - _slice.shape[0])

        df['probability'] = pred_prob
        df.loc[_slice.index, 'probability'] = slice_prob

        rest = max(0, 1. - df['probability'].sum())
        df.iloc[np.random.randint(0, df.shape[0])]['probability'] += rest
        return df

    def get_value(self, parent_labels=None):
        weights = self.weights

        if len(self.parents) > 0:
            # parent_labels are sorted, from the most distance
            for id_parent, label in enumerate(parent_labels):
                weights = weights.loc[weights[id_parent] == label]

            weights['probability'] /= weights.shape[0]

        a, p = (weights[self.name], weights['probability'])
        _sum = p.sum()

        value = np.random.choice(a=a, p=p)
        # ----------------------- #
        return value
