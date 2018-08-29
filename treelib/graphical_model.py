# coding=utf-8
import copy

from node import *
import pandas as pd
from collections import Counter
import operator as op

__author__ = 'Henry Cagnini'


class GraphicalModel(object):
    def __init__(self, D, dataset_info):
        self.D = D
        self.dataset_info = dataset_info

        self.attributes = self.__init_attributes__(D)

    def __init_attributes__(self, D):
        def set_probability(column):
            n_attributes = column.index.shape[0]  # class attribute is the last one

            d = get_depth(column.name)

            class_prob = 0.  # zero probability
            # class_prob = (1. / (D + 1)) * float(d)  # linear progression
            pred_prob = (1. - class_prob) / (n_attributes - 1.)

            column[-1] = class_prob
            column[:-1] = pred_prob

            rest = abs(column.values.sum() - 1.)
            column[np.random.randint(0, n_attributes)] += rest

            return column

        n_variables = get_total_nodes(D - 1)  # since the probability of generating the class at D is 100%

        attributes = pd.DataFrame(
            index=np.hstack((self.dataset_info.pred_attr, [self.dataset_info.target_attr])), columns=range(n_variables)
        ).apply(set_probability, axis=0)

        return attributes
    
    def update(self, fittest):
        def get_label(_fit, _node_id):
            if _node_id not in _fit.tree.node:
                return None

            label = _fit.tree.node[_node_id]['label'] \
                if _fit.tree.node[_node_id]['label'] not in self.dataset_info.class_labels \
                else self.dataset_info.target_attr
            return label

        def __concatenate__(labels):
            all = []
            for _set in labels:
                if isinstance(_set, list):
                    all += _set
                else:
                    all += [_set]
            return all

        def local_update(column):
            labels = __concatenate__([get_label(fit, column.name) for fit in fittest])
            n_unsampled = labels.count(None)
            labels = [x for x in labels if x is not None]  # removes none from unsampled

            graft = [np.random.choice(column.index) for x in xrange(n_unsampled)]

            label_count = Counter(labels)
            graft_count = Counter(graft)

            label_count.update(graft_count)

            column[:] = 0.
            for k, v in label_count.iteritems():
                column[column.index == k] = v

            column /= float(column.sum())
            rest = abs(column.sum() - 1.)
            column[np.random.choice(column.index)] += rest

            return column

        self.attributes = self.attributes.apply(local_update, axis=0)

    def observe(self, node_id, evidence=None):
        """
        Makes observations about a given variable.

        :param node_id: ID of the node (i.e. variable) being observed.
        :param evidence: optional - evidence used for observing the variable. May be None if the variable is independent.
        :return: Observation of the variable, which is a set of values sampled from the variable's distribution.
        """
        node_labels = []
        variable = self.attributes[node_id]
        # variable = self.attributes[node_id]

        label = np.random.choice(a=variable.index, p=variable)
        node_labels += [label]
        # variable.loc[label] = 0
        # variable = variable / variable.sum()
        # rest = abs(variable.sum() - 1.)
        # variable.loc[np.random.choice(variable.index)] += rest

        return np.array(node_labels)
