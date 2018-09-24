# coding=utf-8

from collections import Counter
import numpy as np
from solution import Individual

from solution.trees import DecisionTree

__author__ = 'Henry Cagnini'


class GraphicalModel(object):
    def __init__(self, max_depth, full_df):
        self.max_depth = max_depth

        pred_attr_names = np.array(full_df.columns[:-1])  # type: np.ndarray
        class_attr_name = full_df.columns[-1]  # type: str
        class_labels = np.sort(full_df[full_df.columns[-1]].unique())  # type: np.ndarray

        self.a = np.hstack((pred_attr_names, class_attr_name))
        self.class_labels = class_labels

        self.n_variables = DecisionTree.get_node_count(max_depth - 1)  # since the probability of generating the class at D is 100%
        self.p = np.empty(
            (self.n_variables, len(self.a)), dtype=np.float32
        )
        self.p[:, :-1] = 1. / (len(self.a) - 1)  # excludes class
        self.p[:, -1] = 0.

    def update(self, population):
        raise NotImplementedError('not implemented yet!')


        def get_label(_ind, _node_id):
            A, P, P_fitness = _ind.loc['A'], _ind.at['P'], _ind.at['P_fitness']

            if _node_id in P.tree.node:
                label = P.tree.node[_node_id]['label'] \
                    if P.tree.node[_node_id]['label'] not in self.dataset_info.class_labels \
                    else self.dataset_info.target_attr
                return label
            else:
                return None

        def local_update(attribute):
            labels = [get_label(ind, attribute.name) for (i, ind) in (population.loc[population.A]).iterrows()]
            n_unsampled = labels.count(None)
            labels = [x for x in labels if x is not None]  # removes none from unsampled

            graft = [np.random.choice(attribute.index) for x in range(n_unsampled)]

            label_count = Counter(labels)
            graft_count = Counter(graft)

            label_count.update(graft_count)

            attribute[:] = 0.
            for k, v in label_count.items():
                attribute[attribute.index == k] = v

            attribute /= float(attribute.sum())
            rest = abs(attribute.sum() - 1.)
            attribute[np.random.choice(attribute.index)] += rest

            return attribute

        self.attributes = self.attributes.apply(local_update, axis=0)

    def sample(self, population):
        """
        Samples non-elite individuals.

        :param population: whole population of individuals from a generation.
        :dtype population: pandas.DataFrame
        :return: non-elite individuals are resampled, while elite individuals are kept.
        :rtype: pandas.DataFrame
        """

        for i, row in population.iterrows():
            if not row['A']:
                individual = row['P']  # type: Individual
                for j in range(self.n_variables):
                    individual.nodes[j] = np.random.choice(a=self.a, p=self.p[j])
                    individual.terminal[j] = False
                    individual.threshold[j] = np.nan
                individual.__update_after_sampling__()
                # individual.plot()  # TODO remove!
                # from matplotlib import pyplot as plt  # TODO remove!
                # plt.show()  # TODO remove!
