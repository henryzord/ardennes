# coding=utf-8

from collections import Counter
import numpy as np
from solution import Individual

from solution.trees import DecisionTree

__author__ = 'Henry Cagnini'


class GraphicalModel(object):
    def __init__(self, max_depth, full_df):
        self.max_depth = max_depth

        self.a = full_df.columns

        # since the probability of generating the class at D is 100%
        self.n_variables = DecisionTree.get_node_count(self.max_depth - 1)

        self.p = np.empty(
            (self.n_variables, len(self.a)), dtype=np.float32
        )
        self.p[:, :-1] = 1. / (len(self.a) - 1)  # excludes class
        self.p[:, -1] = 0.

    def update(self, population, elite_threshold):
        """
        Update graphical model based on the elite population.

        :param population: whole population of individuals from a generation.
        :dtype population: pandas.DataFrame
        :param elite_threshold: Maximum number of individuals to belong to the next elite population.
        :type elite_threshold: int
        :return: updated graphical model.
        :rtype: GraphicalModel
        """

        elite = population.loc[population['A']]

        for i in range(self.n_variables):
            counts = Counter([
                individual['P'].nodes[i] if individual['P'].nodes[i] not in individual['P'].attr_values[individual['P'].class_attr_name] else individual['P'].class_attr_name for _, individual in elite.iterrows()])
            vacancy = max(0, elite_threshold - sum(counts.values()))
            counts.update(Counter(np.random.choice(a=self.a, size=vacancy, replace=True)))
            prob_sum = 0.
            for j, val in enumerate(self.a):
                freq = float(counts[val]) / float(elite_threshold)
                prob_sum += freq
                self.p[i, j] = freq
            self.p[i, np.random.choice(len(self.a))] += 1. - prob_sum

        return self

    def sample(self, population, elite_threshold):
        """
        Samples non-elite individuals. Also sorts population in a lexicographic fashion (using first fitness, then tree
        height, and finally the number of nodes), and asserts which individuals will be the next elite population.

        :param population: whole population of individuals from a generation.
        :dtype population: pandas.DataFrame
        :param elite_threshold: Maximum number of individuals to belong to the next elite population.
        :type elite_threshold: int
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
                population.loc[i, ['P', 'P_fitness', 'P_quality', 'A']] = [
                    individual, individual.fitness, individual.quality, False
                ]
            else:
                population.loc[i, 'A'] = False

        population.sort_values(by='P_fitness', axis=0, inplace=True, ascending=False)
        population.loc[population.index[:elite_threshold], 'A'] = True
        return population
