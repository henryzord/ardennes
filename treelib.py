# coding=utf-8
import random
from datetime import datetime as dt

from graphical_model import *
from individual import Individual

from device import AvailableDevice
from individual import DecisionTree
from utils import MetaDataset
import numpy as np

__author__ = 'Henry Cagnini'


class Ardennes(object):
    def __init__(self, n_individuals, n_generations, max_height=3, decile=0.5, reporter=None):

        self.n_individuals = n_individuals

        self.D = max_height - 1
        self.n_generations = n_generations

        self.decile = decile

        self.trained = False
        self.predictor = None

        self.reporter = reporter

    def __setup__(self, full_df, train_index, val_index):
        dataset_info = MetaDataset(full_df)

        mdevice = AvailableDevice(full_df, dataset_info)

        arg_sets = {
            'train': train_index,
            'val': val_index
        }

        DecisionTree.set_values(
            arg_sets=arg_sets,
            y_train_true=full_df.loc[arg_sets['train'], dataset_info.target_attr],
            y_val_true=full_df.loc[arg_sets['val'], dataset_info.target_attr],
            processor=mdevice,
            dataset_info=dataset_info,
            max_height=self.D,
            dataset=full_df,
            mdevice=mdevice
        )

        gm = GraphicalModel(
            D=self.D,
            dataset_info=dataset_info
        )

        return gm

    def fit(self, full_df, train_index, val_index, verbose=True):
        """
        Fits the algorithm to the provided data.
        """

        # overrides prior seed
        np.random.seed(None)
        random.seed(None)

        assert 1 <= int(self.n_individuals * self.decile) <= self.n_individuals, \
            ValueError('Decile must comprise at least one individual and at maximum the whole population!')

        gm = self.__setup__(full_df=full_df, train_index=train_index, val_index=val_index)

        population = pd.DataFrame({
            'P': np.empty(shape=self.n_individuals, dtype=Individual),
            'P_fitness': np.empty(shape=self.n_individuals, dtype=np.float32),
            'A': np.zeros(self.n_individuals, dtype=np.bool),
        })

        # main loop
        g = 0

        while g < self.n_generations:
            t1 = dt.now()
            population = self.sample_population(gm, population)
            population = self.split_population(population, self.decile)

            gm.update(population)

            # TODO reactivate later!
            # self.__save__(reporter=self.reporter, generation=g, population=population, gm=gm)

            if self.__early_stop__(population):
                break

            median = float(np.median(population.P_fitness, axis=0))  # type: float

            if verbose:
                print('generation %02.d: best: %.4f median: %.4f time elapsed: %f' % (
                    g, population.P_fitness.max(), median, (dt.now() - t1).total_seconds()
                ))

            g += 1

        self.predictor = self.get_best_individual(population)
        self.trained = True
        return self

    @staticmethod
    def sample_population(gm, population):
        def __sample_func__(row):
            row['P'] = Individual(gm)
            row['P_fitness'] = row['P'].fitness
            return row

        population = population.apply(__sample_func__, axis=1)
        return population

    def split_population(self, population, decile):
        """

        :param population:
        :type population: pandas.DataFrame
        :param decile:
        :return:
        """

        integer_decile = int(self.n_individuals * decile)

        population.sort_values(by='P_fitness', axis=0, inplace=True, ascending=False)
        population.loc[population.index[:integer_decile], 'A'] = True

        return population

    @staticmethod
    def get_best_individual(population):
        outer_fitness = [0.5 * (ind.P.train_acc_score + ind.P.val_acc_score) for (i, ind) in population.iterrows()]
        return population.loc[np.argmax(outer_fitness)]

    @property
    def tree_height(self):
        if self.trained:
            return self.predictor.height

    @staticmethod
    def __save__(reporter, generation, population, gm):
        try:
            reporter.save_population(generation=generation, population=population)
            reporter.save_gm(generation=generation, gm=gm)
        except AttributeError:
            pass

    @staticmethod
    def __early_stop__(population):
        return population.P.min() == population.P.max()

    def predict(self, test_set):
        y_test_pred = list(self.predictor.predict(test_set))
        return y_test_pred
