# coding=utf-8
import json
import random
from datetime import datetime as dt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from graphical_model import GraphicalModel
from reporter import EDAReporter
from solution import Individual
from utils import get_dataset_name, __get_fold__

__author__ = 'Henry Cagnini'


class Ardennes(object):
    def __init__(self, n_individuals, n_generations, max_height=3, decile=0.5, reporter=None):

        self.n_individuals = n_individuals

        self.max_height = max_height
        self.max_depth = self.max_height - 1
        self.n_generations = n_generations

        self.decile = decile

        self.trained = False
        self.predictor = None

        self.reporter = reporter

    def __setup__(self, full_df, train_index, val_index):
        # mdevice = AvailableDevice(full_df, dataset_info)

        raw_individuals = np.array([
                Individual(
                    full_df=full_df,
                    max_depth=self.max_depth,
                    train_index=train_index,
                    val_index=val_index,
                ) for x in range(self.n_individuals)
            ],
            dtype=Individual
        )

        population = pd.DataFrame({
            'P': raw_individuals,
            'P_fitness': np.empty(shape=self.n_individuals, dtype=np.float32),
            'A': np.zeros(self.n_individuals, dtype=np.bool),
        })

        gm = GraphicalModel(
            max_depth=self.max_depth,
            full_df=full_df
        )

        return gm, population

    def fit(self, full_df, train_index, val_index, verbose=True):
        """
        Fits the algorithm to the provided data.
        """

        # overrides prior seed
        # TODO remove this seed!
        np.random.seed(0)
        random.seed(0)

        assert 1 <= int(self.n_individuals * self.decile) <= self.n_individuals, \
            ValueError('Decile must comprise at least one individual and at maximum the whole population!')

        gm, population = self.__setup__(full_df=full_df, train_index=train_index, val_index=val_index)

        # main loop
        g = 0

        while g < self.n_generations:
            t1 = dt.now()
            population = gm.sample(population)

            population = self.split_population(population, self.decile)

            gm.__update_after_sampling__(population)

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


def train_ardennes(dataset_path, output_path, params_path, n_fold, n_run):
    params = json.load(open(params_path))

    dataset_name = get_dataset_name(dataset_path)

    df, rest_index, test_index = __get_fold__(params=params, dataset_path=dataset_path, n_fold=n_fold)

    X_test = df.loc[test_index, df.columns[:-1]]
    y_test = df.loc[test_index, df.columns[-1]]
    del test_index  # deletes it to prevent from being using later

    rest_df = df.loc[rest_index]  # type: pd.DataFrame
    rest_df.reset_index(inplace=True, drop=True)

    rest_y = rest_df[rest_df.columns[-1]]
    frac = (float(params['n_folds']) - 2) / (float(params['n_folds']) - 1)

    train_index, val_index = train_test_split(
        rest_df.index, train_size=frac,
        shuffle=True, random_state=params['random_state'], stratify=rest_y
    )

    n_classes = len(np.unique(rest_y))

    X_train = rest_df.loc[train_index, rest_df.columns[:-1]]
    y_train = rest_df.loc[train_index, rest_df.columns[-1]]

    X_val = rest_df.loc[val_index, rest_df.columns[:-1]]
    y_val = rest_df.loc[val_index, rest_df.columns[-1]]

    train_index_bool = np.zeros(len(rest_y), dtype=np.bool)
    train_index_bool[train_index] = True
    val_index_bool = np.zeros(len(rest_y), dtype=np.bool)
    val_index_bool[val_index] = True

    reporter = EDAReporter(
        Xs=[X_train, X_val],
        ys=[y_train, y_val],
        n_classes=n_classes,
        set_names=['train', 'val'],
        dataset_name=dataset_name,
        n_fold=n_fold,
        n_run=n_run,
        output_path=output_path,
    )

    model = Ardennes(
        n_individuals=params['n_individuals'],
        n_generations=params['n_generations'],
        max_height=params['max_height'],
        decile=params['decile'],
        reporter=reporter,
    )

    model = model.fit(
        full_df=rest_df,
        train_index=train_index_bool,
        val_index=val_index_bool
    )

    return model
