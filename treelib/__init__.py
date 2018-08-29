# coding=utf-8
import random
from datetime import datetime as dt

from sklearn.metrics import accuracy_score

from device import AvailableDevice
from graphical_model import *
from individual import Individual
from treelib.individual import DecisionTree
from utils import MetaDataset, DatabaseHandler

__author__ = 'Henry Cagnini'


class Ardennes(object):
    val_str = 'val_df'
    train_str = 'train_df'
    test_str = 'test_df'

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

        # raise NotImplementedError('solve X_train and X_val!')

        sample_func = np.vectorize(Individual, excluded=['gm', 'iteration'])

        P = np.empty(shape=self.n_individuals, dtype=Individual)
        A_minus = np.arange(self.n_individuals, dtype=np.int32)  # former to_replace_index

        # main loop
        g = 0

        while g < self.n_generations:
            t1 = dt.now()
            P_fitness, P = self.sample_population(gm, g, sample_func, A_minus, P)

            A_minus, fittest_pop = self.split_population(self.decile, P)

            gm.update(fittest_pop)

            # TODO reactivate!
            # self.__save__(
            #     reporter=self.reporter, generation=g, A=A, P=P, P_fitness=P_fitness, loc=loc, scale=scale
            # )

            if self.__early_stop__(P):
                break

            median = float(np.median(P_fitness, axis=0))  # type: float

            if verbose:
                print('generation %02.d: best: %.4f median: %.4f time elapsed: %f' % (
                    g, np.max(P_fitness), median, (dt.now() - t1).total_seconds()
                ))

            g += 1

        self.predictor = self.get_best_individual(P)
        self.trained = True
        return self

    @staticmethod
    def sample_population(gm, iteration, sample_func, to_replace_index, P):
        """

        :type gm: treelib.graphical_model.GraphicalModel
        :param gm: Current graphical model.
        :type iteration: int
        :param iteration: Current iteration.
        :param sample_func: Sample function.
        :type to_replace_index: list
        :param to_replace_index: List of indexes of individuals to be replaced in the following generation.
        :type P: numpy.ndarray
        :param P: Current population.
        :rtype: tuple
        :return: A tuple where the first item is the population fitness and the second the population.
        """

        P.flat[to_replace_index] = sample_func(
            ind_id=[P[i].ind_id for i in to_replace_index] if iteration > 0 else to_replace_index,
            gm=gm,
            iteration=iteration
        )
        P.sort()  # sorts using quicksort, worst individual to best
        P = P[::-1]  # reverses list so the best individual is in the beginning

        P_fitness = np.array([x.fitness for x in P])

        return P_fitness, P

    def split_population(self, decile, population):
        integer_decile = int(self.n_individuals * decile)

        # refers to indices in the array, not in the population (i.e. individual.ind_id)
        to_replace_index = range(self.n_individuals)[integer_decile:]
        fittest_pop = population[:integer_decile]

        return to_replace_index, fittest_pop

    @staticmethod
    def get_best_individual(population):
        outer_fitness = [0.5 * (ind.train_acc_score + ind.val_acc_score) for ind in population]
        return population[np.argmax(outer_fitness)]

    @property
    def tree_height(self):
        if self.trained:
            return self.predictor.height

    def __save__(self, **kwargs):
        # required data, albeit this method has only a kwargs dictionary
        iteration = kwargs['iteration']  # type: int
        population = kwargs['population']
        elapsed_time = kwargs['elapsed_time']
        fitness = kwargs['fitness']
        gm = kwargs['gm']

        best_individual = self.get_best_individual(population)

        # optional data
        dbhandler = None if 'dbhandler' not in kwargs else kwargs['dbhandler']  # type: utils.DatabaseHandler

        mean = np.mean(fitness)  # type: float
        median = np.median(fitness)  # type: float

        print 'iter: %03.d mean: %0.6f median: %0.6f max: %0.6f ET: %02.2fsec  height: %2.d  n_nodes: %2.d  ' % (
            iteration, mean, median, best_individual.fitness, elapsed_time, best_individual.height, best_individual.n_nodes
        ) + ('test acc: %0.6f' % best_individual.test_acc_score if best_individual.test_acc_score is not None else '')

        if dbhandler is not None:
            dbhandler.write_prototype(iteration, gm)
            dbhandler.write_population(iteration, population)

    @staticmethod
    def __early_stop__(population):
        return population.min() == population.max()

    def predict(self, test_set):
        y_test_pred = list(self.predictor.predict(test_set))
        return y_test_pred
