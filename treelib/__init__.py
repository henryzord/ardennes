# coding=utf-8

import csv
import os
import random
import warnings
from datetime import datetime as dt

from classes import type_check, value_check, MetaDataset
from graphical_model import *
from individual import Individual
from device import AvailableDevice
from treelib.individual import DecisionTree
import cPickle

__author__ = 'Henry Cagnini'


class Ardennes(object):
    global_best = None  # TODO remove once problem with suboptimal individuals is solved

    def __init__(self, n_individuals, n_iterations, uncertainty=0.001, max_height=3):

        self.n_individuals = n_individuals
        self.uncertainty = uncertainty

        self.D = max_height - 1
        self.n_iterations = n_iterations

        self.trained = False
        self.predictor = None

    @staticmethod
    def __initialize_argsets__(dataset, train, val, test):
        _arg_sets = dict()

        train_index = np.zeros(dataset.shape[0], dtype=np.bool)
        val_index = np.zeros(dataset.shape[0], dtype=np.bool)
        test_index = np.zeros(dataset.shape[0], dtype=np.bool)

        train_index[train] = 1
        val_index[val] = 1
        test_index[test] = 1

        _arg_sets['train'] = train_index
        _arg_sets['val'] = val_index
        _arg_sets['test'] = test_index

        return _arg_sets

    def __setup__(self, train, **kwargs):
        if 'random_state' in kwargs and kwargs['random_state'] is not None:
            random_state = kwargs['random_state']
            warnings.warn('WARNING: Using non-randomic sampling with seed=%d' % random_state)
        else:
            random_state = None

        random.seed(random_state)
        np.random.seed(random_state)

        dataset = kwargs['full'] if 'full' in kwargs else train
        val = kwargs['validation'] if 'validation' in kwargs else train
        test = kwargs['test'] if 'test' in kwargs else train

        arg_sets = self.__initialize_argsets__(dataset, train, val, test)

        dataset_info = MetaDataset(dataset)

        mdevice = AvailableDevice(dataset, dataset_info)

        DecisionTree.set_values(
            arg_sets=arg_sets,
            y_train_true=dataset.loc[arg_sets['train'], dataset_info.target_attr],
            y_val_true=dataset.loc[arg_sets['val'], dataset_info.target_attr],
            y_test_true=dataset.loc[test, dataset_info.target_attr] if test is not None else None,
            processor=mdevice,
            dataset_info=dataset_info,
            max_height=self.D,
            dataset=dataset,
            mdevice=mdevice
        )

        gm = GraphicalModel(
            D=self.D,
            dataset_info=dataset_info,
        )

        return gm

    def fit(self, train, decile, verbose=True, **kwargs):
        """
        Fits the

        :param train: train set.
        :type decile: float
        :param decile: decile of individuals which will be used for inducing the graphical model.
        :type verbose: bool
        :param verbose: optional - whether to print metadata to console.
        :param full: optional - full dataset. Will use train if not provided.
        :param validation: optional - Validation set.
        :param test: optional - test set.
        :type fold: int
        :param fold: optional - current fold.
        :type run: int
        :param run: optional - current run.
        :type random_state: int
        :param random_state: optional - random seed, which affects values sampled in the graphical model.
        :type n_stop: int
        :param n_stop: optional - maximum number of generations with the same best individual unchanged. Upon reaching
            this value the evolutionary procedure will stop.
        :type output_path: str
        :param output_path: optional - path to output metadata.
        """

        '''
        output_path=config_file['output_path'] if 'output_path' in config_file else None,  # kwargs
        '''

        assert 1 <= int(self.n_individuals * decile) <= self.n_individuals, \
            ValueError('Decile must comprise at least one individual and at maximum the whole population!')

        gm = self.__setup__(train=train, **kwargs)

        sample_func = np.vectorize(Individual, excluded=['gm', 'iteration'])

        population = np.empty(shape=self.n_individuals, dtype=Individual)
        to_replace_index = np.arange(self.n_individuals, dtype=np.int32)

        '''
            # --- Main loop --- #
        '''
        iteration = 0

        while iteration < self.n_iterations:
            t1 = dt.now()  # starts measuring time

            fitness, population = self.sample_population(gm, iteration, sample_func, to_replace_index, population)

            to_replace_index, fittest_pop = self.split_population(decile, population)

            gm.update(fittest_pop)

            t2 = dt.now()
            self.__report__(
                iteration=iteration,
                population=population,
                fitness=fitness,
                verbose=verbose,
                elapsed_time=(t2 - t1).total_seconds(),
                gm=gm,
                **kwargs
            )

            if self.__early_stop__(population):
                break

            iteration += 1

        self.predictor = self.get_best_individual(population)
        self.trained = True

    @staticmethod
    def sample_population(gm, iteration, func, to_replace_index, population):
        """

        :type gm: treelib.graphical_model.GraphicalModel
        :param gm: Current graphical model.
        :type iteration: int
        :param iteration: Current iteration.
        :param func: Sample function.
        :type to_replace_index: list
        :param to_replace_index: List of indexes of individuals to be replaced in the following generation.
        :type population: numpy.ndarray
        :param population: Current population.
        :rtype: tuple
        :return: A tuple where the first item is the population fitness and the second the population.
        """

        population.flat[to_replace_index] = func(
            ind_id=to_replace_index, gm=gm, iteration=iteration
        )
        population.sort()
        population = population[::-1]

        fitness = np.array([x.fitness for x in population])

        return fitness, population

    def split_population(self, decile, population):
        integer_decile = int(self.n_individuals * decile)

        to_replace_index = [ind.ind_id for ind in population[integer_decile:]]
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

    def __report__(self, **kwargs):
        # required data, albeit this method has only a kwargs dictionary
        iteration = kwargs['iteration']  # type: int
        population = kwargs['population']
        verbose = kwargs['verbose']
        elapsed_time = kwargs['elapsed_time']
        fitness = kwargs['fitness']
        gm = kwargs['gm']

        sep = '/' if os.name is not 'nt' else '\\'

        best_individual = self.get_best_individual(population)

        global_best = population[np.argmax([
            0.33 * (ind.test_acc_score + ind.val_acc_score + ind.train_acc_score) for ind in population
        ])]

        if self.global_best is None:
            self.global_best = global_best

        if 0.33 * (global_best.test_acc_score + global_best.val_acc_score + global_best.train_acc_score) > \
           0.33 * (self.global_best.test_acc_score + self.global_best.val_acc_score + self.global_best.train_acc_score):
            self.global_best = global_best

        # optional data
        n_run = None if 'run' not in kwargs else kwargs['run']
        n_fold = None if 'fold' not in kwargs else kwargs['fold']
        output_path = None if 'output_path' not in kwargs else kwargs['output_path']
        dataset_name = output_path.split(sep)[-1] if output_path is not None else None

        if verbose:
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float

            print 'iter: %03.d mean: %0.6f median: %0.6f max: %0.6f ET: %02.2fsec  height: %2.d  n_nodes: %2.d  ' % (
                iteration, mean, median, best_individual.fitness, elapsed_time, best_individual.height, best_individual.n_nodes
            ) + ('test acc: %0.6f' % best_individual.test_acc_score if best_individual.test_acc_score is not None else '')

        if output_path is not None:
            evo_file = os.path.join(output_path, dataset_name + '_evo_fold_%03.d_run_%03.d.csv' % (n_fold, n_run))
            pgm_file = os.path.join(output_path, dataset_name + '_pgm_fold_%03.d_run_%03.d.csv' % (n_fold, n_run))

            with open(pgm_file, 'a') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')
                csv_w.writerow(gm.attributes.values.ravel())

            with open(evo_file, 'w') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')

                if iteration == 0:  # resets file
                    csv_w.writerow([
                        'individual', 'iteration', 'fitness', 'height', 'n_nodes',
                        'train_correct', 'train_total', 'val_correct', 'val_total', 'test_correct', 'test_total'
                    ])

            with open(evo_file, 'a') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')

                for ind in population:  # type: Individual
                    if Individual.y_test_true is not None:
                        add = [int(ind.test_acc_score * len(ind.y_test_true)), len(ind.y_test_true)]
                    else:
                        add = ['', '']

                    csv_w.writerow([
                        ind.ind_id, iteration, ind.fitness, ind.height, ind.n_nodes,
                        int(ind.train_acc_score * len(ind.y_train_true)), len(ind.y_train_true),
                        int(ind.val_acc_score * len(ind.y_val_true)), len(ind.y_val_true)] + add
                    )

            best_individual.plot(
                savepath=evo_file.split('.')[0].strip() + '.pdf'
            )

            self.global_best.plot(
                savepath=evo_file.split('.')[0].strip() + '_best_overall.pdf'
            )

            cPickle.dump(best_individual, open(evo_file.split('.')[0].strip() + '.bin', 'w'))

    @staticmethod
    def __early_stop__(population):
        return population.min() == population.max()
