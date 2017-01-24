# coding=utf-8

import csv
import os
import random
from datetime import datetime as dt

from classes import type_check, value_check
from graphical_model import *
from individual import Individual
from information import Processor

__author__ = 'Henry Cagnini'


class Ardennes(object):
    gm = None

    best_overall = None  # TODO remove!

    def __init__(self,
                 n_individuals=100, n_iterations=100, uncertainty=0.001,
                 decile=0.9, max_height=3, distribution='univariate',
                 class_probability='declining', random_state=None):
        """
        Default EDA class, with common code to all EDAs -- regardless
        of the complexity of inner GMs or updating techniques.

        :type n_individuals: int
        :param n_individuals: Number of maximum individuals for a any population, throughout the evolutionary process.
        :param n_iterations: First (and most likely to be reached) stopping criterion. Maximum number of generations
            that this EDA is allowed to produce.
        :param uncertainty: Second stopping criterion. If this EDA's GM presents an uncertainty lesser than this
            parameter, then this EDA will likely stop before reaching the maximum number of iterations.
        :param decile: A parameter for determining how much of the population must be used for updatign the GM, and also
            how much of it must be resampled for the next generation. For example, if decile=0.9, then 10% of the
            population will be used for GM updating and 90% will be resampled.
        """
        if random_state is not None:
            warnings.warn('WARNING: Using non-randomic sampling with seed=%d' % random_state)

        random.seed(random_state)
        np.random.seed(random_state)

        self.n_individuals = n_individuals
        self.decile = decile
        self.uncertainty = uncertainty

        self.D = max_height - 1
        self.distribution = distribution
        self.n_iterations = n_iterations

        self.trained = False
        self.best_individual = None
        self.last_population = None

        self.class_probability = class_probability

        self.pred_attr = None
        self.target_attr = None
        self.class_labels = None

        self.processor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        GraphicalModel.clean()
        Individual.clean()
        self.clean()

    @classmethod
    def clean(cls):
        cls.pred_attr = None
        cls.target_attr = None
        cls.class_labels = None

    def fit(self, full, train, val=None, test=None, verbose=True, **kwargs):
        def __initialize_argsets__(_train, _val, _test):
            _arg_sets = dict()
            train_index = np.zeros(full.shape[0], dtype=np.bool)
            train_index[_train.index] = 1
            _arg_sets['train_index'] = train_index

            if _val is not None:
                val_index = np.zeros(full.shape[0], dtype=np.bool)
                val_index[_val.index] = 1
                _arg_sets['val_index'] = val_index
            else:
                _arg_sets['val_index'] = _arg_sets['train_index']

            if _test is not None:
                test_index = np.zeros(full.shape[0], dtype=np.bool)
                test_index[_test.index] = 1
                _arg_sets['test_index'] = test_index

            return _arg_sets

        def __initialize_sets__(_train, _val, _test):
            _sets = dict()
            _sets['train'] = _train
            if _val is not None:
                _sets['val'] = _val
            else:
                _sets['val'] = _sets['train']
            if _test is not None:
                _sets['test'] = _test

            return _sets

        self.processor = Processor(full)
        arg_sets = __initialize_argsets__(train, val, test)
        sets = __initialize_sets__(train, val, test)

        # from now on, considers only a dictionary 'sets' with train and val subsets

        class_values = {
            'pred_attr': list(full.columns[:-1]),
            'target_attr': full.columns[-1],
            'class_labels': np.sort(full[full.columns[-1]].unique()).tolist()
        }

        self.pred_attr = class_values['pred_attr']
        self.target_attr = class_values['target_attr']
        self.class_labels = class_values['class_labels']

        # TODO pass from label to binary representation!
        # self.binary_class = np.zeros()

        # threshold where individuals will be picked for PMF updating/replacing
        to_replace_index = np.arange(
            self.n_individuals - int(self.decile * self.n_individuals), self.n_individuals, dtype=np.int32
        )

        iteration = 0

        Individual.set_values(
            sets=sets,
            arg_sets=arg_sets,
            y_train_true=sets['train'][self.target_attr],
            y_val_true=sets['val'][self.target_attr],
            y_test_true=sets['test'][self.target_attr] if 'test' in sets else None,
            processor=self.processor,
            column_types={
                x: Individual.raw_type_dict[str(full[x].dtype)] for x in full.columns
            },
            pred_attr=self.pred_attr,
            target_attr=self.target_attr,
            class_labels=self.class_labels,
            max_height=self.D,
            full=full
        )

        gm = GraphicalModel(
            pred_attr=self.pred_attr,
            target_attr=self.target_attr,
            class_labels=self.class_labels,
            D=self.D,  # since last level must be all leafs
            distribution=self.distribution
        )

        sample_func = np.vectorize(
            Individual,
            excluded=['gm', 'iteration']
        )

        t1 = dt.now()  # starts measuring time

        population = sample_func(
            ind_id=range(self.n_individuals), gm=gm, iteration=iteration
        )

        population = np.sort(population)[::-1]

        last_best = np.random.rand(
            max(1,
                int(self.n_iterations * kwargs['threshold_stop'] if 'threshold_stop' in kwargs else .2)
            )
        )

        '''
            # --- Main loop --- #
        '''
        while iteration < self.n_iterations:
            t2 = dt.now()

            fitness = np.array([x.fitness for x in population])

            self.__report__(
                iteration=iteration,
                population=population,
                fitness=fitness,
                verbose=verbose,
                elapsed_time=(t2-t1).total_seconds(),
                gm=gm,
                **kwargs
            )
            t1 = t2

            if self.__early_stop__(last_best, iteration, population):
                break

            iteration += 1

            fittest_pop = population[:to_replace_index[0]]

            gm.update(fittest_pop)

            population.flat[to_replace_index] = sample_func(
                ind_id=range(self.n_individuals), gm=gm, iteration=iteration
            )

            population = np.sort(population)[::-1]

        self.gm = gm
        self.last_population = population
        self.best_individual = self.get_best_individual(population)
        self.trained = True

    @staticmethod
    def get_best_individual(population):
        return population.max()
        # argmax_general = np.where(population == population.max())
        # best_from = sorted(population[argmax_general], key=lambda x: x.val_acc_score)
        #
        # best_individual = best_from[-1]
        # return best_individual

    @property
    def tree_height(self):
        if self.trained:
            return self.best_individual.height

    def predict(self, samples):
        df = self.__to_dataframe__(samples)

        predictor = self.best_individual
        all_preds = predictor.predict(df)

        return all_preds

    @staticmethod
    def __generate_columns__(n_predictive, make_class=True):
        columns = ['attr_%d' % d for d in xrange(n_predictive)]
        if make_class:
            columns += ['class']
        return columns

    def __to_dataframe__(self, samples):
        if isinstance(samples, list):
            samples = np.array(samples)

        if isinstance(samples, np.ndarray):
            df = pd.DataFrame(samples, columns=self.__generate_columns__(samples.shape[1], make_class=False))
        elif isinstance(samples, pd.DataFrame):
            df = samples
        else:
            raise TypeError('Invalid type for samples! Must be either a list-like or a pandas.DataFrame!')

        return df

    def __report__(self, **kwargs):

        # required data, albeit this method has only a kwargs dictionary
        iteration = kwargs['iteration']  # type: int
        population = kwargs['population']
        verbose = kwargs['verbose']
        elapsed_time = kwargs['elapsed_time']
        fitness = kwargs['fitness']
        gm = kwargs['gm']

        best_individual = self.get_best_individual(population)

        best_overall = population[np.argmax([
            0.33 * (ind.test_acc_score + ind.val_acc_score + ind.train_acc_score) for ind in population
        ])]

        if self.best_overall is None:
            self.best_overall = best_overall

        if 0.33 * (best_overall.test_acc_score + best_overall.val_acc_score + best_overall.train_acc_score) > \
            0.33 * (self.best_overall.test_acc_score + self.best_overall.val_acc_score + self.best_overall.train_acc_score):
            self.best_overall = best_overall

        # optional data
        n_run = None if 'run' not in kwargs else kwargs['run']
        n_fold = None if 'fold' not in kwargs else kwargs['fold']
        dataset_name = None if 'dataset_name' not in kwargs else kwargs['dataset_name']
        output_path = None if 'output_path' not in kwargs else kwargs['output_path']

        if verbose:
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float

            print 'iter: %03.d  mean: %0.6f  median: %0.6f  max: %0.6f  ET: %02.2fsec  height: %2.d  n_nodes: %2.d  ' % (
                iteration, mean, median, best_individual.fitness, elapsed_time, best_individual.height, best_individual.n_nodes
            ) + ('test acc: %0.6f' % best_individual.test_acc_score if best_individual.test_acc_score is not None else '')

        if output_path is not None:
            evo_file = os.path.join(output_path, dataset_name + '_evo_fold_%03.d_run_%03.d.csv' % (n_fold, n_run))
            pgm_file = os.path.join(output_path, dataset_name + '_pgm_fold_%03.d_run_%03.d.csv' % (n_fold, n_run))

            with open(pgm_file, 'a') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')
                csv_w.writerow(gm.attributes.values.ravel())

            with open(evo_file, 'a') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')

                if iteration == 0:  # resets file
                    csv_w.writerow(['individual', 'iteration', 'fitness', 'height', 'n_nodes',
                                    'train_acc', 'val_acc', 'test_acc', 'test_precision', 'test_f1'])

                for ind in population:  # type: Individual
                    if Individual.y_test_true is not None:
                        add = [ind.test_acc_score, ind.test_precision_score, ind.test_f1_score]
                    else:
                        add = ['', '', '']

                    csv_w.writerow([ind.ind_id, iteration, ind.fitness, ind.height, ind.n_nodes,
                                    ind.train_acc_score, ind.val_acc_score] + add)

            best_individual.plot(
                savepath=evo_file.split('.')[0].strip() + '.pdf'
            )

            self.best_overall.plot(
                savepath=evo_file.split('.')[0].strip() + '_best_overall.pdf'
            )

    @staticmethod
    def __early_stop__(last_best, iteration, population):
        last_best[iteration % last_best.shape[0]] = population.max().fitness
        stop = abs(last_best.min() - last_best.max()) < 1e-06
        return stop
