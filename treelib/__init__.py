# coding=utf-8

import csv
import os
from datetime import datetime as dt

import pandas

from classes import type_check, value_check
from graphical_model import *
from individual import Individual
from information import Processor
from sklearn.metrics import accuracy_score

__author__ = 'Henry Cagnini'


class Ardennes(object):
    gm = None

    def __init__(self,
                 n_individuals=100, n_iterations=100, uncertainty=0.001,
                 decile=0.9, max_height=3, distribution='univariate',
                 class_probability='declining'):
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

        # threshold where individuals will be picked for PMF updating/replacing
        to_replace_index = np.arange(self.n_individuals - int(self.decile * self.n_individuals), self.n_individuals, dtype=np.int32)

        Individual.set_values(
            sets=sets,
            arg_sets=arg_sets,
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
            excluded=['gm']
        )

        t1 = dt.now()  # starts measuring time

        population = sample_func(
            ind_id=range(self.n_individuals), gm=gm
        )

        population = np.sort(population)[::-1]

        fitness = np.array([x.fitness for x in population])

        iteration = 0
        while iteration < self.n_iterations:  # evolutionary process
            t2 = dt.now()

            self.__report__(
                iteration=iteration,
                fitness=fitness,
                population=population,
                verbose=verbose,
                elapsed_time=(t2-t1).total_seconds(),
                gm=gm,
                **kwargs
            )
            t1 = t2

            fittest_pop = population[:to_replace_index[0]]

            gm.update(fittest_pop)

            if len(to_replace_index) > 0:
                population[to_replace_index] = sample_func(
                    ind_id=range(self.n_individuals), gm=gm
                )

                population = np.sort(population)[::-1]

            fitness = np.array([x.fitness for x in population])

            if self.__early_stop__(fitness, self.uncertainty):
                break

            iteration += 1

            population.max().plot()  # TODO remove!
            from matplotlib import pyplot as plt
            plt.show()

        # self.best_individual = sample_func(
        #     ind_id=0, graphical_model=gm, max_height=self.max_height, sets=sets,
        #     pred_attr=self.pred_attr, target_attr=self.target_attr, class_labels=self.class_labels
        # )

        self.gm = gm
        self.best_individual = np.argmax(fitness)
        self.last_population = population
        self.trained = True

    @property
    def tree_height(self):
        if self.trained:
            return self.last_population[self.best_individual].height

    def predict(self, samples, ensemble=False):
        df = self.__to_dataframe__(samples)

        if not ensemble:
            predictor = self.last_population[self.best_individual]
            all_preds = predictor.predict(df)
        else:
            predictor = self.last_population

            def sample_pred(sample):
                sample_predictions = map(lambda x: x.predict(sample), predictor)
                most_common = Counter(sample_predictions).most_common()[0][0]
                return most_common

            all_preds = df.apply(sample_pred, axis=1).as_matrix()

        return all_preds

    @staticmethod
    def __generate_columns__(n_predictive, make_class=True):
        columns = ['attr_%d' % d for d in xrange(n_predictive)]
        if make_class:
            columns += ['class']
        return columns

    def validate(self, test_set=None, X_test=None, y_test=None, ensemble=False):
        """
        Assess the accuracy of this instance against the provided set.

        :type test_set: pandas.DataFrame
        :param test_set: a matrix with the class attribute in the last position (i.e, column).
        :rtype: float
        :return: The accuracy of this model when testing with test_set.
        """

        if test_set is None:
            test_set = pd.DataFrame(
                np.hstack((X_test, y_test[:, np.newaxis])), columns=self.__generate_columns__(X_test.shape[1], make_class=True)
            )

        predictions = self.predict(test_set, ensemble=ensemble)
        acc = (test_set[test_set.columns[-1]] == predictions).sum() / float(test_set.shape[0])
        return acc

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

    @staticmethod
    def __pick_fittest_population__(population, borderline):
        def fit(x):
            return x.fitness >= borderline

        fittest_bool = population.apply(fit)
        fittest = population.loc[fittest_bool]

        return fittest

    @staticmethod
    def __report__(**kwargs):

        # required data, albeit this method has only a kwargs dictionary
        iteration = kwargs['iteration']  # type: int
        fitness = kwargs['fitness']  # type: np.ndarray
        population = kwargs['population']
        verbose = kwargs['verbose']
        elapsed_time = kwargs['elapsed_time']
        gm = kwargs['gm']

        best_individual = population[0]  # best individual in the population

        # optional data
        n_run = None if 'run' not in kwargs else kwargs['run']
        n_fold = None if 'fold' not in kwargs else kwargs['fold']
        dataset_name = None if 'dataset_name' not in kwargs else kwargs['dataset_name']
        output_path = None if 'output_path' not in kwargs else kwargs['output_path']

        if Individual.y_test_true is not None:
            y_pred = best_individual.predict(Individual.sets['test'])
            best_test_fitness = accuracy_score(Individual.y_test_true, y_pred)
        else:
            best_test_fitness = None

        if verbose:
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float

            print 'iter: %03.d  mean: %0.6f  median: %0.6f  max: %0.6f  ET: %02.2fsec  height: %2.d  n_nodes: %2.d  ' % (
                iteration, mean, median, best_individual.fitness, elapsed_time, best_individual.height, best_individual.n_nodes
            ) + ('test acc: %0.6f' % best_test_fitness if best_test_fitness is not None else '')

        if output_path is not None:
            evo_file = os.path.join(output_path, dataset_name + '_evo_fold_%03.d_run_%03.d.csv' % (n_fold, n_run))
            pgm_file = os.path.join(output_path, dataset_name + '_pgm_fold_%03.d_run_%03.d.csv' % (n_fold, n_run))

            with open(pgm_file, 'a') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')
                csv_w.writerow(gm.attributes.values.ravel())

            with open(evo_file, 'a') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')

                if iteration == 0:  # resets file
                    header_add = [] if Individual.y_test_true is None else ['test accuracy']
                    csv_w.writerow(['individual', 'iteration', 'validation accuracy', 'tree height'] + header_add)

                for ind in population:  # type: Individual
                    if Individual.y_test_true is not None:
                        y_pred = ind.predict(Individual.sets['test'])
                        add = [accuracy_score(Individual.y_test_true, y_pred)]
                    else:
                        add = []

                    csv_w.writerow([ind.ind_id, iteration, ind.fitness, ind.height] + add)

            population[np.argmax(fitness)].plot(
                savepath=evo_file.split('.')[0].strip() + '.pdf',
                test_acc=best_test_fitness
            )

    @staticmethod
    def __early_stop__(fitness, uncertainty):
        return abs(fitness.min() - fitness.max()) < uncertainty
