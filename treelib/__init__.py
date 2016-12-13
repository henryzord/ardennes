# coding=utf-8
import warnings

from graphical_model import *
from classes import type_check, value_check
from individual import Individual
from sklearn.tree.tree import DecisionTreeClassifier

from collections import Counter
from datetime import datetime as dt
import os

import numpy as np
import pandas
import csv
import copy

__author__ = 'Henry Cagnini'


def get_max_height(train_set, random_state=None):
    """
    Picks the maximum height for a decision tree induced by the Scikit-Learn deterministic algorithm.

    :param train_set:
    :param random_state:
    :return:
    """

    if isinstance(train_set, pd.DataFrame):
        x_train = train_set[train_set.columns[:-1]]
        y_train = train_set[train_set.columns[-1]]
    elif isinstance(train_set, tuple):
        x_train = train_set[0]
        y_train = train_set[1]
    else:
        raise TypeError('Invalid type for this function! Must be either a pandas.DataFrame or a tuple of numpy.ndarray!')

    try:
        cls = DecisionTreeClassifier(
            criterion='entropy', random_state=random_state, min_samples_split=2, min_samples_leaf=1
        )
        cls = cls.fit(x_train, y_train)
        max_depth = cls.tree_.max_depth
        return max_depth
    except ValueError as ve:
        ve.message = 'This function only supports datasets with numerical predictive attributes!'
        raise ve


class Ardennes(object):
    gm = None

    def __init__(self,
                 n_individuals=100, n_iterations=100, uncertainty=0.01,
                 decile=0.9, max_height=3, distribution='multivariate',
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

        self.max_height = max_height
        self.distribution = distribution
        self.n_iterations = n_iterations

        self.trained = False
        self.best_individual = None
        self.last_population = None

        self.class_probability = class_probability

        self.pred_attr = None
        self.target_attr = None
        self.class_labels = None

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

    def fit(self, train, val=None, verbose=True, output_path=None, **kwargs):
        def __treat__(_train, _val, _test=None):
            type_check(_train, [pd.DataFrame, tuple])

            _sets = dict()

            if isinstance(_train, tuple):
                _sets['train'] = pd.DataFrame(np.hstack((_train[0], _train[1][:, np.newaxis])), columns=self.__generate_columns__(_train[0].shape[1], make_class=True))
            elif isinstance(_train, pd.DataFrame):
                _sets['train'] = _train

            if _val is not None:
                type_check(_val, [pd.DataFrame, tuple])

                if isinstance(_val, tuple):
                    _sets['val'] = pd.DataFrame(np.hstack((_val[0], _val[1][:, np.newaxis])), columns=self.__generate_columns__(_val[0].shape[1], make_class=True))
                else:
                    _sets['val'] = _val
            else:
                _sets['val'] = _sets['train']

            if _test is not None:
                type_check(_test, [pd.DataFrame, tuple])

                if isinstance(_test, tuple):
                    _sets['test'] = pd.DataFrame(np.hstack((_test[0], _test[1][:, np.newaxis])), columns=self.__generate_columns__(_test[0].shape[1], make_class=True))
                else:
                    _sets['test'] = _test

            return _sets

        sets = __treat__(train, val, kwargs['test'] if 'test' in kwargs else None)

        # from now on, considers only a dictionary 'sets' with train and val subsets

        class_values = {
            'pred_attr': list(sets['train'].columns[:-1]),
            'target_attr': sets['train'].columns[-1],
            'class_labels': np.sort(sets['train'][sets['train'].columns[-1]].unique()).tolist()
        }

        self.pred_attr = class_values['pred_attr']
        self.target_attr = class_values['target_attr']
        self.class_labels = class_values['class_labels']

        # threshold where individuals will be picked for PMF updating/replacing
        integer_threshold = int(self.decile * self.n_individuals)

        t1 = dt.now()  # starts measuring time

        gm = GraphicalModel(
            pred_attr=self.pred_attr,
            target_attr=self.target_attr,
            class_labels=self.class_labels,
            max_depth=self.max_height - 1,  # since last level must be all leafs
            distribution=self.distribution
        )

        sample_func = np.vectorize(
            Individual,
            excluded=['graphical_model', 'max_height', 'sets', 'pred_attr', 'target_attr', 'class_labels']
        )

        population = sample_func(
            ind_id=range(self.n_individuals), graphical_model=gm, max_height=self.max_height, sets=sets,
            pred_attr=self.pred_attr, target_attr=self.target_attr, class_labels=self.class_labels
        )

        fitness = np.array([x.fitness for x in population])

        iteration = 0
        while iteration < self.n_iterations:  # evolutionary process
            t2 = dt.now()

            self.__report__(
                iteration=iteration,
                fitness=fitness,
                population=population,
                verbose=verbose,
                output_path=output_path,
                elapsed_time=(t2-t1).total_seconds(),
                test_set=sets['test'] if 'test' in sets else None,
                **kwargs
            )
            t1 = t2

            borderline = np.partition(fitness, integer_threshold)[integer_threshold]

            fittest_pop = population[fitness > borderline]
            # to_replace_index = np.arange(self.n_individuals)
            # warnings.warn('WARNING: replacing whole population!')
            to_replace_index = np.flatnonzero(fitness < borderline)

            gm.update(fittest_pop)

            if len(to_replace_index) > 0:
                population[to_replace_index] = sample_func(
                    ind_id=to_replace_index, graphical_model=gm, max_height=self.max_height,
                    sets=self.__get_local_sets__(sets),
                    pred_attr=self.pred_attr, target_attr=self.target_attr, class_labels=self.class_labels
                )

            fitness = np.array([x.fitness for x in population])

            warnings.warn('WARNING: Plotting best individual!')
            # TODO remove me!
            population[np.argmax(fitness)].plot(savepath='/home/henry/Desktop/plots/%03.d.pdf' % iteration, test_set=sets['test'])

            if self.__early_stop__(fitness, self.uncertainty):
                break

            iteration += 1

        # self.best_individual = sample_func(
        #     ind_id=0, graphical_model=gm, max_height=self.max_height, sets=sets,
        #     pred_attr=self.pred_attr, target_attr=self.target_attr, class_labels=self.class_labels
        # )

        self.gm = gm
        self.best_individual = np.argmax(fitness)
        self.last_population = population
        self.trained = True

    def __get_local_sets__(self, sets):
        warnings.warn('WARNING: using a subset of train for induction!')
        train_set = sets['train']
        val_set = sets['val']

        train_train_index = np.random.choice(train_set.index, size=int(train_set.shape[0] * 0.5), replace=False)
        train_test_index = list(set(train_set.index) - set(train_train_index))

        cpy = val_set.append(train_set.loc[train_test_index])

        return {'train': train_set.loc[train_train_index], 'val': cpy}

    def predict_proba(self, samples, ensemble=False):
        raise NotImplementedError('not implemented yet!')

        df = self.__to_dataframe__(samples)

        if not ensemble:
            # using predict_proba with a single tree has the same effect as simply using predict
            predictor = self.last_population[self.best_individual]
            all_preds = predictor.predict(df)
        else:
            predictor = self.last_population

            labels = {label: i for i, label in enumerate(self.class_labels)}

            def sample_prob(sample):
                preds = np.empty(len(self.class_labels), dtype=np.float32)

                sample_predictions = map(lambda x: x.predict(sample), predictor)
                count = Counter(sample_predictions)
                count_probs = {k: v / float(len(predictor)) for k, v in count.iteritems()}
                for k, v in count_probs.items():  # TODO wrong!
                    preds[labels[k]] = v

                return preds

            all_preds = df.apply(sample_prob, axis=1).as_matrix()

        return all_preds

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
        iteration = kwargs['iteration']  # type: int
        fitness = kwargs['fitness']  # type: np.ndarray
        population = kwargs['population']
        test_set = kwargs['test_set'] if 'test_set' in kwargs else None

        if 'output_path' in kwargs:
            if kwargs['output_path'] is not None and 'fold' in kwargs and 'run' in kwargs:
                output_file = os.path.join(
                    kwargs['output_path'], kwargs['dataset_name'] + '_fold_%03.d_run_%03.d.csv' % (
                        kwargs['fold'], kwargs['run']
                    )
                )
            else:
                output_file = None
        else:
            output_file = None

        if kwargs['verbose']:
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float
            max_fitness = np.max(fitness)  # type: float
            elapsed_time = kwargs['elapsed_time']
            best_test_fitness = population[np.argmax(fitness)].validate(test_set)

            print 'iter: %03.d\tmean: %0.6f\tmedian: %0.6f\tmax: %0.6f\tET: %02.2fsec' % (iteration, mean, median, max_fitness, elapsed_time) + ' Max test: %0.6f' % best_test_fitness if test_set is not None else ''

        if output_file is not None:
            if iteration == 0:  # resets file
                try:
                    os.remove(output_file)
                except OSError as ose:
                    pass
                finally:
                    with open(output_file, 'w') as f:
                        csv_w = csv.writer(f, delimiter=',', quotechar='\"')
                        add = [] if test_set is None else ['test accuracy']

                        csv_w.writerow(['individual', 'iteration', 'validation accuracy', 'tree height'] + add)

            with open(output_file, 'a') as f:
                csv_w = csv.writer(f, delimiter=',', quotechar='\"')
                for ind in population:
                    add = [] if test_set is None else [ind.validate(test_set)]
                    csv_w.writerow([ind.id_ind, iteration, ind.fitness, ind.height] + add)

            population[np.argmax(fitness)].plot(
                savepath=output_file.split('.')[0].strip() + '.pdf',
                test_set=test_set
            )

    @staticmethod
    def __early_stop__(fitness, uncertainty):
        return abs(fitness.min() - fitness.max()) < uncertainty
