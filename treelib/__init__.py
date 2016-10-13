# coding=utf-8

from graphical_models import *
from classes import AbstractTree, type_check, value_check
from individual import Individual

from collections import Counter
from datetime import datetime as dt

import numpy as np
import pandas


__author__ = 'Henry Cagnini'


class Ardennes(AbstractTree):
    gm = None

    def __init__(self,
                 n_individuals=100, n_iterations=100, uncertainty=0.01,
                 decile=0.9, initial_tree_size=3, distribution='multivariate',
                 class_probability='decreased', **kwargs
                 ):
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
        super(Ardennes, self).__init__(**kwargs)

        self.n_individuals = n_individuals
        self.n_iterations = n_iterations
        self.uncertainty = uncertainty
        self.decile = decile

        self.trained = False
        self.best_individual = None
        self.last_population = None
        self.initial_tree_size = initial_tree_size
        self.distribution = distribution
        self.class_probability = class_probability

    def fit(self, train, val=None, verbose=True, output_file=None):
        def __treat__(_train, _val):
            type_check(_train, [pd.DataFrame, tuple])

            _sets = dict()

            if isinstance(_train, tuple):
                _sets['train'] = pd.DataFrame(np.hstack((_train[0], _train[1][:, np.newaxis])))
            elif isinstance(_train, pd.DataFrame):
                _sets['train'] = _train

            if val is not None:
                type_check(_val, [pd.DataFrame, tuple])

                if isinstance(val, tuple):
                    _sets['val'] = pd.DataFrame(np.hstack((_val[0], _val[1][:, np.newaxis])))
                else:
                    _sets['val'] = _val
            else:
                _sets['val'] = _sets['train']

            return _sets

        sets = __treat__(train, val)

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

        df_replace = pd.DataFrame(np.empty((self.n_individuals, self.initial_tree_size), dtype=np.object))

        gm = GraphicalModel(
            initial_tree_size=self.initial_tree_size,
            distribution=self.distribution,
            class_probability=self.class_probability,
            **class_values
        )

        population = self.sample_individuals(df=df_replace, graphical_model=gm, sets=sets)

        fitness = np.array([x.fitness for x in population])

        iteration = 0
        while iteration < self.n_iterations:  # evolutionary process
            t2 = dt.now()

            self.__report__(
                iteration=iteration,
                fitness=fitness,
                verbose=verbose,
                output_file=output_file,
                elapsed_time=(t2-t1).total_seconds()
            )
            t1 = t2

            borderline = np.partition(fitness, integer_threshold)[integer_threshold]

            # picks fittest population
            fittest_pop = self.__pick_fittest_population__(population, borderline)  # type: pd.Series
            gm.update(fittest_pop)

            to_replace_integer = self.n_individuals - fittest_pop.shape[0]
            if to_replace_integer <= 0:
                to_replace_integer = self.n_individuals

            df_replace = pd.DataFrame(
                np.empty((to_replace_integer, self.initial_tree_size), dtype=np.object)
            )

            replaced = self.sample_individuals(df=df_replace, graphical_model=gm, sets=sets)  # type: pd.DataFrame
            if to_replace_integer == self.n_individuals:
                population = replaced
            else:
                population = replaced.append(fittest_pop, ignore_index=True)
                population.reset_index(inplace=True, drop=True)

            if self.__early_stop__(gm, self.uncertainty):
                break

            fitness = np.array([x.fitness for x in population])

            iteration += 1

        self.best_individual = np.argmax(fitness)
        self.last_population = population
        self.trained = True
        GraphicalModel.reset_globals()

    def predict_proba(self, samples, ensemble=False):
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
                np.hstack((X_test, y_test[:, np.newaxis]))
            )

        predictions = self.predict(test_set, ensemble=ensemble)
        acc = (test_set[test_set.columns[-1]] == predictions).sum() / float(test_set.shape[0])
        return acc

    def plot(self):
        raise NotImplementedError('not implemented yet!')

    @staticmethod
    def sample_individuals(df, graphical_model, sets):
        df.reset_index(drop=True, inplace=True)
        df = graphical_model.sample(df)

        def create_individual(row):
            ind = Individual(id=row.index, sess=row, sets=sets)
            return ind

        population = df.apply(create_individual, axis=1)

        return population

    @staticmethod
    def __to_dataframe__(samples):
        if isinstance(samples, np.ndarray) or isinstance(samples, list):
            df = pd.DataFrame(samples)
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

        if kwargs['verbose']:
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float
            max_fitness = np.max(fitness)  # type: float
            elapsed_time = kwargs['elapsed_time']

            print 'iter: %03.d\tmean: %0.6f\tmedian: %0.6f\tmax: %0.6f\tET: %0.2fsec' % (iteration, mean, median, max_fitness, elapsed_time)

        if kwargs['output_file']:
            output_file = kwargs['output_file']  # type: str
            with open(output_file, 'a') as f:
                np.savetxt(f, fitness[:, np.newaxis].T, delimiter=',')

    @staticmethod
    def __early_stop__(gm, uncertainty=0.01):
        """

        :type gm: treelib.graphical_models.GraphicalModel
        :param gm: The Probabilistic Graphical Model (GM) for the current generation.
        :type uncertainty: float
        :param uncertainty: Maximum allowed uncertainty for each probability, for each node.
        :return:
        """

        should_stop = True
        for tensor in gm.tensors:
            weights = tensor.weights
            upper = abs(1. - weights['probability'].max())
            lower = weights['probability'].min()

            if upper > uncertainty or lower > uncertainty:
                should_stop = False
                break

        return should_stop

    @staticmethod
    def __check_tree_size__(initial_tree_size):
        if (initial_tree_size - 1) % 2 != 0:
            raise ValueError('Invalid number of nodes! (initial_tree_size - 1) % 2 must be an integer!')