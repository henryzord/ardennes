from collections import Counter

from evolution.graphical_models import *
from evolution.trees import Individual
import itertools as it

__author__ = 'Henry Cagnini'


class Ardennes(object):
    def __init__(self, n_individuals, max_height, n_iterations=100, threshold=0.9, uncertainty=0.01):
        self._n_individuals = n_individuals
        self._max_height = max_height
        self._n_iterations = n_iterations
        self._threshold = threshold
        self._uncertainty = uncertainty

    @staticmethod
    def group(lst, n):
        """
        Groups objects in a list into n-tuples.
        
        :param lst: A list.
        :param n: Size of the tuple.
        :return: Objects from the list grouped into n-tuples.
        """
        for i in range(0, len(lst), n):
            val = lst[i:i + n]
            if len(val) == n:
                yield tuple(val)
                # yield val
    
    def init_population(self, n_individuals, gm, sets):
        """
        
        :param n_individuals:
        :type gm: evolution.graphical_models.StartGM
        :param gm:
        :param sets:
        :return:
        """
        
        # def sample_below(level, n_upper=0):
        #     n_sample = np.power(2, level) * n_upper
        #     nodes = gm.sample_by_level(level, n_sample=n_sample)
        #     lower_count = Counter(nodes)
        #
        #     if level + 1 < self._max_height:
        #         return vfunc(level+1, lower_count.values())
        #     return lower_count.items()
        
        levels = map(
            lambda (l, n): list(self.group(gm.sample_by_level(level=l, n_sample=n), np.power(2, l))),
            it.izip(
                xrange(self._max_height),
                map(
                    lambda i: self._n_individuals * np.power(2, i), xrange(self._max_height)
                )
            )
        )
        
        trees = list(it.izip(*levels))
        # unique_trees = Counter(trees)
        some_other_trees = Individual.mash(trees, sets, self._max_height)
        z = 0
        
        # pop = np.array(
        #     map(
        #         lambda x: Individual(
        #             initial_pmf=gm,
        #             sets=sets,
        #             id=x
        #         ),
        #         xrange(n_individuals)
        #     )
        # )
        # return pop
    
    @staticmethod
    def early_stop(gm, diff=0.01):
        """

        :type gm: FinalGM
        :param gm: The Probabilistic Graphical Model (GM) for the current generation.
        :type diff: float
        :param diff: Maximum allowed uncertainty for each probability, for each node.
        :return:
        """
        
        def __max_prob__(node):
            max_prob = node['probs'].max().values[0]
            min_prob = node['probs'].min().values[0]
            
            val = max(
                abs(1. - max_prob),
                min_prob
            )
            
            return val
        
        nodes = gm.graph.node
        maximum_prob = reduce(
            max,
            map(
                __max_prob__,
                nodes.itervalues()
            )
        )
        if maximum_prob < diff:
            return True
        return False
    
    def fit_predict(self, sets, verbose=True):
        pred_attr = sets['train'].columns[:-1]
        target_attr = sets['train'].columns[-1]
        class_values = sets['train'][sets['train'].columns[-1]].unique()
        
        # pmf only for initializing the population
        gm = StartGM(
            pred_attr=pred_attr, target_attr=target_attr, class_values=class_values, max_height=self._max_height
        )
        
        population = self.init_population(
            n_individuals=self._n_individuals,
            gm=gm,
            sets=sets
        )
        
        heights = map(lambda x: x.height, population)
        
        # changes the pmf to a final one
        gm = FinalGM(pred_attr=pred_attr, target_attr=target_attr, class_values=class_values, population=population)
        
        fitness = np.array(map(lambda x: x.fitness, population))
        
        # threshold where individuals will be picked for PMF updating/replacing
        integer_threshold = int(self._threshold * self._n_individuals)
        
        iteration = 0
        while iteration < self._n_iterations:  # evolutionary process
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float
            _max = np.max(fitness)  # type: float
            
            if verbose:
                print 'iter: %03.d\tmean: %+0.6f\tmedian: %+0.6f\tmax: %+0.6f' % (iteration, mean, median, _max)
            
            # TODO slow. test other implementation!
            borderline = np.partition(fitness, integer_threshold)[integer_threshold]
            fittest_pop = population[np.flatnonzero(fitness >= borderline)]  # TODO slow. test other implementation!
            
            gm.update(fittest_pop)
            
            # warnings.warn('WARNIGN: Plotting gm!')
            # gm.plot()
            # plt.show()
            
            # warnings.warn('WARNING: Plotting fittest population!')
            # map(lambda x: x.plot(), fittest_pop)
            # plt.show()
            
            to_replace = population[np.flatnonzero(fitness < borderline)]  # TODO slow. test other implementation!
            for ind in to_replace:
                ind.sample_by_id(gm)
            
            if self.early_stop(gm, self._uncertainty):
                break
            
            fitness = np.array(map(lambda x: x.fitness, population))
            
            iteration += 1
        
        fittest_ind = population[np.argmax(fitness)]
        return fittest_ind
