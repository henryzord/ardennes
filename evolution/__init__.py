# coding=utf-8

from treelib.classes import AbstractEDA
from treelib.graphical_models import *

__author__ = 'Henry Cagnini'


class Ardennes(AbstractEDA):
    
    def fit_predict(self, verbose=True, **kwargs):
        if 'sets' not in kwargs or not all(map(lambda x: x in kwargs['sets'], ['train', 'val'])):  # TODO optimize!
            raise KeyError('You need to pass train and val sets to this method!')
        else:
            sets = kwargs['sets']

        class_values = {
            'pred_attr': list(sets['train'].columns[:-1]),
            'target_attr': sets['train'].columns[-1],
            'class_labels': list(sets['train'][sets['train'].columns[-1]].unique())
        }

        gm = GraphicalModel(**class_values)

        population = self.init_population(
            n_individuals=self.n_individuals,
            graphical_model=gm,
            sets=sets
        )

        fitness = np.array(map(lambda x: x.fitness, population))

        # threshold where individuals will be picked for PMF updating/replacing
        integer_threshold = int(self.decile * self.n_individuals)

        iteration = 0
        while iteration < self.n_iterations:  # evolutionary process
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float
            max_fitness = np.max(fitness)  # type: float

            self.verbose(
                iteration=iteration,
                mean=mean,
                median=median,
                max_fitness=max_fitness,
                verbose=verbose
            )

            borderline = np.partition(fitness, integer_threshold)[integer_threshold] # TODO slow. test other implementation!
            fittest_pop = population[np.flatnonzero(fitness >= borderline)]  # TODO slow. test other implementation!

            gm.update(fittest_pop)

            to_replace = population[np.flatnonzero(fitness < borderline)]  # TODO slow. test other implementation!
            for ind in to_replace:
                ind.sample_by_id(gm)

            if self.early_stop(gm, self.uncertainty):
                break

            fitness = np.array(map(lambda x: x.fitness, population))

            iteration += 1

        fittest_ind = population[np.argmax(fitness)]
        return fittest_ind
    
    @staticmethod
    def init_population(n_individuals, graphical_model, sets):
        population = np.array(map(
            lambda i: Individual(id=i, graphical_model=graphical_model, sets=sets),
            xrange(n_individuals)
        ))
        return population
    
    def verbose(self, **kwargs):
        iteration = kwargs['iteration']
        mean = kwargs['mean']
        median = kwargs['median']
        max_fitness = kwargs['max_fitness']
        
        if kwargs['verbose']:
            print 'iter: %03.d\tmean: %+0.6f\tmedian: %+0.6f\tmax: %+0.6f' % (iteration, mean, median, max_fitness)
    
    @staticmethod
    def early_stop(gm, diff=0.01):
        """

        :type gm: FinalAbstractGraphicalModel
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