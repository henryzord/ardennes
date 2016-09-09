from evolution.trees import Individual
from evolution.graphical_models import *

__author__ = 'Henry Cagnini'


class Ardennes(object):
    def __init__(self):
        pass
    
    @staticmethod
    def init_population(n_individuals, gm, sets):
        pop = np.array(
            map(
                lambda x: Individual(
                    initial_pmf=gm,
                    sets=sets,
                    id=x
                ),
                xrange(n_individuals)
            )
        )
        return pop
    
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
    
    def fit_predict(self, sets, n_individuals, target_add, n_iterations=100, inf_thres=0.9, diff=0.01, verbose=True):
        pred_attr = sets['train'].columns[:-1]
        target_attr = sets['train'].columns[-1]
        class_values = sets['train'][sets['train'].columns[-1]].unique()
        
        # pmf only for initializing the population
        gm = StartGM(pred_attr=pred_attr, target_attr=target_attr, class_values=class_values, target_add=target_add)
        
        population = self.init_population(
            n_individuals=n_individuals,
            gm=gm,
            sets=sets
        )
        
        # changes the pmf to a final one
        gm = FinalGM(pred_attr=pred_attr, target_attr=target_attr, class_values=class_values, population=population)
        
        fitness = np.array(map(lambda x: x.fitness, population))
        
        # threshold where individuals will be picked for PMF updating/replacing
        integer_threshold = int(inf_thres * n_individuals)
        
        n_past = 15
        past = np.random.rand(n_past)
        
        iteration = 0
        while iteration < n_iterations:  # evolutionary process
            mean = np.mean(fitness)  # type: float
            median = np.median(fitness)  # type: float
            _max = np.max(fitness)  # type: float
            
            if verbose:
                print 'iter: %03.d\tmean: %+0.6f\tmedian: %+0.6f\tmax: %+0.6f' % (iteration, mean, median, _max)
            
            # TODO slow. test other implementation!
            borderline = np.partition(fitness, integer_threshold)[integer_threshold]
            fittest_pop = population[np.flatnonzero(fitness >= borderline)]  # TODO slow. test other implementation!
            
            gm.update(fittest_pop)
            
            to_replace = population[np.flatnonzero(fitness < borderline)]  # TODO slow. test other implementation!
            for ind in to_replace:
                ind.sample(gm)
            
            if self.early_stop(gm, diff):
                break
            
            fitness = np.array(map(lambda x: x.fitness, population))
            
            iteration += 1
        
        fittest_ind = population[np.argmax(fitness)]
        return fittest_ind
