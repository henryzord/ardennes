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
        
        if 'initial_tree_size' in kwargs:
            initial_tree_size = kwargs['initial_tree_size']
        else:
            initial_tree_size = 3
        
        gm = GraphicalModel(initial_tree_size=initial_tree_size, **class_values)
        
        population = self.sample_individuals(
            n_sample=self.n_individuals,
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
            
            borderline = np.partition(fitness, integer_threshold)[
                integer_threshold]  # TODO slow. test other implementation!
            
            # picks fittest population
            fittest_pop = self.pick_fittest_population(population, borderline)
            gm.update(fittest_pop)
            
            n_replace = np.count_nonzero(fitness < borderline)
            replaced = self.sample_individuals(n_replace, gm, sets)
            population = fittest_pop + replaced
            
            if self.early_stop(gm, self.uncertainty):
                break
            
            fitness = np.array(map(lambda x: x.fitness, population))
            
            iteration += 1
        
        fittest_ind = population[np.argmax(fitness)]
        
        GraphicalModel.reset_globals()
        
        return fittest_ind
    
    @staticmethod
    def pick_fittest_population(population, borderline):
        fittest_pop = []
        for ind in population:
            if ind.fitness >= borderline:
                fittest_pop += [ind]
        return fittest_pop
    
    @staticmethod
    def sample_individuals(n_sample, graphical_model, sets):
        sample = map(
            lambda i: Individual(id=i, graphical_model=graphical_model, sets=sets),
            xrange(n_sample)
        )
        return sample
    
    def verbose(self, **kwargs):
        iteration = kwargs['iteration']
        mean = kwargs['mean']
        median = kwargs['median']
        max_fitness = kwargs['max_fitness']
        
        if kwargs['verbose']:
            print 'iter: %03.d\tmean: %+0.6f\tmedian: %+0.6f\tmax: %+0.6f' % (iteration, mean, median, max_fitness)
    
    @staticmethod
    def early_stop(gm, uncertainty=0.01):
        """

        :type gm: FinalAbstractGraphicalModel
        :param gm: The Probabilistic Graphical Model (GM) for the current generation.
        :type uncertainty: float
        :param uncertainty: Maximum allowed uncertainty for each probability, for each node.
        :return:
        """
        import warnings
        warnings.warn('WARNING: implement!')
        return False
