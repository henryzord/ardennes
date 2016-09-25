from evolution.graphical_models import *
from evolution.trees import Individual
from treelib.eda import AbstractEDA

__author__ = 'Henry Cagnini'


class Ardennes(AbstractEDA):
    pred_attr = None
    target_attr = None
    class_labels = None

    def fit_predict(self, verbose=True, **kwargs):
        if 'sets' not in kwargs or not all(map(lambda x: x in kwargs['sets'], ['train', 'val'])):  # TODO optimize!
            raise KeyError('You need to pass train and val sets to this method!')
        else:
            sets = kwargs['sets']

        class_values = {
            'pred_attr': sets['train'].columns[:-1],
            'target_attr': sets['train'].columns[-1],
            'class_labels': sets['train'][sets['train'].columns[-1]].unique()
        }

        self.__class__.set_values(**class_values)
        gm = GraphicalModel(**class_values)

        population = self.init_population(
            n_individuals=self.n_individuals,
            gm=gm,
            sets=sets
        )

        raise NotImplementedError('implement!')

        # TODO update!
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

    def init_population(self, n_individuals, gm, sets):
        vfunc = np.vectorize(Individual)

        individuals = vfunc()

        population = pd.DataFrame(individuals, columns=['individual'], dtype=np.object)

        z = 0
    
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