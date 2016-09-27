# coding=utf-8

__author__ = 'Henry Cagnini'


class Session(dict):
    pass


class SetterClass(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @classmethod
    def set_class_values(cls, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(cls, k, v)


class AbstractTree(object):
    pred_attr = None
    target_attr = None
    class_labels = None
    
    def __init__(self, **kwargs):
        attrs = ['pred_attr', 'target_attr', 'class_labels']
        
        for k in attrs:
            if k in kwargs and getattr(self.__class__.__base__, k) is None:
                setattr(self.__class__.__base__, k, kwargs[k])
            else:
                setattr(self, k, getattr(self.__class__.__base__, k))

    def plot(self):
        pass


class AbstractEDA(AbstractTree):
    gm = None

    def __init__(self, n_individuals=100, n_iterations=100, uncertainty=0.01, decile=0.9, **kwargs):
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
        super(AbstractEDA, self).__init__(**kwargs)

        self.n_individuals = n_individuals
        self.n_iterations = n_iterations
        self.uncertainty = uncertainty
        self.decile = decile

    def early_stop(self, **kwargs):
        pass

    def fit_predict(self, verbose=True, **kwargs):
        pass

    def verbose(self, **kwargs):
        pass
