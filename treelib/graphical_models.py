from treelib.utils import SetterClass, Session
import pandas as pd
import numpy as np

__author__ = 'Henry Cagnini'


class Tensor(SetterClass):
    global_gms = dict()
    
    def __init__(self, name, values, parents=None, probability='uniform', gm_id=0, **kwargs):
        super(Tensor, self).__init__(**kwargs)
        
        if gm_id not in self.__class__.global_gms:
            self.__class__.global_gms[gm_id] = dict()
        
        if name in self.__class__.global_gms[gm_id]:
            raise KeyError('Variable is already defined for this scope!')  # name is also an axis
                
        self.gm_id = gm_id
        self.name = name
        self.parents = parents if parents is not None else []
        self.values = values
        self.reverse_values = {k: i for i, k in enumerate(values)}
        
        self.weights = self.__init_probabilities__(values, probability)

        self.__class__.global_gms[gm_id][name] = self
        
    def __init_probabilities__(self, values, probability='uniform'):
        n_values = len(values)
        n_parents = len(self.parents)
        
        if probability == 'uniform':
            # TODO tuple with self.parents or 1 positions and values values for each!
            shape = tuple([n_values for i in xrange(n_parents + 1)])
            weights = np.zeros(shape=shape) + 1./n_values
            return weights
        elif isinstance(probability, list):
            if len(probability) != len(values):
                raise IndexError('number of weights must be the same as the number of values!')
            else:
                raise NotImplementedError('not implemented yet!')  # TODO guarantee that weights have the same order than values!
        else:
            raise TypeError('probability must be either a string or a list!')
            
    def sample(self, session):
        if self.name in session:
            raise KeyError('value already sampled in this session!')
        if len(self.parents) == 0:
            p = self.weights
        else:
            axis = [str(self.reverse_values[session[p]]) for p in self.parents]
            coords = ','.join(axis)
            p = eval('self.weights[%s]' % coords)
        
        value = np.random.choice(a=self.values, p=p)  # weights has the same order than values
        session[self.name] = value
        return value
            

class AbstractGraphicalModel(SetterClass):
    pred_attr = None
    target_attr = None
    class_labels = None
    
    def __init__(self, **kwargs):
        super(AbstractGraphicalModel, self).__init__(**kwargs)

    def plot(self):
        pass


class GraphicalModel(AbstractGraphicalModel):
    """
        A graphical model is a tree itself.
    """

    tensors = None  # tensor is a dependency graph

    def __init__(self, pattern=None, gm_id=0, **kwargs):
        super(GraphicalModel, self).__init__(**kwargs)
        
        self.gm_id = gm_id
        
        if pattern is None:
            self.tensors = self.__init_tensor__()
        else:
            raise NotImplementedError('not implemented yet!')
        
    def __init_tensor__(self):
        """
        Initializes a simple 3 nodes tree.
        
        :rtype: list
        :return: A list of 3 tensors.
        """
        values = self.pred_attr + [self.target_attr]
        
        tensors = [
            Tensor(0, values, gm_id=self.gm_id),
            Tensor(1, values, [0], gm_id=self.gm_id),
            Tensor(2, values, [0, 1], gm_id=self.gm_id)
        ]
        
        return tensors

    def sample(self):
        sess = Session()
        
        for tensor in self.tensors:
            tensor.sample(sess)
        
        # TODO must suffer possibility to mutate!
        
        raise NotImplementedError('not implemented yet!')
