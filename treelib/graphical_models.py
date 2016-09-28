from collections import Counter

import numpy as np
import xarray

from treelib.classes import SetterClass, Session, AbstractTree
from treelib.individual import Individual
import itertools as it
import operator as op
import pandas as pd

__author__ = 'Henry Cagnini'


class Tensor(SetterClass):
    global_gms = dict()
    
    def __init__(self, name, values, parents=None, probability='uniform', gm_id=0, **kwargs):
        super(Tensor, self).__init__(**kwargs)
        
        if gm_id not in self.__class__.global_gms:
            self.__class__.global_gms[gm_id] = dict()
        
        if name in self.__class__.global_gms[gm_id]:
            raise KeyError('Variable is already defined for this scope!')  # name is also an axis
                
        self.gm_id = gm_id  # type: float
        self.name = name  # type: int
        self.parents = parents if parents is not None else []
        self.values = values  # type: list of str
        
        # self.reverse_values = {k: i for i, k in enumerate(values)}  # type: dict of str
        
        self.weights = self.__init_probabilities__(values, probability)  # type: np.ndarray

        self.__class__.global_gms[gm_id][name] = self
            
    @property
    def n_values(self):
        return len(self.values)
        
    def __init_probabilities__(self, values, probability='uniform'):
        if probability == 'uniform':
            shape = [self.n_values] + [self.global_gms[self.gm_id][p].n_values for p in self.parents]
            if len(shape) == 0:
                shape = [self.n_values]
            
            n_lines = reduce(op.mul, shape)
            
            # TODO error here! in the dimensions!
            raise IndexError('error here!')
            
            df = pd.DataFrame(
                data=np.zeros(shape=(n_lines, max(len(self.parents), 1)), dtype=np.float32) + 1./n_lines,
                index=np.repeat(self.values, max(len(self.parents), 1)),  # TODO possible error here!
                columns=[self.name] + self.parents
            )
            return df
        elif isinstance(probability, list):
            if len(probability) != len(values):
                raise IndexError('number of weights must be the same as the number of values!')
            else:
                # TODO guarantee that weights have the same order than values!
                raise NotImplementedError('not implemented yet!')
        else:
            raise TypeError('probability must be either a string or a list!')
            
    def sample(self, session):
        if self.name in session:
            raise KeyError('value already sampled in this session!')
        if len(self.parents) == 0:
            p = self.weights
        else:
            axis = [
                str(self.global_gms[self.gm_id][p].reverse_values[session[p]]) for p in self.parents
                ]
            coords = ','.join(axis)
            p = eval('self.weights[%s]' % coords)
            p = [p] if not isinstance(p, list) else p

        value = np.random.choice(a=self.values, p=p)  # weights has the same order than values
        session[self.name] = value
        return value


class GraphicalModel(AbstractTree):
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
        # TODO enhance to perform any kind of initialization!
        
        tensors = [
            Tensor(0, self.pred_attr, gm_id=self.gm_id),
            Tensor(1, [self.target_attr], [0], gm_id=self.gm_id),
            Tensor(2, [self.target_attr], [0], gm_id=self.gm_id)
        ]
        
        return tensors

    def update(self, fittest):
        """
        
        :type fittest: list of Individual
        :param fittest:
        :return:
        """
        
        to_update = [0]
        for current_node in to_update:
            tensor = self.tensors[current_node]
            
            to_update.remove(current_node)
            to_update += fittest[0].tree.successors(current_node)
            
            current_parents = tensor.parents
            
            married_values = []
            for fit in fittest:
                married_values += [[fit.tree.node[x]['label'] for x in current_parents + [current_node]]]  # TODO must now pick reversed values!!!
            z = 0
        
        # to_update = [0]
        # for current_node in to_update:  # current_node will update a tensor on its own
        #     to_update.remove(current_node)
        #     successors = fittest[0].tree.successors(current_node)
        #     to_update.extend(successors)
        #
        #     labels = map(lambda x: x.tree.node[current_node]['label'], fittest)
        #     count = Counter(labels)
        #     _sum = sum(count.values())
        #
        #     tensor = self.tensors[current_node]
        #     print tensor.weights
        #     if len(tensor.parents) > 0:
        #         parents = tensor.parents
        #         self_values = ([tensor.values] if isinstance(tensor.values, list) else [[tensor.values]])
        #         all_vals = [self.tensors[parent].values for parent in parents] + self_values
        #         all_combinations = list(it.product(*all_vals))
        #
        #         tensor.weights[]
        #
        #         z = 0
        #         # TODO update accordingly to the parents!!!
        #     else:
        #         for k, v in count.iteritems():
        #             tensor.weights[tensor.reverse_values[k]] = v / float(_sum)  # TODO must pick parents for other nodes!!!
        #             self.tensors[current_node] = tensor
        #         print tensor.weights
        #         z = 0
                
        raise NotImplementedError('TODO implement!')
        # TODO must suffer possibility to mutate!
        pass

    def sample(self):
        sess = Session()
        
        for tensor in self.tensors:
            tensor.sample(sess)
        
        return sess
