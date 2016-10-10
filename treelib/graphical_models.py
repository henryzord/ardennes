# coding=utf-8

import itertools as it
from collections import Counter

import numpy as np
import pandas as pd

from treelib.classes import SetterClass, AbstractTree
from treelib.individual import Individual
from treelib import node

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
        self.values = values  # type: list of str
        self.parents = parents if (len(parents) > 0 or parents is not None) else []  # type: list of int
        self.weights = self.__init_probabilities__(values, probability)  # type: pd.DataFrame
        
        self.__class__.global_gms[gm_id][name] = self
    
    @property
    def n_parents(self):
        return len(self.parents)
    
    @property
    def n_values(self):
        return len(self.values)
    
    def __init_probabilities__(self, values, probability='uniform'):
        if probability == 'uniform':
            vec_vals = [self.values] + [self.global_gms[self.gm_id][p].values for p in self.parents]
            
            combs = list(it.product(*vec_vals))
            columns = [self.name] + [self.global_gms[self.gm_id][p].name for p in self.parents]
            
            df = pd.DataFrame(
                data=combs,
                columns=columns
            )
            df['probability'] = 1. / df.shape[0]
            return df
        elif isinstance(probability, list):
            if len(probability) != len(values):
                raise IndexError('number of weights must be the same as the number of values!')
            else:
                # TODO guarantee that weights have the same order than values!
                raise NotImplementedError('not implemented yet!')
        else:
            raise TypeError('probability must be either a string or a list!')
    
    def sample(self, sessions):
        if isinstance(sessions, Session):
            sessions = [sessions]

        if len(self.parents) == 0:
            p = self.weights['probability']
            a = self.weights[self.name]

            vals = np.random.choice(a=a, p=p, replace=True, size=len(sessions))
        else:
            # TODO group session values!!!
            tuples = [tuple([sess[p] for p in self.parents]) for sess in sessions]
            parent_sets = Counter(tuples)

            for parent_set, count_set in parent_sets.iteritems():
                df = self.weights.copy()  # type: pd.DataFrame
                for i, p in enumerate(self.parents):
                    df = df.loc[df[p] == parent_set[i]]  # must not come from session; instead from parent_sets!

                _sum = df['probability'].sum()

                df['probability'] = df['probability'].apply(lambda x: x / _sum)

                a = df[self.name]
                p = df['probability']

                vals = np.random.choice(a=a, p=p, replace=True, size=count_set)

                # TODO must insert children in the correct order now!
                # TODO must insert children in the correct order now!
                # TODO must insert children in the correct order now!

        # ----------------------- #

        for session, val in it.izip(sessions, vals):
            if self.name in session:
                raise KeyError('value already sampled in this session!')

            session[self.name] = val

        raise NotImplementedError('not implemented yet!')
        return sessions


class GraphicalModel(AbstractTree):
    """
        A graphical model is a tree itself.
    """
    
    tensors = None  # tensor is a dependency graph
    
    def __init__(self, gm_id=0, initial_tree_size=3, distribution='multivariate', class_probability=None, **kwargs):
        super(GraphicalModel, self).__init__(**kwargs)
        
        self.gm_id = gm_id
        self.tensors = self.__init_tensor__(initial_tree_size, distribution, class_probability)
    
    def __init_tensor__(self, initial_tree_size, distribution, class_probability):
        # TODO enhance to perform any kind of initialization!
        # TODO must be able to automatically perform this kind of initialization!

        def get_parents(id, distribution):
            if distribution == 'multivariate':
                parent = node.get_parent(id)
                val = parent
                parents = []
                while val is not None:
                    parents += [parent]
                    parent = node.get_parent(val)
                    val = parent
            elif distribution == 'bivariate':
                parents = [node.get_parent(id)]
            elif distribution == 'univariate':
                parents = []
            else:
                raise ValueError('Distribution must be either multivariate, bivariate or univariate!')
            return parents

        def is_terminal(id):
            return node.get_right_child(id) >= initial_tree_size
        
        inner_values = self.pred_attr + [self.target_attr]
        outer_values = [self.target_attr]
        
        tensors = map(
            lambda i: Tensor(
                i,
                parents=get_parents(i, distribution),
                values=inner_values if not is_terminal(i) else outer_values,
                gm_id=self.gm_id
            ),
            xrange(initial_tree_size)
        )
        
        return tensors
    
    def update(self, fittest):
        """
        
        :type fittest: list of Individual
        :param fittest:
        :return:
        """

        pd.options.mode.chained_assignment = None
        
        n_fittest = float(len(fittest))

        for i, tensor in enumerate(self.tensors):
            parents = tensor.parents
            order = [tensor.name] + parents
        
            all_vec = []
            for fit in fittest:
                vec = map(
                    lambda x: fit.tree.node[x]['label'] if not fit.tree.node[x]['terminal'] else self.target_attr,
                    order
                )
                all_vec += [tuple(vec)]

            count = Counter(all_vec)
            weights = tensor.weights  # type: pd.DataFrame
            weights['probability'] = 0.
            
            for comb, n_occur in count.iteritems():
                click = it.izip(order, comb)
                _slice = weights  # type: pd.DataFrame
                for var_name, value in click:
                    _slice = weights.loc[weights[var_name] == value]
                
                _slice['probability'] = n_occur
                weights['probability'][_slice.index] = _slice['probability']  # TODO not assigning correctly!

            weights['probability'] = weights['probability'].apply(lambda x: x / n_fittest)
            
            tensor.weights = weights
            self.tensors[i].weights = weights

        pd.options.mode.chained_assignment = 'warn'
    
    def sample(self, n_sample=1):
        raise NotImplementedError('sunset_sessions must be a pandas.dataframe with size n_individuals, n_variables!')

        sunset_sessions = [Session() for i in xrange(n_sample)]

        for tensor in self.tensors:
            tensor.sample(sessions=sunset_sessions)

        return sunset_sessions

    @classmethod
    def reset_globals(cls):
        del Tensor.global_gms
        Tensor.global_gms = dict()
