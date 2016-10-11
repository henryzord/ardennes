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
    
    def __init__(self, name, values, parents=None, class_probability='same', probability='uniform', gm_id=0, **kwargs):
        super(Tensor, self).__init__(**kwargs)
        
        if gm_id not in self.__class__.global_gms:
            self.__class__.global_gms[gm_id] = dict()
        
        if name in self.__class__.global_gms[gm_id]:
            raise KeyError('Variable is already defined for this scope!')  # name is also an axis
        
        self.gm_id = gm_id  # type: float
        self.name = name  # type: int
        self.values = values  # type: list of str
        self.parents = parents if (len(parents) > 0 or parents is not None) else []  # type: list of int

        self.weights = self.__init_probabilities__(
            values,
            probability,
            class_probability if len(self.values) > 1 else 'same'
        )  # type: pd.DataFrame
        
        self.__class__.global_gms[gm_id][name] = self
    
    @property
    def n_parents(self):
        return len(self.parents)
    
    @property
    def n_values(self):
        return len(self.values)
    
    def __init_probabilities__(self, values, probability='uniform', class_probability='same'):
        if probability == 'uniform':
            vec_vals = [self.values] + [self.global_gms[self.gm_id][p].values for p in self.parents]
            
            combs = list(it.product(*vec_vals))
            columns = [self.name] + [self.global_gms[self.gm_id][p].name for p in self.parents]
            
            df = pd.DataFrame(
                data=combs,
                columns=columns
            )

            if class_probability == 'same':
                df['probability'] = 1. / df.shape[0]
            elif class_probability == 'decreased':
                _slice = df.loc[df[self.name] == 'class']
                df['probability'] = 1. / (df.shape[0] - _slice.shape[0])
                df.loc[_slice.index, 'probability'] = 0.
            else:
                raise ValueError('class probability must be either \'same\' or \'decreased\'!')

            return df
        elif isinstance(probability, list):
            if len(probability) != len(values):
                raise IndexError('number of weights must be the same as the number of values!')
            else:
                # TODO guarantee that weights have the same order than values!
                raise NotImplementedError('not implemented yet!')
        else:
            raise TypeError('probability must be either a string or a list!')
    
    def get_value(self, samples):
        """

        :type samples: pandas.DataFrame
        :param samples:
        :return:
        """
        if len(self.values) == 1:
            samples[self.name] = self.values[0]

        elif len(self.parents) == 0:
            a, p = (self.weights[self.name], self.weights['probability'])

            values = np.random.choice(a=a, p=p, replace=True, size=samples.shape[0])
            samples[self.name] = values
        else:
            # print samples
            grouped = samples.groupby(by=self.parents, axis=0)
            groups = grouped.groups

            # iterates over values
            for group_name, group_index in groups.iteritems():
                if isinstance(group_name, str):
                    group_name = [group_name]

                group_size = group_index.shape[0]

                # localize probabilities accordingly to parent values
                df = self.weights.copy()  # type: pd.DataFrame
                for i_p, p in it.izip(self.parents, group_name):
                    df = df.loc[df[i_p] == p]

                # update probabilities
                _sum = df['probability'].sum()
                df['probability'] = df['probability'].apply(lambda x: x / _sum)
                a, p = (df[self.name], df['probability'])

                # proper sample
                values = np.random.choice(a=a, p=p, replace=True, size=group_size)

                samples.loc[group_index, self.name] = values

        # ----------------------- #
        return samples


class GraphicalModel(AbstractTree):
    """
        A graphical model is a tree itself.
    """
    
    tensors = None  # tensor is a dependency graph
    
    def __init__(self, gm_id=0, initial_tree_size=3, distribution='multivariate', class_probability='uniform', **kwargs):
        super(GraphicalModel, self).__init__(**kwargs)
        
        self.gm_id = gm_id
        self.tensors = self.__init_tensor__(initial_tree_size, distribution, class_probability)
    
    def __init_tensor__(self, initial_tree_size, distribution, class_probability):
        # TODO enhance to perform any kind of initialization!
        # TODO must be able to automatically perform this kind of initialization!

        def get_parents(_id, _distribution):
            if _distribution == 'multivariate':
                parent = node.get_parent(_id)
                val = parent
                parents = []
                while val is not None:
                    parents += [parent]
                    parent = node.get_parent(val)
                    val = parent
            elif _distribution == 'bivariate':
                parents = [node.get_parent(_id)]
            elif _distribution == 'univariate':
                parents = []
            else:
                raise ValueError('Distribution must be either multivariate, bivariate or univariate!')
            return parents

        def is_terminal(_id):
            return node.get_right_child(_id) >= initial_tree_size
        
        inner_values = self.pred_attr + [self.target_attr]
        outer_values = [self.target_attr]
        
        tensors = map(
            lambda i: Tensor(
                i,
                parents=get_parents(i, distribution),
                values=inner_values if not is_terminal(i) else outer_values,
                class_probability=class_probability,
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
    
    def sample(self, df):
        for tensor in self.tensors:
            df = tensor.get_value(df)

        raise NotImplementedError('df must be a pandas.dataframe with size n_individuals, n_variables!')
        return df

    @classmethod
    def reset_globals(cls):
        del Tensor.global_gms
        Tensor.global_gms = dict()
