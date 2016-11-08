# coding=utf-8

import itertools as it
from collections import Counter

import pandas as pd

from treelib import node
from treelib.classes import SetterClass, AbstractTree
from treelib.node import *

__author__ = 'Henry Cagnini'


class Tensor(SetterClass):
    target_attr = None

    def __init__(
            self, name, values, parents=None, class_probability='declining',
            max_height=None, probability='uniform', **kwargs
    ):
        super(Tensor, self).__init__(**kwargs)

        self.name = name  # type: int
        self.values = values  # type: list of str
        self.parents = parents if (len(parents) > 0 or parents is not None) else []  # type: list of int

        self.weights = self.__init_probabilities__(
            values,
            probability,
            class_probability if len(self.values) > 1 else 'same',
            max_height=max_height
        )  # type: pd.DataFrame

    @property
    def n_parents(self):
        return len(self.parents)
    
    @property
    def n_values(self):
        return len(self.values)
    
    def __init_probabilities__(self, values, probability='uniform', class_probability='declining', max_height=None):
        if probability == 'uniform':
            vec_vals = [self.values] + [self.values for p in self.parents]
            
            combs = list(it.product(*vec_vals))
            columns = [self.name] + [p for p in self.parents]

            df = pd.DataFrame(
                data=combs,
                columns=columns
            )

            if class_probability == 'same':
                df['probability'] = 1. / df.shape[0]
            elif class_probability == 'decreased':
                _slice = df.loc[df[self.name] == self.target_attr]
                df['probability'] = 1. / (df.shape[0] - _slice.shape[0])
                df.loc[_slice.index, 'probability'] = 0.
            elif class_probability == 'declining':
                _slice = df.loc[df[self.name] == self.target_attr]
                if len(self.parents) > 0:
                    depth = get_depth(self.name)
                    class_prob = (1./(max_height - 1.)) * float(depth)
                    slice_prob = class_prob / _slice.shape[0]

                    pred_prob = (1. - class_prob) / (df.shape[0] - _slice.shape[0])

                    df['probability'] = pred_prob
                    df.loc[_slice.index, 'probability'] = slice_prob
                elif len(values) > 1:
                    _slice = df.loc[df[self.name] == self.target_attr]
                    df['probability'] = 1. / (df.shape[0] - _slice.shape[0])
                    df.loc[_slice.index, 'probability'] = 0.
                else:
                    df['probability'] = 1. / df.shape[0]

            else:
                raise ValueError('class probability must be either \'same\', \'decreased\' or \'declining\'!')

            return df
        elif isinstance(probability, list):
            if len(probability) != len(values):
                raise IndexError('number of weights must be the same as the number of values!')
            else:
                raise NotImplementedError('not implemented yet!')
        else:
            raise TypeError('probability must be either a string or a list!')
    
    def get_value(self, parent_labels=None):

        if len(self.parents) == 0:
            a, p = (self.weights[self.name], self.weights['probability'])
            value = np.random.choice(a=a, p=p)
        else:
            raise NotImplementedError('not implemented yet!')

            # print samples
            grouped = samples.groupby(by=self.parents, axis=0)
            groups = grouped.groups

            # iterates over values
            for group_name, group_index in groups.iteritems():
                if not isinstance(group_name, str):
                    try:
                        group_name = iter(group_name)
                    except TypeError, te:
                        group_name = [group_name]
                else:
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

                repeat = (2 ** get_depth(self.name))  # TODO see if this is correct!
                size = group_size * repeat

                # proper sample
                value = np.random.choice(a=a, p=p, replace=True, size=size)
                _index = index_at_level(get_depth(self.name))
                samples.loc[group_index, _index] = value.reshape(group_size, repeat)

        # ----------------------- #
        return value


class GraphicalModel(AbstractTree):
    """
        A graphical model is a tree itself.
    """
    
    tensors = None  # tensor is a dependency graph
    
    def __init__(
            self, max_height=3, distribution='multivariate', class_probability='declining', **kwargs
    ):
        super(GraphicalModel, self).__init__(**kwargs)

        self.tensors = self.__init_tensors__(max_height, distribution, class_probability)
    
    def __init_tensors__(self, max_height, distribution='uniform', class_probability='declining'):
        def get_parents(_id, _distribution):
            if _distribution == 'multivariate':
                parents = range(_id) if _id > 0 else []
            elif _distribution == 'bivariate':
                parents = [_id - 1] if _id > 0 else []
            elif _distribution == 'univariate':
                parents = []
            else:
                raise ValueError('Distribution must be either multivariate, bivariate or univariate!')
            return parents

        sample_values = self.pred_attr + [self.target_attr]

        tensors = map(
            lambda i: Tensor(
                i,
                parents=get_parents(i, distribution),
                values=sample_values,
                class_probability=class_probability,
                max_height=max_height,
                target_attr=self.target_attr
            ),
            xrange(max_height)
        )
        
        return tensors
    
    def update(self, fittest):
        """
        Updates graphical model.

        :type fittest: pandas.Series
        :param fittest:
        :return:
        """

        def get_node_label(_level):
            _index = index_at_level(_level)
            labels = []
            for _ind in _index:
                try:
                    labels += [fit.tree.node[_ind]['label'] if not fit.tree.node[_ind]['terminal'] else self.target_attr]
                except KeyError:
                    pass
            return labels

        pd.options.mode.chained_assignment = None
        
        n_fittest = float(len(fittest))

        for i, tensor in enumerate(self.tensors):
            parents = tensor.parents
            order = [tensor.name] + parents

            all_trees = []
            for fit in fittest:
                if i == 2:  # TODO remove me!
                    pass
                    # raise NotImplementedError('not implemented yet!')

                _children = get_node_label(tensor.name)
                _parents = map(lambda x: tuple(get_node_label(x)), parents)
                if len(parents) > 0:
                    trees = []
                    for c in _children:
                        sub = [c]
                        sub.extend(it.chain(*_parents))
                        trees.append(tuple(sub))
                else:
                    trees = _children
                all_trees.extend(trees)

            count = Counter(all_trees)
            weights = tensor.weights  # type: pd.DataFrame
            weights['probability'] = 0.

            for comb, n_occur in count.iteritems():
                if comb[0] is None:
                    continue  # does nothing

                click = it.izip(order, comb)
                _slice = weights  # type: pd.DataFrame
                for var_name, value in click:
                    _slice = weights.loc[weights[var_name] == value]
                
                _slice['probability'] = n_occur
                weights['probability'][_slice.index] = _slice['probability']

            weights['probability'] = weights['probability'].apply(lambda x: x / n_fittest)
            
            tensor.weights = weights

            # if, for some reason, this node hasn't received any value through
            # ALL fittest individual samplings, then assume an uniform distribution
            sum_prob = weights['probability'].sum()
            if sum_prob < 1.:
                rest = 1. - sum_prob
                to_add = [rest / weights.shape[0] for b in xrange(weights.shape[0])]
                further_add = 1. - (sum(to_add) + sum_prob)
                to_add[np.random.randint(weights.shape[0])] += further_add

                weights['probability'] = weights.apply(lambda x: x['probability'] + to_add[x.name], axis=1)

            self.tensors[i].weights = weights

        pd.options.mode.chained_assignment = 'warn'
    
    def sample(self, level, parent_labels=None, enforce_nonterminal=False):
        value = self.tensors[level].get_value(parent_labels=parent_labels)
        if enforce_nonterminal:
            while value == self.target_attr:
                value = self.tensors[level].get_value(parent_labels=parent_labels)

        return value

    @classmethod
    def reset_globals(cls):
        del Tensor.global_gms
        Tensor.global_gms = dict()
