from collections import Counter
from heap import Node

import numpy as np
import networkx as nx
import pandas as pd

__author__ = 'Henry Cagnini'


class GM(object):
    _pred_attr = None
    _target_attr = None
    _class_values = None
    
    def __init__(self, pred_attr, target_attr, class_values):
        if any(map(lambda x: x is None, [self._pred_attr, self._target_attr])):
            GM._pred_attr = pred_attr.values.tolist()
            GM._target_attr = target_attr
            GM._class_values = class_values
        
        self._pred_attr = GM._pred_attr
        self._target_attr = GM._target_attr
        self._class_values = GM._class_values
    
    def sample(self, id_node, id_parent):
        pass


class StartGM(GM):
    """
    Initial PMF for generating diverse individuals.
    """
    
    def __init__(self, pred_attr, target_attr, class_values, target_add):
        super(StartGM, self).__init__(pred_attr, target_attr, class_values)
        self._target_add = target_add  # type: float
    
    def sample(self, id_node, id_parent):
        depth = Node.get_depth(id_node)
        
        target_prob = np.clip(depth * self._target_add, a_min=0., a_max=1.)  # type: float
        pred_prob = [(1. - target_prob) / len(self._pred_attr) for x in xrange(len(self._pred_attr))]  # type: list
        a = self._pred_attr + [self._target_attr]  # type: list
        p = pred_prob + [target_prob]  # type: list
        
        chosen = np.random.choice(a=a, p=p)
        return chosen


class FinalGM(GM):
    def __init__(self, pred_attr, target_attr, class_values, population):
        super(FinalGM, self).__init__(pred_attr, target_attr, class_values)
        self._graph = None

        # maximum depth tree
        self._max_node = max(
            map(
                lambda x:
                max(x),
                map(
                    lambda x: x.node.keys(),
                    map(lambda x: x.tree, population)
                )
            )
        )
        
    def update(self, fittest_pop):
        # TODO parents in the columns, children on the rows!
        
        # picks each individual tree
        trees = map(lambda x: x.tree, fittest_pop)
        
        inner = nx.DiGraph()
        
        def pick_label(tree, i, parent):
            try:
                node_label = tree.node[i]['label']
                if node_label in self._class_values:  # changes name of class for name of class attribute
                    node_label = self._target_attr
                if parent is not None:  # if this is not the root node
                    parent_node = tree.node[parent]['label']
                    return parent_node, node_label
                return None, node_label
            except KeyError:
                pass
        
        for i in xrange(self._max_node + 1):
            parent = Node.get_parent(i)
            node_labels = map(lambda t: pick_label(t, i, parent), trees)
            node_labels = [node for node in node_labels if node is not None]  # remove Nones from the list
            
            if all(map(lambda x: x is None, node_labels)):  # if node is not present in none of the provided trees
                continue  # pass on

            parents, nodes = zip(*node_labels)

            prob_matrix = pd.DataFrame(
                data=np.zeros(shape=(len(set(nodes)), len(set(parents))), dtype=np.float32),
                index=set(nodes),
                columns=set(parents)
            )
            # TODO wrong!!! (in some sense)
            for (p, c) in node_labels:
                prob_matrix[p][c] += 1
            
            count_sum = float(len(node_labels))
            # else:
            #     # matrix with len(set(node_labels)) columns and 1 row
            #     count = Counter(node_labels)
            #     count_sum = float(max(sum(count.itervalues()), 1.))  # prevents zero division
            #
            #     prob_matrix = pd.DataFrame(
            #         data=np.zeros(shape=(1, len(count)), dtype=np.float32),
            #         columns=count.keys()
            #     )
            #     prob_matrix[count.keys()] += count.values()
                
            prob_matrix = prob_matrix.applymap(lambda x: x / count_sum)
            inner.add_node(n=i, attr_dict={'probs': prob_matrix})
            
            if parent is not None:
                inner.add_edge(parent, i)

        self._graph = inner
    
    def sample(self, id_node, id_parent):
        prob_matrix = self._graph.node[id_node]['probs']

        a = prob_matrix.index
        p = prob_matrix[id_parent]/prob_matrix[id_parent].sum()

        # try:
        # must build a conditional probability from the marginals; thus, sums the probability
        chosen = np.random.choice(a=a, p=p)
        # except ValueError:
        #     chosen = None  # when the node doesn't have any probability attached to it
        return chosen
    
    @property
    def graph(self):
        return self._graph
