from matplotlib import pyplot
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
    
    def sample(self, id_node, parent_label):
        pass
    
    def plot(self):
        pass


class StartGM(GM):
    """
    Initial PMF for generating diverse individuals.
    """
    
    max_depth = 0
    
    def __init__(self, pred_attr, target_attr, class_values, max_height):
        super(StartGM, self).__init__(pred_attr, target_attr, class_values)

        self._target_add = 1. / max_height  # type: float
    
    def sample(self, id_node, parent_label):
        depth = Node.get_depth(id_node)
        
        StartGM.max_depth = max(depth, StartGM.max_depth)
            
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
    
    def sample(self, id_node, parent_label):
        prob_matrix = self._graph.node[id_node]['probs']

        a = prob_matrix.index
        if parent_label is not None:
            p = prob_matrix[parent_label] / prob_matrix[parent_label].sum()
        else:
            p = prob_matrix[None]
        # try:
        # must build a conditional probability from the marginals; thus, sums the probability
        chosen = np.random.choice(a=a, p=p)
        # except ValueError:
        #     chosen = None  # when the node doesn't have any probability attached to it
        return chosen
    
    @property
    def graph(self):
        return self._graph

    def plot(self):
        from matplotlib import pyplot as plt
    
        fig = plt.figure()
    
        tree = self.graph   # type: nx.DiGraph
        pos = nx.spectral_layout(tree)
    
        node_list = tree.nodes()
        edge_list = tree.edges()
    
        nx.draw_networkx_nodes(tree, pos, node_size=1000)  # nodes
        nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
        # nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
        # nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)
    
        plt.axis('off')
