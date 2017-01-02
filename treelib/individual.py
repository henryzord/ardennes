# coding=utf-8

import collections
import copy
import itertools as it
from collections import Counter

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import *

from treelib.node import *

__author__ = 'Henry Cagnini'


class Individual(object):
    _terminal_node_color = '#98FB98'
    _inner_node_color = '#0099ff'
    _root_node_color = '#FFFFFF'
    column_types = None  # type: dict

    target_attr = None
    pred_attr = None
    class_labels = None
    n_objects = None
    n_attributes = None

    handler = None

    ind_id = None
    fitness = None  # type: float
    height = None
    max_height = -1
    tree = None  # type: nx.DiGraph

    thresholds = dict()  # thresholds for nodes

    _shortest_path = dict()  # type: dict

    full = None
    sets = None
    arg_sets = None
    y_test_true = None
    y_val_true = None

    rtol = 1e-3

    def __init__(self, gm, **kwargs):
        """
        
        :type gm: treelib.graphical_model.GraphicalModel
        :param gm:
        :type sets: dict
        :param sets:
        """
        if 'ind_id' in kwargs:
            self.ind_id = kwargs['ind_id']
        else:
            self.ind_id = None

        self.sample(gm, Individual.arg_sets['train_index'])

    @classmethod
    def set_values(cls, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(cls, k, v)

        Individual.column_types['class'] = 'class'

    @classmethod
    def clean(cls):
        cls.pred_attr = None
        cls.target_attr = None
        cls.class_labels = None
        cls.column_types = None

    def nodes_at_depth(self, depth):
        """
        Picks all nodes which are in the given level.

        :type depth: int
        :param depth: The level to pick
        :rtype: list of dict
        :return: A list of the nodes at the given level.
        """
        depths = {k: self.depth_of(k) for k in self._shortest_path.iterkeys()}
        at_level = []
        for k, d in depths.iteritems():
            if d == depth:
                at_level.append(self.tree.node[k])
        return at_level

    def parents_of(self, node_id):
        """
        The parents of the given node.

        :type node_id: int
        :param node_id: The id of the node, starting from zero (root).
        :rtype: list of int
        :return: A list of parents of this node, excluding the node itself.
        """
        parents = copy.deepcopy(self._shortest_path[node_id])
        parents.remove(node_id)
        return parents

    def height_and_label_to(self, node_id):
        """
        Returns a dictionary where the keys are the depth of each one
        of the parents, and the values the label of the parents.

        :param node_id: ID of the node in the decision tree.
        :return:
        """
        parents = self.parents_of(node_id)
        parent_labels = {
            self.tree.node[p]['level']: (
                self.tree.node[p]['label'] if
                self.tree.node[p]['label'] not in self.class_labels else
                self.target_attr
            ) for p in parents
            }
        return parent_labels

    def depth_of(self, node_id):
        """
        The depth which a node lies in the tree.

        :type node_id: int
        :param node_id: The id of the node, starting from zero (root).
        :rtype: int
        :return: Depth of the node, starting with zero (root).
        """

        return len(self._shortest_path[node_id]) - 1

    # ############################ #
    # sampling and related methods #
    # ############################ #

    def sample(self, gm, arg_train):
        self.tree = self.tree = self.__set_node__(
            node_id=0,
            gm=gm,
            tree=nx.DiGraph(),
            subset_index=arg_train,
            level=0,
            parent_labels=[],
            coordinates=[],
        )  # type: nx.DiGraph

        self._shortest_path = nx.shortest_path(self.tree, source=0)  # source equals to root

        y_pred = self.predict(Individual.sets['val'])
        acc_score = accuracy_score(Individual.y_val_true, y_pred)

        self.fitness = acc_score
        self.height = max(map(len, self._shortest_path.itervalues()))

    def __set_node__(self, node_id, gm, tree, subset_index, level, parent_labels, coordinates):
        try:
            label = gm.sample(
                node_id=node_id, level=level, parent_labels=parent_labels, enforce_nonterminal=(level == 0)
            )
        except KeyError as ke:
            if level >= self.max_height:
                label = self.target_attr
            else:
                raise ke

        if any((
                Individual.full.loc[subset_index, self.target_attr].unique().shape[0] == 1,  # only one class
                level >= self.max_height,  # level deeper than maximum depth
                label == self.target_attr   # was sampled to be a class
        )):
            meta, subsets = self.__set_terminal__(
                node_label=None,
                node_id=node_id,
                node_level=level,
                subset_index=subset_index,
                parent_labels=parent_labels,
                coordinates=coordinates
            )
        else:
            meta, subsets = self.__set_inner_node__(
                node_label=label,
                parent_labels=parent_labels,
                coordinates=coordinates,
                node_level=level,
                gm=gm,
                subset_index=subset_index,
                node_id=node_id
            )

            if meta['threshold'] is not None:
                children_id = [get_left_child(node_id), get_right_child(node_id)]

                for c, child_id, child_subset in it.izip(range(len(children_id)), children_id, subsets):
                    tree = self.__set_node__(
                        node_id=child_id,
                        tree=tree,
                        gm=gm,
                        subset_index=child_subset,
                        level=level + 1,
                        parent_labels=parent_labels + [label],
                        coordinates=coordinates + [c]
                    )

                if all([tree.node[child_id]['label'] in self.class_labels for child_id in children_id]) \
                        and tree.node[children_id[0]]['label'] == tree.node[children_id[1]]['label']:
                    for child_id in children_id:
                        tree.remove_node(child_id)

                    meta, subsets = self.__set_terminal__(
                        node_label=None,
                        node_id=node_id,
                        node_level=level,
                        subset_index=subset_index,
                        parent_labels=parent_labels,
                        coordinates=coordinates
                    )

                else:
                    if isinstance(meta['threshold'], float):
                        attr_dicts = [
                            {'threshold': '<= %0.2f' % meta['threshold']},
                            {'threshold': '> %0.2f' % meta['threshold']}
                        ]
                    elif isinstance(meta['threshold'], collections.Iterable):
                        attr_dicts = [{'threshold': t} for t in meta['threshold']]
                    else:
                        raise TypeError('invalid type for threshold!')

                    for child_id, attr_dict in it.izip(children_id, attr_dicts):
                        tree.add_edge(node_id, child_id, attr_dict=attr_dict)

        tree.add_node(node_id, attr_dict=meta)
        return tree

    def __predict_object__(self, obj):
        arg_node = 0  # always start with root

        tree = self.tree  # type: nx.DiGraph

        node = tree.node[arg_node]

        while not node['terminal']:
            go_left = obj[node['label']] <= node['threshold']
            successors = tree.successors(arg_node)
            arg_node = (int(go_left) * min(successors)) + (int(not go_left) * max(successors))

            node = tree.node[arg_node]

        return node['label']


    def predict(self, samples):
        """
        Makes predictions for unseen samples.

        :param samples: Either a pandas.DataFrame (for multiple samples)or pandas.Series (for a single object).
        :rtype: numpy.ndarray
        :return: The predicted class for each sample.
        """
        if isinstance(samples, pd.DataFrame):
            preds = samples.apply(self.__predict_object__, axis=1).as_matrix()
        elif isinstance(samples, pd.Series):
            preds = self.__predict_object__(samples)
        else:
            raise TypeError('Invalid type for this method! Must be either a pandas.DataFrame or pandas.Series!')
        return preds

    def __set_inner_node__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        attr_type = Individual.column_types[node_label]

        out = self.handler_dict[attr_type](
            self,
            node_label=node_label,
            node_id=node_id,
            node_level=node_level,
            subset_index=subset_index,
            parent_labels=parent_labels,
            coordinates=coordinates,
            **kwargs
        )
        return out

    def __store_threshold__(self, node_label, parent_labels, coordinates, threshold):
        key = '[' + ','.join(parent_labels + [node_label]) + '][' + ','.join([str(c) for c in coordinates]) + ']'
        self.__class__.thresholds[key] = threshold

    def __retrieve_threshold__(self, node_label, parent_labels, coordinates):
        key = '[' + ','.join(parent_labels + [node_label]) + '][' + ','.join([str(c) for c in coordinates]) + ']'
        return self.__class__.thresholds[key]

    def __set_numerical__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        try:
            best_threshold = self.__retrieve_threshold__(node_label, parent_labels, coordinates)
            meta, subsets = self.__subsets_and_meta__(
                node_label=node_label,
                node_id=node_id,
                node_level=node_level,
                subset_index=subset_index,
                threshold=best_threshold,
            )
        except KeyError as ke:
            unique_vals = sorted(Individual.full.loc[subset_index, node_label])

            if self.handler.max_n_candidates is None:
                candidates = np.array(unique_vals + [
                    (unique_vals[i] + unique_vals[i + 1]) / 2.
                    if (i + 1) < len(unique_vals) else unique_vals[i] for i in xrange(len(unique_vals))
                ][:-1])
            else:
                candidates = np.linspace(unique_vals[0], unique_vals[-1], self.handler.max_n_candidates)

            gains = self.handler.get_ratios(subset_index, node_label, candidates)

            argmax = np.argmax(gains)
            if gains[argmax] <= 0:
                meta, subsets = self.__set_terminal__(
                    node_label=node_label,
                    node_id=node_id,
                    node_level=node_level,
                    subset_index=subset_index,
                    parent_labels=parent_labels,
                    coordinates=coordinates,
                    **kwargs
                )
            else:
                best_threshold = candidates[argmax]
                self.__store_threshold__(node_label, parent_labels, coordinates, best_threshold)

                meta, subsets = self.__subsets_and_meta__(
                    node_label=node_label,
                    node_id=node_id,
                    node_level=node_level,
                    subset_index=subset_index,
                    threshold=best_threshold,
                )
        except Exception as e:
            raise e

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return subsets
        else:
            return meta, subsets

    def __set_terminal__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        # node_label in this case is probably the self.target_attr; so it
        # is not significant for the **real** label of the terminal node.
        label = Counter(Individual.full.loc[subset_index, self.target_attr]).most_common()[0][0]

        meta = {
            'label': label,
            'threshold': None,
            'terminal': True,
            'level': node_level,
            'node_id': node_id,
            'color': Individual._terminal_node_color
        }

        return meta, (None, None)

    def __set_categorical__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        raise NotImplementedError('not implemented yet!')

    @staticmethod
    def __set_error__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        raise TypeError('Unsupported data type for column %s!' % node_label)

    @staticmethod
    def __subsets_and_meta__(node_label, node_id, node_level, subset_index, threshold):
        meta = {
            'label': node_label,
            'threshold': threshold,
            'terminal': False,
            'level': node_level,
            'node_id': node_id,
            'color': Individual._root_node_color if
            node_level == 0 else Individual._inner_node_color
        }

        less_or_equal = (Individual.full[node_label] <= threshold).values.ravel()
        subset_left = less_or_equal & subset_index
        subset_right = np.invert(less_or_equal) & subset_index

        return meta, (subset_left, subset_right)

    def plot(self, savepath=None, test_acc=None):
        """
        Draw this individual.
        """

        # from wand.image import Image
        # from wand import display
        # img = Image(filename='.temp.pdf')
        # display.display(img)

        fig = plt.figure(figsize=(40, 30))

        tree = self.tree  # type: nx.DiGraph
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(tree, root=0, prog='dot')

        node_list = tree.nodes(data=True)
        edge_list = tree.edges(data=True)

        node_labels = {x[0]: str(x[1]['node_id']) + ': ' + str(x[1]['label']) for x in node_list}
        node_colors = [x[1]['color'] for x in node_list]
        edge_labels = {(x1, x2): d['threshold'] for x1, x2, d in edge_list}

        nx.draw_networkx_nodes(tree, pos, node_size=1000, node_color=node_colors)  # nodes
        nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
        nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)

        if self.ind_id is not None:
            plt.text(
                0.8,
                0.9,
                'individual id: %03.d' % self.ind_id,
                fontsize=15,
                horizontalalignment='left',
                verticalalignment='center',
                transform=fig.transFigure
            )

        plt.text(
            0.8,
            0.94,
            'val accuracy: %0.4f' % self.fitness,
            fontsize=15,
            horizontalalignment='left',
            verticalalignment='center',
            transform=fig.transFigure
        )

        if test_acc is not None:
            plt.text(
                0.8,
                0.98,
                'test acc: %0.4f' % test_acc,
                fontsize=15,
                horizontalalignment='left',
                verticalalignment='center',
                transform=fig.transFigure
            )

        plt.axis('off')

        # plt.show()
        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight', format='pdf')
            plt.close()

    @property
    def inverse_height(self):
        height = self.height
        inv_height = self.max_height - height
        return inv_height

    @property
    def n_nodes(self):
        return len(self.tree)

    def __isclose__(self, other):
        quality_diff = abs(self.fitness - other.quality)
        return quality_diff <= Individual.rtol

    def __le__(self, other):  # less or equal
        if self.__isclose__(other):
            return self.height <= other.height
        return self.fitness < other.quality

    def __lt__(self, other):  # less than
        if self.__isclose__(other):
            return self.height < other.height
        else:
            return self.fitness < other.quality

    def __ge__(self, other):  # greater or equal
        if self.__isclose__(other):
            return self.height >= other.height
        else:
            return self.fitness >= other.quality

    def __gt__(self, other):  # greater than
        if self.__isclose__(other):
            return self.height > other.height
        else:
            return self.fitness > other.quality

    def __eq__(self, other):  # equality
        if self.__isclose__(other):
            return self.height == other.height
        else:
            return False

    def __ne__(self, other):  # inequality
        if self.__isclose__(other):
            return self.height != other.height
        else:
            return True

    def __str__(self):
        return 'fitness: %0.2f height: %d' % (self.fitness, self.height)

    def get_predictive_type(self, dtype):
        """
        Tells whether the attribute is categorical or numerical.

        :type dtype: type
        :param dtype: dtype of an attribute.
        :rtype: str
        :return: Whether this attribute is categorical or numerical.
        """

        raw_type = self.raw_type_dict[str(dtype)]
        func = self.handler_dict[raw_type]

        if func.__name__ == self.__set_categorical__.__name__:
            return 'categorical'
        elif func.__name__ == self.__set_numerical__.__name__:
            return 'numerical'
        else:
            raise TypeError('Unsupported column type! Column type is: %s' % dtype)

    handler_dict = {
        'object': __set_categorical__,
        'str': __set_categorical__,
        'int': __set_numerical__,
        'float': __set_numerical__,
        'bool': __set_categorical__,
        'complex': __set_error__,
        'class': __set_terminal__
    }

    # TODO replace with np.sctypes!

    raw_type_dict = {
        'int': 'int',
        'int_': 'int',
        'intc': 'int',
        'intp': 'int',
        'int8': 'int',
        'int16': 'int',
        'int32': 'int',
        'int64': 'int',
        'uint8': 'int',
        'uint16': 'int',
        'uint32': 'int',
        'uint64': 'int',
        'float': 'float',
        'float_': 'float',
        'float16': 'float',
        'float32': 'float',
        'float64': 'float',
        'complex_': 'complex',
        'complex64': 'complex',
        'complex128': 'complex',
        'object': 'object',
        'bool_': 'bool',
        'bool': 'bool',
        'str': 'str',
    }
