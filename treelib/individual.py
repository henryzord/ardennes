# coding=utf-8

import collections
import copy
import itertools as it
from collections import Counter

import math
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.metrics import *

from treelib.node import *
from c_individual import make_predictions
import operator as op
from scipy.stats import hmean

import warnings

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
    max_height = -1

    processor = None

    thresholds = dict()  # thresholds for nodes

    full = None
    sets = None
    arg_sets = None
    y_test_true = None
    y_val_true = None
    y_train_true = None

    _shortest_path = dict()  # type: dict
    tree = None  # type: nx.DiGraph

    ind_id = None
    fitness = None  # type: float
    height = None
    n_nodes = None
    train_acc_score = None
    val_acc_score = None
    test_acc_score = None
    test_precision_score = None
    test_f1_score = None

    rtol = 1e-3

    def __init__(self, gm, **kwargs):
        """
        
        :type gm: treelib.graphical_model.GraphicalModel
        :param gm:
        :type sets: dict
        :param sets:
        """

        self.ind_id = kwargs['ind_id'] if 'ind_id' in kwargs else None
        self.iteration = kwargs['iteration'] if 'iteration' in kwargs else None
        self.whole = kwargs['whole'] if 'whole' in kwargs else None

        self.sample(gm)

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
        Selects all nodes which are in the given level.

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

    def reduced_error_pruning(self, tree):
        # depth first search

        def go_removing_deep(current_node, _tree):
            successors = _tree.successors(current_node)
            for successor in successors:
                _tree = go_removing_deep(successor, _tree)

            _tree.remove_node(current_node)

            return _tree

        def _sub_remove_(current_node, _tree, _train_acc, _val_acc):
            successors = _tree.successors(current_node)

            for successor in successors:
                _tree = _sub_remove_(successor, _tree, _train_acc, _val_acc)

            if current_node == 0:
                return _tree

            n_tree = copy.deepcopy(_tree)

            n_tree.node[current_node]['terminal'] = True
            n_tree.node[current_node]['label'] = n_tree.node[current_node]['common_class']

            for successor in successors:
                go_removing_deep(successor, n_tree)

            n_train_acc = reduce(
                op.add,
                map(lambda x: x['inst_correct'] if x['terminal'] else 0,
                    n_tree.node.itervalues()
                    )
            ) / float(n_tree.node[0]['inst_total'])

            n_val_acc = accuracy_score(Individual.y_val_true, self.predict(Individual.sets['val'], tree=n_tree))

            if n_train_acc >= _train_acc and n_val_acc >= _val_acc:
                return n_tree
            return _tree

        val_acc = accuracy_score(Individual.y_val_true, self.predict(Individual.sets['val'], tree=tree))

        train_acc = reduce(
            op.add,
            map(lambda x: x['inst_correct'] if x['terminal'] else 0,
                tree.node.itervalues()
                )
        ) / float(tree.node[0]['inst_total'])

        tree = _sub_remove_(0, tree, train_acc, val_acc)

        return tree

    def sample(self, gm):
        warnings.warn('WARNING: using ALSO validation for threshold setting!')
        if not self.whole:
            if np.sin(5. * (self.iteration/math.pi)) <= 0:
                del self.__class__.thresholds
                self.__class__.thresholds = dict()  # thresholds for nodes
                arg_threshold = Individual.arg_sets['train_index']
            else:
                del self.__class__.thresholds
                self.__class__.thresholds = dict()  # thresholds for nodes
                arg_threshold = Individual.arg_sets['val_index']
        else:
            arg_threshold = Individual.arg_sets['train_index'] | Individual.arg_sets['val_index']

        self.tree = self.tree = self.__set_node__(
            node_id=0,
            gm=gm,
            tree=nx.DiGraph(),
            subset_index=arg_threshold,
            level=0,
            parent_labels=[],
            coordinates=[],
        )  # type: nx.DiGraph

        self._shortest_path = nx.shortest_path(self.tree, source=0)  # source equals to root

        y_train_pred = self.predict(Individual.sets['train'])
        y_val_pred = self.predict(Individual.sets['val'])
        y_test_pred = self.predict(Individual.sets['test'])

        self.train_acc_score = accuracy_score(Individual.y_train_true, y_train_pred)
        self.val_acc_score = accuracy_score(Individual.y_val_true, y_val_pred)
        self.test_acc_score = accuracy_score(Individual.y_test_true, y_test_pred)
        self.test_precision_score = precision_score(Individual.y_test_true, y_test_pred, average='micro')
        self.test_f1_score = f1_score(Individual.y_test_true, y_test_pred, average='micro')

        self.fitness = hmean([self.val_acc_score, self.train_acc_score])

        self.height = max(map(len, self._shortest_path.itervalues()))
        self.n_nodes = len(self.tree.node)

    def __set_node__(self, node_id, gm, tree, subset_index, level, parent_labels, coordinates):
        try:
            label = gm.sample(
                node_id=node_id, level=level, parent_labels=parent_labels, enforce_nonterminal=(level == 0)
            )
        except KeyError as ke:
            if level >= Individual.max_height:
                label = Individual.target_attr
            else:
                raise ke

        if any((
                Individual.full.loc[subset_index, Individual.target_attr].unique().shape[0] == 1,  # only one class
                level >= Individual.max_height,  # level deeper than maximum depth
                label == Individual.target_attr   # was sampled to be a class
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
            arg_node = (arg_node * 2) + (not go_left) + 1
            node = tree.node[arg_node]

        return node['label']

    def predict(self, samples, tree=None):
        """
        Makes predictions for unseen samples.

        :param samples: Either a pandas.DataFrame (for multiple samples)or pandas.Series (for a single object).
        :rtype: numpy.ndarray
        :return: The predicted class for each sample.
        """
        # if isinstance(samples, pd.DataFrame):
        #     preds = samples.apply(self.__predict_object__, axis=1).as_matrix()
        # elif isinstance(samples, pd.Series):
        #     preds = self.__predict_object__(samples)
        # else:
        #     raise TypeError('Invalid type for this method! Must be either a pandas.DataFrame or pandas.Series!')

        data = samples.values.ravel().tolist()
        if tree is None:
            tree = self.tree.node
        else:
            tree = tree.node

        preds = make_predictions(
            samples.shape,
            data,
            tree,
            range(samples.shape[0]),
            self.processor.attribute_index
        )

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

            if self.processor.max_n_candidates is None:
                candidates = np.array(unique_vals + [
                    (unique_vals[i] + unique_vals[i + 1]) / 2.
                    if (i + 1) < len(unique_vals) else unique_vals[i] for i in xrange(len(unique_vals))
                ][:-1])
            else:
                candidates = np.linspace(unique_vals[0], unique_vals[-1], self.processor.max_n_candidates)

            gains = self.processor.get_ratios(subset_index, node_label, candidates)

            argmax = np.argmax(gains)

            best_gain = gains[argmax]

            if best_gain <= 0:
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
        # node_label in this case is probably the Individual.target_attr; so it
        # is not significant for the **real** label of the terminal node.

        counter = Counter(Individual.full.loc[subset_index, Individual.target_attr])
        label, count_frequent = counter.most_common()[0]

        meta = {
            'label': label,
            'common_class': label,
            'threshold': None,
            'terminal': True,
            'inst_correct': count_frequent,
            'inst_total': subset_index.sum(),
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

        counter = Counter(Individual.full.loc[subset_index, Individual.target_attr])

        meta = {
            'label': node_label,
            'threshold': threshold,
            'inst_correct': counter.most_common()[0][1],
            'inst_total': subset_index.sum(),
            'common_class': counter.most_common()[0][0],
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

    def plot(self, savepath=None):
        """
        Draw this individual.
        """

        from networkx.drawing.nx_agraph import graphviz_layout

        fig = plt.figure(figsize=(40, 30))

        tree = self.tree  # type: nx.DiGraph
        pos = graphviz_layout(tree, root=0, prog='dot')

        node_list = tree.nodes(data=True)
        edge_list = tree.edges(data=True)

        node_labels = {
            x[0]: '%s: %s\n%s' % (str(x[1]['node_id']), str(x[1]['label']), '%s/%s' % (str(x[1]['inst_correct']), str(x[1]['inst_total'])) if x[1]['terminal'] else '')
            for x in node_list
        }
        node_colors = [x[1]['color'] for x in node_list]
        edge_labels = {(x1, x2): d['threshold'] for x1, x2, d in edge_list}

        nx.draw_networkx_nodes(tree, pos, node_size=1000, node_color=node_colors)  # nodes
        nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
        nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)

        plt.text(
            0.2,
            0.9,
            '\n'.join([
                'individual id: %03.d' % self.ind_id,
                'height: %d' % self.height,
                'n_nodes: %d' % self.n_nodes,
                'train accuracy: %0.4f' % self.train_acc_score,
                'val accuracy: %0.4f' % self.val_acc_score,
                'test accuracy: %0.4f' % self.test_acc_score if self.test_acc_score is not None else '',
                'iteration: %d' % self.iteration if self.iteration is not None else ''

            ]),
            fontsize=15,
            horizontalalignment='right',
            verticalalignment='top',
            transform=fig.transFigure
        )

        plt.axis('off')

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight', format='pdf')
            plt.close()

    def __is_close__(self, other):
        quality_diff = abs(self.fitness - other.fitness)
        return quality_diff <= Individual.rtol

    def __le__(self, other):  # less or equal
        if self.__is_close__(other):
            return self.n_nodes >= other.n_nodes
        return self.fitness <= other.fitness

    def __lt__(self, other):  # less than
        if self.__is_close__(other):
            return self.n_nodes > other.n_nodes
        else:
            return self.fitness < other.fitness

    def __ge__(self, other):  # greater or equal
        if self.__is_close__(other):
            return self.n_nodes <= other.n_nodes
        else:
            return self.fitness >= other.fitness

    def __gt__(self, other):  # greater than
        if self.__is_close__(other):
            return self.n_nodes < other.n_nodes
        else:
            return self.fitness > other.fitness

    def __eq__(self, other):  # equality
        if self.__is_close__(other):
            return self.n_nodes == other.n_nodes
        else:
            return False

    def __ne__(self, other):  # inequality
        if self.__is_close__(other):
            return self.n_nodes != other.n_nodes
        else:
            return True

    def __str__(self):
        return 'fitness: %0.2f height: %d n_nodes: %d' % (self.fitness, self.height, self.n_nodes)

    def get_predictive_type(self, dtype):
        """
        Tells whether the attribute is categorical or numerical.

        :type dtype: type
        :param dtype: dtype of an attribute.
        :rtype: str
        :return: Whether this attribute is categorical or numerical.
        """

        raw_type = Individual.raw_type_dict[str(dtype)]
        func = Individual.handler_dict[raw_type]

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
