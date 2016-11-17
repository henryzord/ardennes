# coding=utf-8

import itertools as it
from collections import Counter

import collections
import networkx as nx
import pandas as pd

from treelib.classes import AbstractTree

from matplotlib import pyplot as plt
import numpy as np

__author__ = 'Henry Cagnini'


class Individual(AbstractTree):
    _terminal_node_color = '#98FB98'
    _inner_node_color = '#0099ff'
    _root_node_color = '#FFFFFF'
    column_types = None  # type: dict
    sets = None  # type: dict
    tree = None  # type: nx.DiGraph
    val_acc = None  # type: float
    ind_id = None

    thresholds = dict()
    max_height = -1

    shortest_path = dict()  # type: dict
    """
    A dictionary where each key is the node's name, and each value a list of integers denoting the
    \'shortest path\' from the node to the root.
    """

    def __init__(self, graphical_model, max_height, sets, **kwargs):
        """
        
        :type graphical_model: treelib.graphical_model.GraphicalModel
        :param graphical_model:
        :type sets: dict
        :param sets:
        :type kwargs: dict
        :param kwargs:
        """
        super(Individual, self).__init__(**kwargs)

        if 'ind_id' in kwargs:
            self.ind_id = kwargs['ind_id']
        else:
            self.ind_id = None

        if Individual.column_types is None:
            Individual.column_types = {
                x: self.raw_type_dict[str(sets['train'][x].dtype)] for x in sets['train'].columns
                }  # type: dict
            Individual.column_types['class'] = 'class'
        self.column_types = Individual.column_types

        self.max_height = max_height
        self.id_generator = None

        self.sets = sets
        self.sample(graphical_model, sets)

    @property
    def id_ind(self):
        return self.ind_id

    @property
    def height(self):
        return max(map(len, self.shortest_path.itervalues()))

    @property
    def fitness(self):
        """
        :rtype: float
        :return: Fitness of this individual.
        """
        return self.val_acc

    def nodes_at_depth(self, depth):
        """
        Picks all nodes which are in the given level.

        :type depth: int
        :param depth: The level to pick
        :rtype: list of dict
        :return: A list of the nodes at the given level.
        """
        depths = {k: self.depth_of(k) for k in self.shortest_path.iterkeys()}
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
        parents = self.shortest_path[node_id]
        parents.remove(node_id)
        return parents

    def depth_of(self, node_id):
        """
        The depth which a node lies in the tree.

        :type node_id: int
        :param node_id: The id of the node, starting from zero (root).
        :rtype: int
        :return: Depth of the node, starting with zero (root).
        """

        return len(self.shortest_path[node_id]) - 1

    def __str__(self):
        return 'fitness: %0.2f' % self.val_acc

    def plot(self, savepath=None, test_set=None):
        """
        Draw this individual.
        """

        # from wand.image import Image
        # from wand import display
        # img = Image(filename='.temp.pdf')
        # display.display(img)

        fig = plt.figure()

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
            'val accuracy: %0.4f' % self.val_acc,
            fontsize=15,
            horizontalalignment='left',
            verticalalignment='center',
            transform=fig.transFigure
        )

        if test_set is not None:
            test_acc = self.validate(test_set)

            plt.text(
                0.8,
                0.98,
                'test accuracy: %0.4f' % test_acc,
                fontsize=15,
                horizontalalignment='left',
                verticalalignment='center',
                transform=fig.transFigure
            )

        plt.axis('off')

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight', format='pdf')
            plt.close()

    # ############################ #
    # sampling and related methods #
    # ############################ #

    def sample(self, graphical_model, sets):
        self.id_generator = it.count(start=0, step=1)

        self.tree = self.__set_tree__(graphical_model, sets['train'])  # type: nx.DiGraph

        self.shortest_path = nx.shortest_path(self.tree, 0)

        self.val_acc = self.validate(self.sets['val'])

    def __set_tree__(self, graphical_model, train_set):
        tree = nx.DiGraph()

        subset = train_set

        tree, root_id = self.__set_node__(
            graphical_model=graphical_model,
            tree=tree,
            subset=subset,
            level=0,
            parent_labels=[]
        )
        return tree

    def __set_node__(self, graphical_model, tree, subset, level, parent_labels):
        """

        :param graphical_model:
        :type tree: networkx.DiGraph
        :param tree:
        :param subset:
        :param level:
        :param parent_labels:
        :return:
        """

        # if:
        # 1. there is only one instance (or none) coming to this node; or
        # 2. there is only one class coming to this node;
        # then set this as a terminal node
        node_id = next(self.id_generator)

        try:
            label = graphical_model.sample(level=level, parent_labels=parent_labels, enforce_nonterminal=(level == 0))
        except IndexError as ie:
            if level >= self.max_height:
                label = self.target_attr
            else:
                raise ie

        if any((
            subset.empty,
            subset[subset.columns[-1]].unique().shape[0] == 1,
            level >= self.max_height,
            label == self.target_attr
        )):
            meta, subset_left, subset_right = self.__set_terminal__(
                node_label=None,
                parent_labels=parent_labels,
                level=level,
                subset=subset,
                node_id=node_id
            )
        else:
            meta, subsets = self.__set_inner_node__(
                label=label,
                parent_labels=parent_labels,
                node_level=level,
                graphical_model=graphical_model,
                subset=subset,
                node_id=node_id
            )

            if meta['threshold'] is not None:
                children_id = []
                for child_subset in subsets:
                    # self, graphical_model, tree, subset, level, parent_labels

                    tree, some_id = self.__set_node__(
                        tree=tree,
                        graphical_model=graphical_model,
                        subset=child_subset,
                        level=level + 1,
                        parent_labels=parent_labels + [label]
                    )
                    children_id += [some_id]

                if isinstance(meta['threshold'], float):
                    attr_dicts = [
                        {'threshold': '< %0.2f' % meta['threshold']},
                        {'threshold': '>= %0.2f' % meta['threshold']}
                    ]
                elif isinstance(meta['threshold'], collections.Iterable):
                    attr_dicts = [{'threshold': t} for t in meta['threshold']]
                else:
                    raise TypeError('invalid type for threshold!')

                for some_id, attr_dict in it.izip(children_id, attr_dicts):
                    tree.add_edge(node_id, some_id, attr_dict=attr_dict)

        tree.add_node(node_id, attr_dict=meta)
        return tree, node_id

    @staticmethod
    def entropy(subset, target_attr):
        # the smaller, the better
        size = float(subset.shape[0])

        counter = Counter(subset[target_attr])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy

    def __predict_object__(self, obj):
        arg_node = 0  # always start with root

        tree = self.tree  # type: nx.DiGraph

        node = tree.node[arg_node]

        while not node['terminal']:
            if isinstance(node['threshold'], float):
                go_left = obj[node['label']] < node['threshold']
                successors = tree.successors(arg_node)
                arg_node = (int(go_left) * min(successors)) + (int(not go_left) * max(successors))
            elif isinstance(node['threshold'], collections.Iterable):
                edges = self.tree.edge[arg_node]
                neither_case = None
                was_set = False
                for v, d in edges.iteritems():
                    if d['threshold'] == 'None':
                        neither_case = v

                    if obj[node['label']] == d['threshold']:
                        arg_node = v
                        was_set = True
                        break

                # next node is the one which the category is neither one of the seen ones in the training phase
                if not was_set:
                    arg_node = neither_case
            else:
                raise TypeError('invalid type for threshold!')

            node = tree.node[arg_node]

        return node['label']

    def __validate_object__(self, obj):
        """
        
        :type obj: pandas.Series
        :param obj:
        :return:
        """
        label = self.__predict_object__(obj)
        return obj.iloc[-1] == label

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

    def validate(self, test_set=None, X_test=None, y_test=None):
        """
        Assess the accuracy of this Individual against the provided set.
        
        :type test_set: pandas.DataFrame
        :param test_set: a matrix with the class attribute in the last position (i.e, column).
        :return: The accuracy of this model when testing with test_set.
        """

        if test_set is None:
            test_set = pd.DataFrame(
                np.hstack((X_test, y_test[:, np.newaxis]))
            )

        hit_count = test_set.apply(self.__validate_object__, axis=1).sum()
        acc = hit_count / float(test_set.shape[0])
        return acc

    def __set_inner_node__(self, label, parent_labels, node_level, subset, node_id, **kwargs):
        attr_type = Individual.column_types[label]

        out = self.handler_dict[attr_type](
            self,
            node_label=label,
            parent_labels=parent_labels,
            node_level=node_level,
            subset=subset,
            node_id=node_id,
            **kwargs
        )
        return out

    def __store_threshold__(self, node_label, subset, threshold):
        """

        :type node_label: str
        :param node_label:
        :type subset: pandas.DataFrame
        :param subset:
        :param threshold:
        """
        column_type = subset.dtypes[node_label]

        if column_type in [np.float32, np.float64, np.int32, np.int64]:
            _mean = subset[node_label].mean()
            _std = subset[node_label].std()
        elif column_type == object:
            counts = subset[node_label].apply(len)
            _mean = counts.mean()
            _std = counts.std()
        else:
            raise TypeError('invalid type for threshold! Encountered %s' % str(column_type))

        key = '[%s][%05.8f][%05.8f]' % (str(node_label), _mean, _std)
        self.__class__.thresholds[key] = threshold

    def __retrieve_threshold__(self, node_label, subset):
        """

        :param node_label:
        :type subset: pandas.DataFrame
        :param subset:
        :return:
        """

        column_type = subset.dtypes[node_label]

        if column_type in [np.float32, np.float64, np.int32, np.int64]:
            _mean = subset[node_label].mean()
            _std = subset[node_label].std()
        elif column_type == object:
            counts = subset[node_label].apply(len)
            _mean = counts.mean()
            _std = counts.std()
        else:
            raise TypeError('invalid type for threshold! Encountered %s' % str(column_type))

        key = '[%s][%05.8f][%05.8f]' % (str(node_label), _mean, _std)
        return self.__class__.thresholds[key]

    def __set_numerical__(self, node_label, parent_labels, node_level, subset, node_id, **kwargs):
        try:
            best_threshold = self.__retrieve_threshold__(node_label, subset)
            meta, subsets = self.__subsets_and_meta__(
                node_label, best_threshold, subset, node_id, node_level
            )
        except KeyError:
            unique_vals = sorted(subset[node_label].unique())
            candidates = [
                (unique_vals[i] + unique_vals[i + 1]) / 2.
                if (i + 1) < len(unique_vals) else unique_vals[i] for i in xrange(len(unique_vals))
            ][:-1]

            best_threshold = -np.inf
            best_entropy = np.inf
            for cand in candidates:
                entropy = self.__get_entropy__(node_label, cand, subset)
                if entropy < best_entropy:
                    best_threshold = cand
                    best_entropy = entropy

            self.__store_threshold__(node_label, subset, best_threshold)

            meta, subsets = self.__subsets_and_meta__(
                node_label, best_threshold, subset, node_id, node_level
            )

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return subsets
        else:
            return meta, subsets

    def __set_terminal__(self, node_label, parent_labels, level, subset, node_id, **kwargs):
        # node_label in this case is probably the self.target_attr; so it
        # is not significant for the **real** label of the terminal node.

        if not subset.empty:
            label = Counter(subset[self.target_attr]).most_common()[0][0]
        else:
            # if there's no data to train, how can I know which class is it? Simply pick one and throw at the user!
            label = np.random.choice(self.class_labels)

        meta = {
            'label': label,
            'threshold': None,
            'terminal': True,
            'level': level,
            'node_id': node_id,
            'color': Individual._terminal_node_color
        }

        return meta, pd.DataFrame([]), pd.DataFrame([])

    def __set_categorical__(self, node_label, parent_labels, node_level, subset, node_id, **kwargs):
        # adds an option where the category is neither one of the found in the training set
        categories = subset[node_label].unique().tolist() + ['None']

        out = self.__subsets_and_meta__(
            node_label=node_label,
            threshold=categories,
            subset=subset,
            node_id=node_id,
            node_level=node_level
        )

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return list(out)[1:]
        else:
            return out

    @staticmethod
    def __set_error__(self, node_label, parent_label, subset, **kwargs):
        raise TypeError('Unsupported data type for column %s!' % node_label)

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

    @staticmethod
    def __get_entropy__(node_label, threshold, subset):
        """
        Gets entropy for a given threshold.

        :type subset: pandas.DataFrame
        :param subset: data coming to this node.
        :type node_label: int
        :param node_label: for picking the attribute in the subset.
        :param threshold: Threshold value. May be a floating (for numerical attributes) or string (categorical).
        :rtype: float
        :return: the entropy.
        """

        subset_left = subset.loc[subset[node_label] < threshold]
        subset_right = subset.loc[subset[node_label] >= threshold]

        entropy = \
            Individual.entropy(subset_left, Individual.target_attr) + \
            Individual.entropy(subset_right, Individual.target_attr)

        return entropy

    @staticmethod
    def __subsets_and_meta__(node_label, threshold, subset, node_id, node_level):
        meta = {
            'label': node_label,
            'threshold': threshold,
            'terminal': False,
            'level': node_level,
            'node_id': node_id,
            'color': Individual._root_node_color if
            node_level == 0 else Individual._inner_node_color
        }

        if isinstance(threshold, collections.Iterable):
            subsets = [subset.loc[subset[node_label] == x] for x in threshold]

        else:
            import operator as op
            ops = [op.lt, op.ge]  # <, >=
            subsets = [
                subset.loc[x(subset[node_label], threshold)] for x in ops
            ]

        return meta, subsets
