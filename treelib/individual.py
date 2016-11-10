# coding=utf-8

import itertools as it
from collections import Counter

import networkx as nx
import pandas as pd

from treelib.classes import AbstractTree
from treelib import node

from matplotlib import pyplot as plt
from node import *

__author__ = 'Henry Cagnini'


class Individual(AbstractTree):
    _terminal_node_color = '#98FB98'
    _inner_node_color = '#0099ff'
    _root_node_color = '#FFFFFF'
    column_types = None  # type: dict
    sets = None  # type: dict
    tree = None  # type: nx.DiGraph
    val_acc = None  # type: float
    node_id = None

    thresholds = dict()
    max_height = -1

    shortest_path = dict()  # type: dict
    """
    A dictionary where each key is the node's name, and each value a list of integers denoting the
    \'shortest path\' from the node to the root.
    """

    def __init__(self, graphical_model, max_height, sets, **kwargs):
        """
        
        :type graphical_model: treelib.graphical_models.GraphicalModel
        :param graphical_model:
        :type sets: dict
        :param sets:
        :type kwargs: dict
        :param kwargs:
        """
        super(Individual, self).__init__(**kwargs)

        if 'node_id' in kwargs:
            self.node_id = kwargs['node_id']

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
    def height(self):
        return max(map(len, self.shortest_path.itervalues()))

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

    def plot(self, metadata_path=None):
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

        node_labels = {x[0]: x[1]['label'] for x in node_list}
        node_colors = [x[1]['color'] for x in node_list]
        edge_labels = {(x1, x2): d['threshold'] for x1, x2, d in edge_list}

        nx.draw_networkx_nodes(tree, pos, node_size=1000, node_color=node_colors)  # nodes
        nx.draw_networkx_edges(tree, pos, edgelist=edge_list, style='dashed')  # edges
        nx.draw_networkx_labels(tree, pos, node_labels, font_size=16)  # node labels
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_size=16)

        plt.text(
            0.8,
            0.9,
            'Fitness: %0.4f' % self.val_acc,
            fontsize=15,
            horizontalalignment='center',
            verticalalignment='center',
            transform=fig.transFigure
        )

        if self.node_id is not None:
            plt.text(
                0.1,
                0.1,
                'node_id: %03.d' % self.node_id,
                fontsize=15,
                horizontalalignment='center',
                verticalalignment='center',
                transform=fig.transFigure
            )

        plt.axis('off')

    @property
    def fitness(self):
        """
        :rtype: float
        :return: Fitness of this individual.
        """
        return self.val_acc

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
            meta, subset_left, subset_right = self.__set_inner_node__(
                label=label,
                parent_labels=parent_labels,
                level=level,
                graphical_model=graphical_model,
                subset=subset,
                node_id=node_id
            )

            if meta['threshold'] is not None:
                children_id = []
                for child_subset in [subset_left, subset_right]:
                    tree, some_id = self.__set_node__(
                        graphical_model=graphical_model,
                        tree=tree,
                        subset=child_subset,
                        level=level + 1,
                        parent_labels=parent_labels + [label]
                    )
                    children_id += [some_id]

                if isinstance(meta['threshold'], float):
                    attr_dict_left = {'threshold': '< %0.2f' % meta['threshold']}
                    attr_dict_right = {'threshold': '>= %0.2f' % meta['threshold']}
                elif type(meta['threshold']) in [list, tuple]:
                    raise NotImplementedError('not implemented yet!')
                    attr_dict_left = {'threshold': '%s' % ','.join(meta['threshold'])}
                    attr_dict_right = {
                        'threshold': '%s' % ', '.join(
                            set(subset[meta['label']].unique()) - set(meta['threshold']))
                    }
                else:
                    raise TypeError('invalid type for threshold!')

                for some_id, attr_dict in it.izip(children_id, [attr_dict_left, attr_dict_right]):
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
        successors = tree.successors(arg_node)

        while not node['terminal']:
            if isinstance(node['threshold'], float):
                go_left = obj[node['label']] < node['threshold']
            elif type(node['threshold']) in [tuple, list]:
                go_left = obj[node['label']] in node['threshold']
            else:
                raise TypeError('invalid type for threshold!')

            arg_node = (int(go_left) * min(successors)) + (int(not go_left) * max(successors))
            successors = tree.successors(arg_node)
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

    def __set_inner_node__(self, label, parent_labels, level, graphical_model, subset, node_id, **kwargs):
        attr_type = Individual.column_types[label]
        out = self.handler_dict[attr_type](
            self,
            node_label=label,
            parent_labels=parent_labels,
            level=level,
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

    def __set_numerical__(self, node_label, parent_labels, level, subset, node_id, **kwargs):
        pd.options.mode.chained_assignment = None

        def middle_value(x):
            """
            Gets the middle value between two values.

            :type x: pandas.core.series.Series
            :param x: A row in the node_label column.
            :rtype: float
            :return: middle value between this row and the predecessor of it.
            """
            first = ((x.name - 1) * (x.name > 0)) + (x.name * (x.name <= 0))
            second = x.name

            average = (border_vals.loc[first, node_label] + border_vals.loc[second, node_label]) / 2.

            return average

        def same_class(x):
            """
            Verifies if two neighboring objects have the same class.
            
            :type x: pandas.core.series.Series
            :param x: An object with a predictive attribute and the class attribute.
            :rtype: bool
            :return: True if the neighbor of x have the same class; False otherwise.
            """

            first = ((x.name - 1) * (x.name > 0)) + (x.name * (x.name <= 0))
            second = x.name
            column = Individual.target_attr

            # return ss[column].iloc[first] != ss[column].iloc[second]
            return ss.loc[first, column] != ss.loc[second, column]

        def get_entropy(threshold):
            """
            Gets entropy for a given threshold.
            
            :type threshold: float
            :param threshold: Threshold value.
            :rtype: float
            :return: the entropy.
            """

            subset_left = subset.loc[subset[node_label] < threshold]
            subset_right = subset.loc[subset[node_label] >= threshold]

            entropy = \
                Individual.entropy(subset_left, Individual.target_attr) + \
                Individual.entropy(subset_right, Individual.target_attr)

            return entropy

        def __subsets_and_meta__(_best_threshold, _level):
            _best_subset_left = subset.loc[subset[node_label] < _best_threshold]
            _best_subset_right = subset.loc[subset[node_label] >= _best_threshold]

            _meta = {
                'label': node_label,
                'threshold': _best_threshold,
                'terminal': False,
                'level': _level,
                'node_id': node_id,
                'color': Individual._root_node_color if
                _level == 0 else Individual._inner_node_color
            }

            return _meta, _best_subset_left, _best_subset_right

        try:
            best_threshold = self.__retrieve_threshold__(node_label, subset)
            meta, best_subset_left, best_subset_right = __subsets_and_meta__(best_threshold, level)
        except KeyError:
            ss = subset[[node_label, Individual.target_attr]]  # type: pd.DataFrame
            ss = ss.sort_values(by=node_label).reset_index(inplace=False, drop=True)

            ss['same_class'] = ss.apply(same_class, axis=1)
            border_vals = ss.loc[ss['same_class'] == True]

            if border_vals.empty:
                meta, best_subset_left, best_subset_right = self.__set_terminal__(
                    node_label=Individual.target_attr, parent_labels=parent_labels, subset=subset, node_id=node_id,
                    **kwargs
                )
            else:
                # this only clips the range of values to pick; it doesn't prevent picking a not promising value.
                border_vals.reset_index(inplace=True, drop=True)

                candidates = border_vals.apply(middle_value, axis=1).unique()

                best_threshold = -np.inf
                best_entropy = np.inf
                for cand in candidates:
                    entropy = get_entropy(cand)
                    if entropy < best_entropy:
                        best_threshold = cand
                        best_entropy = entropy

                self.__store_threshold__(node_label, subset, best_threshold)

                meta, best_subset_left, best_subset_right = __subsets_and_meta__(best_threshold, level)

                pd.options.mode.chained_assignment = 'warn'

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return best_subset_left, best_subset_right
        else:
            return meta, best_subset_left, best_subset_right

    def __set_terminal__(self, node_label, parent_labels, level, subset, node_id, **kwargs):
        # node label in this case is probably the self.target_attr; so it is not significant
        # for the **real** label of the terminal node.

        if not subset.empty:
            label = Counter(subset[self.target_attr]).most_common()[0][0]
        else:
            label = np.random.choice(self.class_labels)
            import warnings
            warnings.warn('WARNING: ramdonly picking a class label!')

        meta = {
            'label': label,
            'threshold': None,
            'terminal': True,
            'level': level,
            'node_id': node_id,
            'color': Individual._terminal_node_color
        }

        # TODO set all children node as None in the label!

        return meta, pd.DataFrame([]), pd.DataFrame([])

    # def __set_categorical__(self, node_label, parent_label, subset, **kwargs):
    #     # TODO enhance this method to perform more smart splits.
    #     # TODO currently it tries all combinations. what a mess!
    #
    #     def __subsets_and_meta__(group):
    #         _best_subset_left = subset.loc[subset[node_label].apply(lambda x: x in group).index]
    #         _best_subset_right = subset.loc[subset[node_label].apply(lambda x: x not in group).index]
    #
    #         _meta = {
    #             'label': node_label,
    #             'threshold': group,
    #             'terminal': False,
    #             'color': Individual._root_node_color if
    #             kwargs['variable_name'] == node.root else Individual._inner_node_color
    #         }
    #
    #         return _meta, _best_subset_left, _best_subset_right
    #
    #     def get_entropy(group):
    #         """
    #         Gets entropy for a given set of values.
    #
    #         :type group: tuple
    #         :param group: Set of values.
    #         :rtype: float
    #         :return: the entropy.
    #         """
    #
    #         subset_left = subset.loc[subset[node_label].apply(lambda x: x in group).index]
    #         subset_right = subset.loc[subset[node_label].apply(lambda x: x not in group).index]
    #
    #         entropy = \
    #             Individual.entropy(subset_left, Individual.target_attr) + \
    #             Individual.entropy(subset_right, Individual.target_attr)
    #
    #         return entropy
    #
    #     try:
    #         best_threshold = self.__retrieve_threshold__(node_label, subset)
    #         meta, best_subset_left, best_subset_right = __subsets_and_meta__(best_threshold)
    #     except KeyError:
    #         # TODO PCA code!
    #         # TODO transform ALL categorical attributes to binary!
    #         # from sklearn.preprocessing import LabelBinarizer
    #
    #         def pull_left_by_purity(smaller):
    #             counter = Counter(smaller.apply(tuple, axis=1))
    #
    #             if len(counter) <= 1:
    #                 raise ValueError('partition is already pure!')
    #
    #             prob_matrix = pd.DataFrame(counter.keys(), columns=[node_label, self.target_attr])
    #             prob_matrix['prob'] = counter.values()
    #             prob_matrix = prob_matrix.sort_values(by=[subset.columns[-1], 'prob'], ascending=False)
    #
    #             all_splits = {}
    #
    #             for target in self.class_labels:
    #                 sub_matrix = prob_matrix.loc[prob_matrix[self.target_attr] == target]
    #
    #                 left_bag = sub_matrix[node_label].tolist()
    #                 right_bag = []
    #
    #                 entropies = []
    #
    #                 while len(left_bag) > 1:
    #                     argmax = sub_matrix['prob'].argmax()
    #                     picked = sub_matrix.loc[argmax, node_label]
    #                     right_bag += [picked]
    #                     left_bag.remove(picked)
    #                     sub_matrix = sub_matrix.loc[sub_matrix.index != argmax]  # removes picked row from sub_matrix
    #                     entropy = get_entropy(left_bag)
    #                     entropies += [entropy]
    #
    #                     all_splits[tuple(left_bag)] = entropy
    #
    #             inverse_dict = {x: y for y, x in all_splits.iteritems()}
    #             key = min(inverse_dict)
    #             return inverse_dict[key]
    #
    #         try:
    #             best_split = pull_left_by_purity(subset[[node_label, self.target_attr]])
    #             self.__store_threshold__(node_label, subset, best_split)
    #             meta, best_subset_left, best_subset_right = __subsets_and_meta__(best_split)
    #
    #         except ValueError:
    #             meta, best_subset_left, best_subset_right = self.__set_terminal__(
    #                 node_label=Individual.target_attr, parent_label=parent_label, subset=subset, **kwargs
    #             )
    #
    #     if 'get_meta' in kwargs and kwargs['get_meta'] == False:
    #         return best_subset_left, best_subset_right
    #     else:
    #         return meta, best_subset_left, best_subset_right

    def __set_categorical__(self, node_label, parent_labels, level, subset, node_id, **kwargs):
        # TODO enhance this method to perform smarter splits.
        # TODO currently it tries all combinations. what a mess!

        raise NotImplementedError('not implemented yet!')

        def __subsets_and_meta__(group):
            _best_subset_left = subset.loc[subset[node_label].apply(lambda x: x in group).index]
            _best_subset_right = subset.loc[subset[node_label].apply(lambda x: x not in group).index]

            _meta = {
                'label': node_label,
                'threshold': group,
                'terminal': False,
                'color': Individual._root_node_color if
                kwargs['variable_name'] == node.root else Individual._inner_node_color
            }

            return _meta, _best_subset_left, _best_subset_right

        def get_entropy(group):
            """
            Gets entropy for a given set of values.

            :type group: tuple
            :param group: Set of values.
            :rtype: float
            :return: the entropy.
            """

            subset_left = subset.loc[subset[node_label].apply(lambda x: x in group).index]
            subset_right = subset.loc[subset[node_label].apply(lambda x: x not in group).index]

            entropy = \
                Individual.entropy(subset_left, Individual.target_attr) + \
                Individual.entropy(subset_right, Individual.target_attr)

            return entropy

        try:
            best_threshold = self.__retrieve_threshold__(node_label, subset)
            meta, best_subset_left, best_subset_right = __subsets_and_meta__(best_threshold)
        except KeyError:
            groupby = Counter(subset[node_label])

            all_splits = []
            for i in xrange(1, len(groupby)):
                combs = list(it.combinations(groupby.keys(), i))  # order does not matter
                all_splits.extend(combs)
                # print 'x:', i, 'y:', len(all_splits)  # TODO error here! array gets too big!

            if len(all_splits) <= 1:
                meta, best_subset_left, best_subset_right = self.__set_terminal__(
                    node_label=Individual.target_attr, parent_labels=parent_label, subset=subset, **kwargs
                )
            else:
                entropies = map(get_entropy, all_splits)
                argmin_entropy = np.argmin(entropies)
                best_split = all_splits[argmin_entropy]

                self.__store_threshold__(node_label, subset, best_split)

                meta, best_subset_left, best_subset_right = __subsets_and_meta__(best_split)

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return best_subset_left, best_subset_right
        else:
            return meta, best_subset_left, best_subset_right

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
