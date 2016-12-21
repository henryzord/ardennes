# coding=utf-8

import collections
import copy
import itertools as it
import warnings
from collections import Counter

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from treelib.node import *

from sklearn.metrics import *

__author__ = 'Henry Cagnini'


class Individual(object):
    _terminal_node_color = '#98FB98'
    _inner_node_color = '#0099ff'
    _root_node_color = '#FFFFFF'
    column_types = None  # type: dict
    sets = None  # type: dict
    tree = None  # type: nx.DiGraph
    quality = None  # type: float
    ind_id = None

    thresholds = dict()
    max_height = -1

    handler = None

    n_objects = None
    n_attributes = None

    shortest_path = dict()  # type: dict

    height = None

    rtol = 1e-3

    def __init__(self, gm, max_height, sets, pred_attr, target_attr, class_labels, handler, **kwargs):
        """
        
        :type gm: treelib.graphical_model.GraphicalModel
        :param gm:
        :type sets: dict
        :param sets:
        """
        self.pred_attr = pred_attr
        self.target_attr = target_attr
        self.class_labels = class_labels

        if 'ind_id' in kwargs:
            self.ind_id = kwargs['ind_id']
        else:
            self.ind_id = None

        if Individual.handler is None:
            Individual.handler = handler
        self.handler = Individual.handler

        if Individual.column_types is None:
            Individual.column_types = {
                x: self.raw_type_dict[str(sets['train'][x].dtype)] for x in sets['train'].columns
                }  # type: dict
            Individual.column_types['class'] = 'class'
        self.column_types = Individual.column_types

        self.max_height = max_height

        self.sample(gm, sets['train'], sets['val'])

    @classmethod
    def clean(cls):
        cls.pred_attr = None
        cls.target_attr = None
        cls.class_labels = None
        cls.column_types = None

    @property
    def id_ind(self):
        return self.ind_id

    @property
    def inverse_height(self):
        height = self.height
        inv_height = self.max_height - height
        return inv_height

    @property
    def fitness(self):
        """
        :rtype: float
        :return: Fitness of this individual.
        """
        return self.quality

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
        parents = copy.deepcopy(self.shortest_path[node_id])
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

        return len(self.shortest_path[node_id]) - 1

    def __isclose__(self, other):
        quality_diff = abs(self.quality - other.quality)
        return quality_diff <= Individual.rtol

    def __le__(self, other):  # less or equal
        if self.__isclose__(other):
            return self.height <= other.height
        return self.quality < other.quality

    def __lt__(self, other):  # less than
        if self.__isclose__(other):
            return self.height < other.height
        else:
            return self.quality < other.quality

    def __ge__(self, other):  # greater or equal
        if self.__isclose__(other):
            return self.height >= other.height
        else:
            return self.quality >= other.quality

    def __gt__(self, other):  # greater than
        if self.__isclose__(other):
            return self.height > other.height
        else:
            return self.quality > other.quality

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
        return 'fitness: %0.2f height: %d' % (self.quality, self.height)

    def plot(self, savepath=None, test_set=None):
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
            'val accuracy: %0.4f' % self.quality,
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

        # plt.show()
        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight', format='pdf')
            plt.close()

    # ############################ #
    # sampling and related methods #
    # ############################ #

    def sample(self, gm, loc_train_set, loc_val_set):
        self.tree = self.tree = self.__set_node__(
            node_id=0,
            gm=gm,
            tree=nx.DiGraph(),
            subset=loc_train_set,
            level=0,
            parent_labels=[]
        )  # type: nx.DiGraph

        self.shortest_path = nx.shortest_path(self.tree, source=0)  # source equals to root

        y_true = loc_val_set[loc_val_set.columns[-1]]
        y_pred = self.predict(loc_val_set)

        acc_score = accuracy_score(y_true, y_pred)
        # _f1_score = f1_score(y_true, y_pred, average='micro')

        self.quality = acc_score
        self.height = max(map(len, self.shortest_path.itervalues()))

    def __set_node__(self, node_id, gm, tree, subset, level, parent_labels):
        """

        :param gm:
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
                subset.empty,  # empty subset
                subset[subset.columns[-1]].unique().shape[0] == 1,  # only one class
                level >= self.max_height,  # level deeper than maximum depth
                label == self.target_attr   # was sampled to be a class
        )):
            meta, subset_left, subset_right = self.__set_terminal__(
                node_label=None,
                parent_labels=parent_labels,
                node_level=level,
                subset=subset,
                node_id=node_id
            )
        else:
            # TODO try/except here for when a node must be a leaf!

            meta, subsets = self.__set_inner_node__(
                label=label,
                parent_labels=parent_labels,
                node_level=level,
                gm=gm,
                subset=subset,
                node_id=node_id
            )

            if meta['threshold'] is not None:
                children_id = [get_left_child(node_id), get_right_child(node_id)]

                for child_id, child_subset in it.izip(children_id, subsets):
                    tree = self.__set_node__(
                        node_id=child_id,
                        tree=tree,
                        gm=gm,
                        subset=child_subset,
                        level=level + 1,
                        parent_labels=parent_labels + [label]
                    )

                if all([tree.node[child_id]['label'] in self.class_labels for child_id in children_id]) \
                        and tree.node[children_id[0]]['label'] == tree.node[children_id[1]]['label']:
                    for child_id in children_id:
                        tree.remove_node(child_id)

                    meta, subset_left, subset_right = self.__set_terminal__(
                        node_label=None,
                        parent_labels=parent_labels,
                        node_level=level,
                        subset=subset,
                        node_id=node_id
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
            # if isinstance(node['threshold'], float):
            go_left = obj[node['label']] <= node['threshold']
            successors = tree.successors(arg_node)
            arg_node = (int(go_left) * min(successors)) + (int(not go_left) * max(successors))
            # elif isinstance(node['threshold'], collections.Iterable):
            #     raise StandardError('not valid!')
            #     edges = self.tree.edge[arg_node]
            #     neither_case = None
            #     was_set = False
            #     for v, d in edges.iteritems():
            #         if d['threshold'] == 'None':
            #             neither_case = v
            #
            #         if obj[node['label']] == d['threshold']:
            #             arg_node = v
            #             was_set = True
            #             break
            #
            #     # next node is the one which the category is neither one of the seen ones in the training phase
            #     if not was_set:
            #         arg_node = neither_case
            # else:
            #     raise TypeError('invalid type for threshold!')

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

    def __store_threshold__(self, node_label, parent_labels, threshold):
        """

        :type node_label: str
        :param node_label:
        :type subset: pandas.DataFrame
        :param subset:
        :param threshold:
        """
        # column_type = subset.dtypes[node_label]
        #
        # if column_type in [np.float32, np.float64, np.int32, np.int64]:
        #     _mean = subset[node_label].mean()
        #     _std = subset[node_label].std()
        # elif column_type == object:
        #     counts = subset[node_label].apply(len)
        #     _mean = counts.mean()
        #     _std = counts.std()
        # else:
        #     raise TypeError('invalid type for threshold! Encountered %s' % str(column_type))
        # key = '[%s][%05.8f][%05.8f]' % (str(node_label), _mean, _std)

        key = ','.join(parent_labels + [node_label])
        self.__class__.thresholds[key] = threshold

    def __retrieve_threshold__(self, node_label, parent_labels):
        """

        :param node_label:
        :type subset: pandas.DataFrame
        :param subset:
        :return:
        """

        # column_type = subset.dtypes[node_label]
        #
        # if column_type in [np.float32, np.float64, np.int32, np.int64]:
        #     _mean = subset[node_label].mean()
        #     _std = subset[node_label].std()
        # elif column_type == object:
        #     counts = subset[node_label].apply(len)
        #     _mean = counts.mean()
        #     _std = counts.std()
        # else:
        #     raise TypeError('invalid type for threshold! Encountered %s' % str(column_type))
        #
        # key = '[%s][%05.8f][%05.8f]' % (str(node_label), _mean, _std)
        key = ','.join(parent_labels + [node_label])
        return self.__class__.thresholds[key]

    def __set_numerical__(self, node_label, parent_labels, node_level, subset, node_id, **kwargs):
        try:
            best_threshold = self.__retrieve_threshold__(node_label, parent_labels)
            meta, subsets = self.__subsets_and_meta__(
                node_label, best_threshold, subset, node_id, node_level
            )
        except KeyError as ke:
            unique_vals = [float(x) for x in sorted(subset[node_label].unique())]

            if self.handler.max_n_candidates is None:
                candidates = np.array(unique_vals + [
                    (unique_vals[i] + unique_vals[i + 1]) / 2.
                    if (i + 1) < len(unique_vals) else unique_vals[i] for i in xrange(len(unique_vals))
                ][:-1])
            else:
                candidates = np.linspace(unique_vals[0], unique_vals[-1], self.handler.max_n_candidates)

            subset_index = np.zeros(self.handler.n_objects)
            subset_index[subset.index] = 1
            gains = self.handler.batch_gain_ratio(subset_index, node_label, candidates)

            argmax = np.argmax(gains)
            if gains[argmax] <= 0:
                meta, subset_left, subset_right = self.__set_terminal__(
                    node_label, parent_labels, node_level, subset, node_id
                )
                subsets = [subset_left, subset_right]
            else:
                best_threshold = candidates[argmax]

                # best_threshold = -np.inf
                # best_gr = -np.inf
                # for cand in candidates:
                #     gr = self.gain_ratio(
                #         subset,
                #         subset.loc[subset[node_label] < cand],
                #         subset.loc[subset[node_label] >= cand],
                #         self.target_attr
                #     )
                #     if gr > best_gr:
                #         best_gr = gr
                #         best_threshold = cand

                self.__store_threshold__(node_label, parent_labels, best_threshold)

                meta, subsets = self.__subsets_and_meta__(
                    node_label, best_threshold, subset, node_id, node_level
                )
        except Exception as e:
            raise e

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return subsets
        else:
            return meta, subsets

    def __set_terminal__(self, node_label, parent_labels, node_level, subset, node_id, **kwargs):
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
            'level': node_level,
            'node_id': node_id,
            'color': Individual._terminal_node_color
        }

        return meta, None, None  # pd.DataFrame([]), pd.DataFrame([])

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
            ops = [op.le, op.gt]  # <=, >
            subsets = [
                subset.loc[x(subset[node_label], threshold)] for x in ops
                ]

        return meta, subsets
