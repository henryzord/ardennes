# coding=utf-8

import collections
import copy
import itertools as it
import json
from collections import Counter
from sklearn.metrics import *
from treelib.node import *
import networkx as nx
import pandas as pd

__author__ = 'Henry Cagnini'


class DecisionTree(object):
    _terminal_node_color = '#98FB98'
    _inner_node_color = '#0099ff'
    _root_node_color = '#FFFFFF'

    dataset = None
    dataset_info = None

    max_height = -1

    thresholds = dict()  # thresholds for nodes

    arg_sets = None

    y_test_true = None
    y_val_true = None
    y_train_true = None

    _shortest_path = dict()  # type: dict
    tree = None  # type: nx.DiGraph

    fitness = None  # type: float
    height = None
    n_nodes = None

    mdevice = None

    train_acc_score = None
    val_acc_score = None
    test_acc_score = None

    def __init__(self, gm, **kwargs):
        self.sample(gm)

    @classmethod
    def set_values(cls, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(cls, k, v)

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
                self.tree.node[p]['label'] not in DecisionTree.dataset_info.class_labels else
                DecisionTree.dataset_info.target_attr
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

    def sample(self, gm):
        arg_threshold = DecisionTree.arg_sets['train']

        self.tree = self.tree = self.__set_node__(
            node_id=0,
            gm=gm,
            tree=nx.DiGraph(),
            subset_index=arg_threshold,
            depth=0,
            parent_labels=[],
            coordinates=[],
        )  # type: nx.DiGraph

        self._shortest_path = nx.shortest_path(self.tree, source=0)  # source equals to root

        predictions = np.array(self.mdevice.predict(self.mdevice.dataset, self, inner=True))

        self.train_acc_score = accuracy_score(DecisionTree.y_train_true, predictions[self.arg_sets['train']])
        self.val_acc_score = accuracy_score(DecisionTree.y_val_true, predictions[self.arg_sets['val']])
        self.test_acc_score = accuracy_score(DecisionTree.y_test_true, predictions[self.arg_sets['test']])

        self.fitness = self.train_acc_score

        self.height = max(map(len, self._shortest_path.itervalues()))
        self.n_nodes = len(self.tree.node)

    def predict(self, samples):
        return self.mdevice.predict(samples, self, inner=False)

    def to_matrix(self):
        """
        Converts the inner decision tree from this class to a matrix with n_nodes
        rows and [Left, Right, Attribute, Threshold, Terminal] attributes.
        """

        tree = self.tree

        matrix = pd.DataFrame(
            index=tree.node.keys(),
            columns=['left', 'right', 'attribute', 'threshold', 'terminal']
        )

        conv_dict = {k: i for i, k in enumerate(matrix.index)}

        for node_id, node in tree.node.iteritems():
            matrix.loc[node_id] = [
                conv_dict[get_left_child(node['node_id'])] if get_left_child(node_id) in conv_dict else None,
                conv_dict[get_right_child(node['node_id'])] if get_right_child(node_id) in conv_dict else None,
                self.dataset_info.attribute_index[node['label']] if node['label'] not in self.dataset_info.class_labels
                    else self.dataset_info.class_label_index[node['label']],
                node['threshold'], node['terminal']
            ]

        matrix.index = [conv_dict[x] for x in matrix.index]
        matrix = matrix.astype(np.float32)

        return matrix

    def __set_node__(self, node_id, gm, tree, subset_index, depth, parent_labels, coordinates):
        try:
            label = gm.observe(node_id=node_id)
        except KeyError as ke:
            if depth >= gm.D:
                label = DecisionTree.dataset_info.target_attr
            else:
                # TODO sampling a node with 100% of being class!
                print gm.attributes
                raise KeyError('Node %s not in graphical model!' % ke.message)

        '''
        if the current node has only one class,
        or is at a level deeper than maximum depth,
        or the class was sampled
        '''
        if any((
                    DecisionTree.dataset.loc[subset_index, DecisionTree.dataset_info.target_attr].unique().shape[0] == 1,
                    depth >= DecisionTree.max_height,
                    label == DecisionTree.dataset_info.target_attr
        )):
            meta, subsets = self.__set_terminal__(
                node_label=None,
                node_id=node_id,
                node_level=depth,
                subset_index=subset_index,
                parent_labels=parent_labels,
                coordinates=coordinates
            )
        else:
            meta, subsets = self.__set_inner_node__(
                node_label=label,
                parent_labels=parent_labels,
                coordinates=coordinates,
                node_level=depth,
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
                        depth=depth + 1,
                        parent_labels=parent_labels + [label],
                        coordinates=coordinates + [c]
                    )

                if all([tree.node[child_id]['label'] in DecisionTree.dataset_info.class_labels for child_id in children_id]) \
                        and tree.node[children_id[0]]['label'] == tree.node[children_id[1]]['label']:
                    for child_id in children_id:
                        tree.remove_node(child_id)

                    meta, subsets = self.__set_terminal__(
                        node_label=None,
                        node_id=node_id,
                        node_level=depth,
                        subset_index=subset_index,
                        parent_labels=parent_labels,
                        coordinates=coordinates
                    )

                else:
                    if type(meta['threshold']) in [np.float32, np.float64, float]:  # TODO use raw_type_dict
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

    def __set_inner_node__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        attr_type = DecisionTree.dataset_info.column_types[node_label]

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
            unique_vals = sorted(DecisionTree.dataset.loc[subset_index, node_label].unique())

            if len(unique_vals) > 1:

                candidates = np.array(
                    [(a + b) / 2. for a, b in it.izip(unique_vals[::2], unique_vals[1::2])], dtype=np.float32
                )
                gains = self.mdevice.get_gain_ratios(subset_index, node_label, candidates)

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
            else:
                meta, subsets = self.__set_terminal__(
                    node_label=node_label,
                    node_id=node_id,
                    node_level=node_level,
                    subset_index=subset_index,
                    parent_labels=parent_labels,
                    coordinates=coordinates,
                    **kwargs
                )
        except Exception as e:
            raise e

        if 'get_meta' in kwargs and kwargs['get_meta'] == False:
            return subsets
        else:
            return meta, subsets

    def __set_terminal__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        # node_label in this case is probably the DecisionTree.target_attr; so it
        # is not significant for the **real** label of the terminal node.

        counter = Counter(DecisionTree.dataset.loc[subset_index, DecisionTree.dataset_info.target_attr])
        label, count_frequent = counter.most_common()[0]

        meta = {
            'label': label,
            'threshold': None,
            'terminal': True,
            'inst_correct': count_frequent,
            'inst_total': subset_index.sum(),
            'level': node_level,
            'node_id': node_id,
            'color': DecisionTree._terminal_node_color
        }

        return meta, (None, None)

    def __set_categorical__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        raise NotImplementedError('not implemented yet!')

    @staticmethod
    def __set_error__(self, node_label, node_id, node_level, subset_index, parent_labels, coordinates, **kwargs):
        raise TypeError('Unsupported data type for column %s!' % node_label)

    @staticmethod
    def __subsets_and_meta__(node_label, node_id, node_level, subset_index, threshold):

        counter = Counter(DecisionTree.dataset.loc[subset_index, DecisionTree.dataset_info.target_attr])

        meta = {
            'label': node_label,
            'threshold': threshold,
            'inst_correct': counter.most_common()[0][1],
            'inst_total': subset_index.sum(),
            'terminal': False,
            'level': node_level,
            'node_id': node_id,
            'color': DecisionTree._root_node_color if
            node_level == 0 else DecisionTree._inner_node_color
        }

        less_or_equal = (DecisionTree.dataset[node_label] <= threshold).values.ravel()
        subset_left = less_or_equal & subset_index
        subset_right = np.invert(less_or_equal) & subset_index

        return meta, (subset_left, subset_right)

    handler_dict = {
        'object': __set_categorical__,
        'str': __set_categorical__,
        'int': __set_numerical__,
        'float': __set_numerical__,
        'bool': __set_categorical__,
        'complex': __set_error__,
    }

    def to_dict(self):
        edges = nx.to_dict_of_dicts(self.tree)
        nodes = self.tree.node
        j = dict(edges=edges, nodes=nodes)
        return j

    def from_json(self, json_string):
        j = json.loads(json_string)
        raise NotImplementedError('not implemented yet!')

    def to_json(self):
        j = self.to_dict()

        for k in j['nodes'].iterkeys():
            j['nodes'][k]['threshold'] = str(j['nodes'][k]['threshold'])

        _str = json.dumps(j, ensure_ascii=False, indent=2)
        return _str
