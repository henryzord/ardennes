# coding=utf-8

import json
from collections import Counter

import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from cpu_device import gain_ratio, make_predictions

from utils import raw_type_dict, mid_type_dict, clean_dataset

__author__ = 'Henry Cagnini'


class HeapTree(object):

    def __init__(self, max_depth=1):
        pass

    @staticmethod
    def get_node_count(depth):
        """
        Get number of total nodes from a tree with given depth.

        :param depth: Depth of binary tree.
        :type depth: int
        :rtype: int
        :return: The number of nodes in this tree.
        """
        return np.power(2, depth + 1) - 1

    @staticmethod
    def node_count_at(depth):
        """
        Get number of nodes at given level.

        :type depth: int
        :param depth: The querying depth. Starts at zero (i.e. root).
        :rtype: int
        :return: Number of nodes at queried depth.
        """

        return np.power(2, depth)

    @staticmethod
    def get_depth(id_node):
        """
        Gets depth of node in a binary heap.

        :param id_node: ID of the node in the binary heap.
        :return: The depth of the node.
        """
        return int(np.log2(id_node + 1))

    @staticmethod
    def get_left_child_id(id_node):
        return (id_node * 2) + 1

    @staticmethod
    def get_right_child_id(id_node):
        return (id_node * 2) + 2

    @staticmethod
    def get_parent_id(id_node):
        if id_node > 0:
            return int((id_node - 1) / 2.)
        return None


class DecisionTree(HeapTree):
    _terminal_node_color = '#98FB98'
    _inner_node_color = '#0099ff'
    _root_node_color = '#FFFFFF'

    initialized = False

    @classmethod
    def __set_class_values__(cls, max_depth, full_df, train_index, val_index):
        if not cls.initialized:
            cls.max_depth = max_depth
            cls.train_index = train_index
            cls.val_index = val_index

            cls.column_types = {x: mid_type_dict[raw_type_dict[str(full_df[x].dtype)]] for x in full_df.columns}

            cls.pred_attr_names = np.array(full_df.columns[:-1])  # type: np.ndarray
            cls.class_attr_name = full_df.columns[-1]  # type: str

            cls.y_train_true = full_df.loc[train_index, cls.class_attr_name]
            cls.y_val_true = full_df.loc[val_index, cls.class_attr_name]

            _attr_forward = {
                k: v for k, v in zip(full_df.columns, range(len(cls.pred_attr_names) + 1))
            }
            _attr_backward = {
                k: v for k, v in zip(range(len(cls.pred_attr_names) + 1), full_df.columns)
            }
            _attr_forward.update(_attr_backward)
            cls.attr_index = _attr_forward

            cls.full_df, cls.attr_values = cls.__normalize_dataset__(full_df)

            # since the probability of generating the class at D is 100%
            cls.n_nodes = cls.get_node_count(cls.max_depth - 1)

            cls.stored_threshold = dict()

            cls.initialized = True

    def __init__(self, max_depth, full_df, train_index, val_index):
        """

        :param max_depth:
        :param full_df:
        :param train_index:
        :param val_index:
        """
        super(DecisionTree, self).__init__(max_depth=max_depth)

        self.__class__.__set_class_values__(
            max_depth=max_depth,
            full_df=full_df,
            train_index=train_index,
            val_index=val_index
        )

        longest_word = ''
        for attr_name, attr_values in self.attr_values.items():
            for k, v in attr_values.items():
                longest_word = max(longest_word, str(v))

        longest_word_length = max(map(len, [longest_word] + list(self.full_df.columns)))
        self.nodes_dtype = '<U%d' % (longest_word_length + 1)  # +1 for NULL character

        self.nodes = np.empty(self.n_nodes, dtype=np.dtype(self.nodes_dtype))
        self.threshold = np.empty(self.n_nodes, dtype=np.float32)
        self.terminal = np.empty(self.n_nodes, dtype=np.bool)
        self.predictions = np.empty(len(self.full_df), dtype=self.nodes_dtype)

        self.train_acc_score = None
        self.val_acc_score = None

        self.fitness = None
        self.quality = None
        self.height = None
        self.n_nodes = None

    @staticmethod
    def __normalize_dataset__(dataset):
        """
        Converts the dataset to a format comprehensible to this class.
        Numeric attributes will be converted to float32, and categorical
        ones will have the labels replaced by their numeric codes.

        :param dataset: The dataset to be normalized.
        :type dataset: pandas.DataFrame
        :rtype: tuple
        :return: a tuple where the first element is the normalized dataset (pandas.DataFrame),
        and the second a dictionary of attribute values (dict).
        """
        dataset = clean_dataset(dataset)
        attr_values = dict()

        for column_index, column_name in enumerate(dataset.columns):
            if str(dataset[column_name].dtype) == 'category':
                zipped = list(zip(
                    dataset[column_name].cat.categories, range(len(dataset[column_name].cat.categories))
                ))
                attr_values[column_name] = dict(zipped)
                attr_values[column_name].update(dict(map(lambda x: reversed(x), zipped)))

                attr_values[column_index] = {v: k for k, v in attr_values[column_name].items()}
                dataset[column_name] = dataset[column_name].cat.codes.astype(np.float32)

        return dataset, attr_values

    def __update_after_sampling__(self):
        arg_threshold = self.train_index + self.val_index

        self.__set_node__(
            node_id=0,
            subset_index=arg_threshold,
            parent_labels=[],
            coordinates=[],
        )
        predictions = self.predict(full_df=self.full_df, predictions=self.predictions)

        self.train_acc_score = accuracy_score(self.y_train_true, predictions[self.train_index])
        self.val_acc_score = accuracy_score(self.y_val_true, predictions[self.val_index])

        self.fitness = self.train_acc_score
        self.quality = 0.5 * (self.train_acc_score + self.val_acc_score)

        self.height = self.get_height()
        self.n_nodes = self.get_n_nodes()

    def get_height(self):
        height = 1
        for i in reversed(range(len(self.terminal))):
            if self.terminal[i]:
                height += self.get_depth(i)
                break

        return height

    def get_n_nodes(self):
        return len(self.threshold) - np.sum(np.isnan(self.threshold)) + np.sum(self.terminal)

    def __predict__(self, full_df, predictions):

        n_predictions = len(full_df)
        for i in range(n_predictions):
            current_node = 0
            while True:
                if self.terminal[current_node]:
                    predictions[i] = self.nodes[current_node]
                    break
                else:
                    pass
                    label = self.nodes[current_node]
                    threshold = self.threshold[current_node]
                    value = full_df.iloc[i][label]

                    go_left = value <= threshold
                    if go_left:
                        current_node = self.get_left_child_id(current_node)
                    else:
                        current_node = self.get_right_child_id(current_node)

        # preds = make_predictions(
        #     full_df.values, self.attr_index,
        #     self.nodes, self.threshold, self.terminal,
        #     predictions
        # )
        return predictions

    def predict(self, full_df, predictions=None):
        """

        :param full_df:
        :type full_df: pandas.DataFrame
        :param predictions: (optional) - Former predictions array, if any. Used internally to save memory allocation.
        :return:
        """
        if predictions is None:
            predictions = np.empty(len(full_df), dtype=self.nodes_dtype)

        return self.__predict__(full_df, predictions)

    def __same_branches__(self, children_left, children_right):
        res = (self.terminal[children_left] and self.terminal[children_right]) and \
               (self.nodes[children_left] == self.nodes[children_right])

        return res

    def __set_node__(self, node_id, subset_index, parent_labels, coordinates):

        label = self.nodes[node_id]

        '''
        if the current node has only one class,
        or is at a level deeper than maximum depth,
        or the class was sampled
        '''
        if any((
            self.get_depth(node_id) >= self.max_depth - 1,
            label == self.class_attr_name,
            len(self.full_df.loc[subset_index, self.class_attr_name].unique()) == 1
        )):
            subset_left, subset_right = self.__set_terminal__(
                node_label=None,
                node_id=node_id,
                subset_index=subset_index,
                parent_labels=parent_labels,
                coordinates=coordinates
            )
        else:
            subset_left, subset_right = self.__set_inner_node__(
                node_label=label,
                node_id=node_id,
                subset_index=subset_index,
                parent_labels=parent_labels,
                coordinates=coordinates,
            )

            if not self.terminal[node_id]:  # continues the process
                children_id = (self.get_left_child_id(node_id), self.get_right_child_id(node_id))
                for c, child_id, child_subset in zip(range(len(children_id)), children_id, [subset_left, subset_right]):
                    self.__set_node__(
                        node_id=child_id,
                        subset_index=child_subset,
                        parent_labels=parent_labels + [label],
                        coordinates=coordinates + [c]
                    )

                # if both branches are terminal, and they share the same class label:
                if self.__same_branches__(children_id[0], children_id[1]):
                    _ = self.__set_terminal__(
                        node_label=label,
                        node_id=node_id,
                        subset_index=subset_index,
                        parent_labels=parent_labels,
                        coordinates=coordinates
                    )

    def __set_inner_node__(self, node_label, node_id, subset_index, parent_labels, coordinates):
        attr_type = self.column_types[node_label]

        subset_left, subset_right = self.handler_dict[attr_type](
            self,
            node_label=node_label,
            node_id=node_id,
            subset_index=subset_index,
            parent_labels=parent_labels,
            coordinates=coordinates
        )

        if self.terminal[node_id]:
            subset_left, subset_right = self.__set_terminal__(
                node_label=None,
                node_id=node_id,
                subset_index=subset_index,
                parent_labels=parent_labels,
                coordinates=coordinates
            )
        elif subset_left.sum() == 0 or subset_right.sum() == 0:  # it is not a terminal, but has no instances to split
            subset_left, subset_right = self.__set_terminal__(
                node_label=None,
                node_id=node_id,
                subset_index=subset_index,
                parent_labels=parent_labels,
                coordinates=coordinates
            )

        return subset_left, subset_right

    @classmethod
    def __store_threshold__(cls, node_label, parent_labels, coordinates, threshold):
        key = '[' + ','.join(parent_labels + [node_label]) + '][' + ','.join([str(c) for c in coordinates]) + ']'
        cls.stored_threshold[key] = threshold

    @classmethod
    def __retrieve_threshold__(cls, node_label, parent_labels, coordinates):
        key = '[' + ','.join(parent_labels + [node_label]) + '][' + ','.join([str(c) for c in coordinates]) + ']'
        return cls.stored_threshold[key]

    def __set_numerical__(self, node_label, node_id, subset_index, parent_labels, coordinates):
        try:
            best_threshold = self.__retrieve_threshold__(
                node_label=node_label,
                parent_labels=parent_labels,
                coordinates=coordinates
            )
        except KeyError as ke:
            unique_vals = sorted(self.full_df.loc[subset_index, node_label].unique())

            if len(unique_vals) <= 1:
                return self.__set_terminal__(
                    node_label=node_label,
                    node_id=node_id,
                    subset_index=subset_index,
                    parent_labels=parent_labels,
                    coordinates=coordinates,
                )

            candidates = np.array(
                [(a + b) / 2. for a, b in zip(unique_vals[::2], unique_vals[1::2])], dtype=np.float32
            )
            gains = gain_ratio(
                dataset=self.full_df.values,
                subset_index=subset_index,
                attribute_index=self.attr_index[node_label],
                candidates=candidates,
                n_classes=len(self.attr_values[self.class_attr_name])
            )

            argmax = np.argmax(gains)
            best_gain = gains[argmax]

            if best_gain <= 0:
                return self.__set_terminal__(
                    node_label=node_label,
                    node_id=node_id,
                    subset_index=subset_index,
                    parent_labels=parent_labels,
                    coordinates=coordinates
                )
            else:
                best_threshold = candidates[argmax]

                self.__class__.__store_threshold__(
                    node_label=node_label,
                    parent_labels=parent_labels,
                    coordinates=coordinates,
                    threshold=best_threshold
                )

        self.terminal[node_id] = False
        self.nodes[node_id] = node_label
        self.threshold[node_id] = best_threshold

        less_or_equal = (self.full_df[node_label] <= best_threshold).values.ravel()
        subset_left = less_or_equal & subset_index
        subset_right = np.invert(less_or_equal) & subset_index

        return subset_left, subset_right

    def __set_terminal__(self, node_label, node_id, subset_index, parent_labels, coordinates):
        # node_label in this case is probably the DecisionTree.target_attr; so it
        # is not significant for the **real** label of the terminal node.

        counter = Counter(
            map(lambda x: self.attr_values[self.class_attr_name][x], self.full_df.loc[subset_index, self.class_attr_name])
        )
        label, count_frequent = counter.most_common()[0]

        self.nodes[node_id] = label
        self.threshold[node_id] = np.nan
        self.terminal[node_id] = True

        return None, None

    def __set_categorical__(self, node_label, node_id, subset_index, parent_labels, coordinates):
        raise TypeError('Unsupported data type for column %s!' % node_label)

    @staticmethod
    def __set_error__(self, node_label, node_id, subset_index, parent_labels, coordinates):
        raise TypeError('Unsupported data type for column %s!' % node_label)

    handler_dict = {
        'numerical': __set_numerical__,
        'categorical': __set_categorical__
    }

    def to_networkx(self):
        def __add_local__(tree, current_node, graph):
            if tree.terminal[current_node]:
                graph.add_node(current_node, label=tree.nodes[current_node], color=self._terminal_node_color)
            else:
                graph.add_node(
                    current_node,
                    label=tree.nodes[current_node],
                    color=self._inner_node_color
                )

                left_child = tree.get_left_child_id(current_node)
                right_child = tree.get_right_child_id(current_node)
                graph = __add_local__(tree=tree, current_node=left_child, graph=graph)
                graph = __add_local__(tree=tree, current_node=right_child, graph=graph)
                graph.add_edge(current_node, left_child, threshold='<= %.2f' % tree.threshold[current_node])
                graph.add_edge(current_node, right_child, threshold='> %.2f' % tree.threshold[current_node])

            return graph

        _tree = __add_local__(tree=self, current_node=0, graph=nx.Graph())
        return _tree

    def to_matrix(self):
        """
        Converts the inner decision tree from this class to a matrix with n_nodes
        rows and [Left, Right, Terminal, Attribute, Threshold] attributes.
        """
        raise NotImplementedError('not implemented yet!')

    def to_dict(self):
        raise NotImplementedError('not implemented yet!')

    def from_json(self, json_string):
        raise NotImplementedError('not implemented yet!')

    def to_json(self):
        raise NotImplementedError('not implemented yet!')
