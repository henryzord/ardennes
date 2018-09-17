from termcolor import colored

import os
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx

# noinspection PyUnresolvedReferences
from cpu_device import make_predictions


class Device(object):
    def __init__(self, dataset, dataset_info):
        """

        :type dataset: pandas.DataFrame
        :param dataset:
        :type dataset_info: treelib.utils.MetaDataset
        :param dataset_info:
        """

        self.dataset_info = dataset_info

        sep = '\\' if os.name == 'nt' else '/'

        cur_path = os.path.abspath(__file__)
        self._split = sep.join(cur_path.split(sep)[:-1])

        self.dataset_info = dataset_info
        self.dataset = dataset.apply(self.__class_to_num__, axis=1).astype(np.float32)

    def __class_to_num__(self, x):
        class_label = x.axes[0][-1]

        new_label = np.float32(self.dataset_info.class_label_index[x[class_label]])
        x[class_label] = new_label
        return x

    def get_gain_ratios(self, subset_index, attribute, candidates):
        pass

    def predict(self, data, dt):
        """

        Makes predictions for unseen samples.

        :param data:
        :type dt: individual.DecisionTree
        :param dt:
        :rtype: pandas.DataFrame
        :return:
        """

        tree = dt.tree._node

        shape = None
        if isinstance(data, pd.DataFrame):
            shape = data.shape
            data = data.values.ravel().tolist()
        elif isinstance(data, np.ndarray):
            shape = data.shape
            data = data.ravel().tolist()
        elif isinstance(data, list):
            n_instances = len(data)
            n_attributes = len(data[0])
            shape = (n_instances, n_attributes)
            data = np.array(data).ravel().tolist()  # removes extra dimensions

        preds = make_predictions(
            shape,  # shape of sample data
            data,  # dataset as a plain list
            tree,  # tree in dictionary format
            list(range(shape[0])),  # shape of prediction array
            self.dataset_info.attribute_index  # dictionary where keys are attributes and values their indices
        )

        raise NotImplementedError('not returning correct values!')

        return preds

    @staticmethod
    def __split_info__(subset, subset_left, subset_right):
        split_info = 0.
        for child_subset in [subset_left, subset_right]:
            if child_subset.shape[0] <= 0 or subset.shape[0] <= 0:
                pass
            else:
                share = (child_subset.shape[0] / float(subset.shape[0]))
                split_info += share * np.log2(share)

        return -split_info

    def information_gain(self, subset, subset_left, subset_right):
        sum_term = 0.
        for child_subset in [subset_left, subset_right]:
            sum_term += (child_subset.shape[0] / float(subset.shape[0])) * self.entropy(child_subset)

        ig = self.entropy(subset) - sum_term
        return ig

    @staticmethod
    def entropy(subset):
        """
        The smaller, the purer.
        Class attribute must be the last attribute.

        :type subset: pandas.DataFrame
        :param subset:
        :rtype: float
        :return:
        """
        size = float(subset.shape[0])

        counter = Counter(subset[subset.columns[-1]])

        _entropy = 0.

        for i, (c, q) in enumerate(counter.items()):
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy

    def gain_ratio(self, subset, subset_left, subset_right):
        ig = self.information_gain(subset, subset_left, subset_right)
        si = self.__split_info__(subset, subset_left, subset_right)

        if ig > 0 and si > 0:
            gr = ig / si
        else:
            gr = 0

        return gr

    def get_gain_ratios(self, subset_index, attribute, candidates):
        proper_subset = self.dataset.loc[subset_index.astype(np.bool)]

        ratios = map(
            lambda c: self.gain_ratio(
                proper_subset,
                proper_subset.loc[proper_subset[attribute] <= c],
                proper_subset.loc[proper_subset[attribute] > c],
            ),
            candidates
        )
        return ratios


try:
    # noinspection PyUnresolvedReferences
    import pyopencl
    from opencl import CLDevice as AvailableDevice
    print(colored('NOTICE: Using OpenCL as device.', 'yellow'))
except ImportError:
    AvailableDevice = Device
    print(colored('NOTICE: Using single-threaded CPU as device.', 'yellow'))

# from termcolor import colored
# from __base__ import  Device as AvailableDevice
# print colored('NOTICE: Using single-threaded CPU as device.', 'yellow')
