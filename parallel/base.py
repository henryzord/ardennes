import warnings
import numpy as np
from collections import Counter


class Handler(object):
    def __init__(self, dataset):
        def get_number(x):
            try:
                return np.float32(self.class_label_index[x])
            except KeyError:
                return np.float32(self.numerical_class_labels[-1])

        self.target_attr = dataset.columns[-1]
        self.class_labels = np.array(dataset[dataset.columns[-1]].unique())
        self.numerical_class_labels = np.arange(self.class_labels.shape[0], dtype=np.float32)
        self.class_label_index = {k: x for x, k in enumerate(self.class_labels)}
        self.attribute_index = {k: x for x, k in enumerate(dataset.columns)}

        if dataset[dataset.columns[-1]].dtype != np.int32:
            dataset[dataset.columns[-1]] = dataset[dataset.columns[-1]].apply(get_number)

        for column in dataset.columns:
            dataset[column] = dataset[column].astype(np.float32)

        self.dataset = dataset

    def gain_ratio(self, subset, subset_left, subset_right, target_attr):
        warnings.filterwarnings('error')

        ig = self.information_gain(subset, subset_left, subset_right, target_attr)
        si = self.__split_info__(subset, subset_left, subset_right)

        try:
            gr = ig / si
        except RuntimeWarning as rw:
            if si <= 0:
                gr = 0.
            else:
                raise rw

        warnings.filterwarnings('default')

        return gr

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

    def information_gain(self, subset, subset_left, subset_right, target_attr):
        sum_term = 0.
        for child_subset in [subset_left, subset_right]:
            sum_term += (child_subset.shape[0] / float(subset.shape[0])) * self.entropy(child_subset, target_attr)

        ig = self.entropy(subset, target_attr) - sum_term
        return ig

    @staticmethod
    def entropy(subset, target_attr):
        # the smaller, the better
        size = float(subset.shape[0])

        counter = Counter(subset[target_attr])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy

    def batch_gain_ratio(self, subset_index, attribute, candidates):
        pass
