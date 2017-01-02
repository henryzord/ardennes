from __base__ import Device

from collections import Counter
import numpy as np


class CPUDevice(Device):
    def __init__(self, dataset):
        super(CPUDevice, self).__init__(dataset)
        self.dataset = dataset

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

    @classmethod
    def information_gain(cls, subset, subset_left, subset_right, target_attr):
        sum_term = 0.
        for child_subset in [subset_left, subset_right]:
            sum_term += (child_subset.shape[0] / float(subset.shape[0])) * cls.entropy(child_subset, target_attr)

        ig = cls.entropy(subset, target_attr) - sum_term
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

    @classmethod
    def gain_ratio(cls, subset, subset_left, subset_right, target_attr):
        ig = cls.information_gain(subset, subset_left, subset_right, target_attr)
        si = cls.__split_info__(subset, subset_left, subset_right)

        if ig > 0 and si > 0:
            gr = ig / si
        else:
            gr = 0
        return gr

    def device_gain_ratio(self, subset_index, attribute, candidates):
        proper_subset = self.dataset.loc[subset_index.astype(np.bool)]

        ratios = map(
            lambda c: self.gain_ratio(
                proper_subset,
                proper_subset.loc[proper_subset[attribute] <= c],
                proper_subset.loc[proper_subset[attribute] > c],
                self.target_attr
            ),
            candidates
        )
        return ratios
