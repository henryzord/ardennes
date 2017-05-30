import os
import numpy as np
import pandas as pd
from collections import Counter
from c_individual import make_predictions


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

    def predict(self, data, dt, inner=False):
        """

        Makes predictions for unseen samples.

        :param data:
        :type dt: treelib.individual.DecisionTree
        :param dt:
        :type inner: bool
        :param inner:
        :rtype: pandas.DataFrame
        :return:
        """

        tree = dt.tree.node

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
            range(shape[0]),  # shape of prediction array
            self.dataset_info.attribute_index,  # dictionary where keys are attributes and values their indices
            dt.multi_tests  # number of tests per node
        )

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

    def entropy(self, subset):
        """
        The smaller, the purer.

        :type subset: pandas.DataFrame
        :param subset:
        :rtype: float
        :return:
        """
        size = float(subset.shape[0])

        counter = Counter(subset[self.dataset_info.target_attr])

        _entropy = 0.

        for i, (c, q) in enumerate(counter.iteritems()):
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


def __test_gain_ratio__():
    from sklearn import datasets
    from treelib.utils import MetaDataset
    from opencl import CLDevice
    from termcolor import colored
    import itertools as it

    dt = datasets.load_iris()
    df = pd.DataFrame(
        data=np.hstack((dt.data.astype(np.float32), dt.target[:, np.newaxis].astype(np.float32))),
        columns=np.hstack((dt.feature_names, 'class'))
    )

    attr = df.columns[0]  # first attribute

    unique_vals = [float(x) for x in sorted(df[attr].unique())]

    candidates = np.array(
        [(a + b) / 2. for a, b in it.izip(unique_vals[::2], unique_vals[1::2])], dtype=np.float32
    )

    subset_index = np.random.randint(2, size=df.shape[0])  # only a subset

    dataset_info = MetaDataset(df)

    cldevice = CLDevice(df, dataset_info)
    cpudevice = Device(df, dataset_info)

    ratios = cldevice.get_gain_ratios(subset_index, attr, candidates)

    for i, candidate in enumerate(candidates):
        _subset = df.loc[subset_index.astype(np.bool)]
        host_gain = cpudevice.gain_ratio(
            _subset, _subset.loc[_subset[attr] < candidate],
            _subset.loc[_subset[attr] >= candidate]
        )
        print colored(
            'seq: %.8f prl: %.8f' % (np.float32(host_gain), np.float32(ratios[i])),
            'red' if abs(np.float32(host_gain) - np.float32(ratios[i])) >= 1e-6 else 'blue'
        )
    print '-------------------'

if __name__ == '__main__':
    __test_gain_ratio__()
