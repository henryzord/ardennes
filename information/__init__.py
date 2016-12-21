import warnings
import numpy as np
from collections import Counter


class Handler(object):
    max_n_candidates = 1024

    def __init__(self, dataset):
        def get_number(x):
            class_label = x.axes[0][-1]

            new_label = np.float32(self.class_label_index[x[class_label]])
            x[class_label] = new_label
            return x

        self.target_attr = dataset.columns[-1]
        self.class_labels = np.array(dataset[dataset.columns[-1]].unique())
        self.numerical_class_labels = np.arange(self.class_labels.shape[0], dtype=np.float32)
        self.class_label_index = {k: x for x, k in enumerate(self.class_labels)}
        self.attribute_index = {k: x for x, k in enumerate(dataset.columns)}

        if dataset[dataset.columns[-1]].dtype != np.int32:
            dataset = dataset.apply(get_number, axis=1)

        dataset = dataset.astype(np.float32)

        self.dataset = dataset
        self.n_objects, self.n_attributes = dataset.shape

        try:
            from cuda import CudaMaster
            self.master = CudaMaster(dataset)
            self.max_n_candidates = self.master.MAX_N_THREADS
        except ImportError:
            self.master = None
            self.max_n_candidates = None

    def gain_ratio(self, subset, subset_left, subset_right, target_attr):
        ig = self.information_gain(subset, subset_left, subset_right, target_attr)
        si = self.__split_info__(subset, subset_left, subset_right)

        if ig > 0 and si > 0:
            gr = ig / si
        else:
            gr = 0
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
        if self.master is not None:
            return self.master.queue_execution(subset_index, attribute, candidates)
        else:
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


def __main__():
    from sklearn import datasets
    import pandas as pd

    dt = datasets.load_iris()
    df = pd.DataFrame(
        data=np.hstack((dt.data.astype(np.float32), dt.target[:, np.newaxis].astype(np.float32))),
        columns=np.hstack((dt.feature_names, 'class'))
    )

    handler = Handler(df)

    attr = df.columns[0]
    class_attribute = df.columns[-1]

    unique_vals = [float(x) for x in sorted(df[attr].unique())]

    candidates = np.array([
        (unique_vals[i] + unique_vals[i + 1]) / 2.
        if (i + 1) < len(unique_vals) else unique_vals[i] for i in xrange(len(unique_vals))
    ][:-1])

    subset_index = np.random.randint(2, size=df.shape[0])  # only a subset

    ratios = handler.batch_gain_ratio(subset_index, attr, candidates)

    for i, candidate in enumerate(candidates):
        _subset = df.loc[subset_index.astype(np.bool)]
        host_gain = handler.gain_ratio(
            _subset, _subset.loc[_subset[attr] < candidate],
            _subset.loc[_subset[attr] >= candidate],
            class_attribute
        )
        print 'seq/prl:', np.float32(host_gain), np.float32(ratios[i])
    print '-------------------'

if __name__ == '__main__':
    __main__()
