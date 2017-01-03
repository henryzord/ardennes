from device import AvailableDevice
import numpy as np


class Processor(object):
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

        self.device = AvailableDevice(dataset)

    def get_ratios(self, subset_index, attribute, candidates):
        ratios = self.device.device_gain_ratio(subset_index, attribute, candidates)
        return ratios


def __test_gain_ratio__():
    from sklearn import datasets
    import pandas as pd
    from device.cpu import CPUDevice

    dt = datasets.load_iris()
    df = pd.DataFrame(
        data=np.hstack((dt.data.astype(np.float32), dt.target[:, np.newaxis].astype(np.float32))),
        columns=np.hstack((dt.feature_names, 'class'))
    )

    attr = df.columns[0]
    class_attribute = df.columns[-1]

    unique_vals = [float(x) for x in sorted(df[attr].unique())]

    candidates = np.array([
        (unique_vals[i] + unique_vals[i + 1]) / 2.
        if (i + 1) < len(unique_vals) else unique_vals[i] for i in xrange(len(unique_vals))
    ][:-1])

    subset_index = np.random.randint(2, size=df.shape[0])  # only a subset

    processor = Processor(df)
    ratios = processor.get_ratios(subset_index, attr, candidates)

    for i, candidate in enumerate(candidates):
        _subset = df.loc[subset_index.astype(np.bool)]
        host_gain = CPUDevice.gain_ratio(
            _subset, _subset.loc[_subset[attr] < candidate],
            _subset.loc[_subset[attr] >= candidate],
            class_attribute
        )
        print 'seq/prl:', np.float32(host_gain), np.float32(ratios[i])
    print '-------------------'

if __name__ == '__main__':
    __test_gain_ratio__()
