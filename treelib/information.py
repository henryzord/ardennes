import numpy as np


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

    from treelib.utils import MetaDataset
    dataset_info = MetaDataset(df)

    processor = Processor(df, dataset_info)
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
