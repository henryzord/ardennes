import numpy as np

try:
    # raise ImportError('hue')
    from cuda import CudaHandler as AvailableHandler
except ImportError as e:
    from sequential import SequentialHandler as AvailableHandler


def __main__():
    from sklearn import datasets
    import pandas as pd
    from sequential import SequentialHandler
    from cuda import CudaHandler

    dt = datasets.load_iris()
    df = pd.DataFrame(
        data=np.hstack((dt.data.astype(np.float32), dt.target[:, np.newaxis].astype(np.float32))),
        columns=np.hstack((dt.feature_names, 'class'))
    )

    cuda = CudaHandler(df)
    seq = SequentialHandler(df)

    attr = df.columns[0]
    class_attribute = df.columns[-1]

    unique_vals = [float(x) for x in sorted(df[attr].unique())]

    candidates = np.array([
        (unique_vals[i] + unique_vals[i + 1]) / 2.
        if (i + 1) < len(unique_vals) else unique_vals[i] for i in xrange(len(unique_vals))
    ][:-1])

    subset_index = np.random.randint(2, size=df.shape[0])  # only a subset
    # subset_index = np.ones(df.shape[0])  # full dataset

    seq_ratios = seq.batch_gain_ratio(subset_index, attr, candidates)
    cuda_ratios = cuda.batch_gain_ratio(subset_index, attr, candidates)

    for i, candidate in enumerate(candidates):
        # _subset = df.loc[subset_index.astype(np.bool)]
        # host_gain = seq.gain_ratio(_subset, _subset.loc[_subset[attr] < candidate], _subset.loc[_subset[attr] >= candidate], class_attribute)
        print 'seq/prl:', np.float32(seq_ratios[i]), np.float32(cuda_ratios[i])

if __name__ == '__main__':
    __main__()
