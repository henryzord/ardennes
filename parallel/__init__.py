import warnings
from collections import Counter

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import os
import pandas as pd
import numpy as np


class CudaHandler(object):
    _MIN_N_THREADS = 32
    _MAX_N_THREADS = 1024
    _N_OUTPUT = 3

    def __init__(self, dataset):
        """

        :type dataset: pandas.DataFrame
        :param dataset:
        """

        sep = '\\' if os.name == 'nt' else '/'

        cur_path = os.path.abspath(__file__)
        split = '/'.join(cur_path.split(sep)[:-1])

        kernel = open(os.path.join(split, 'kernel.cu'), 'r').read()
        mod = SourceModule(source=kernel)

        self._func_gain_ratio = mod.get_function("gain_ratio")

        self.class_labels = np.array(dataset[dataset.columns[-1]].unique())

        self._mem_dataset = cuda.mem_alloc(dataset.values.nbytes)
        self._mem_class_labels = cuda.mem_alloc(self.class_labels.nbytes)
        cuda.memcpy_htod(self._mem_dataset, dataset.values)
        cuda.memcpy_htod(self._mem_class_labels, self.class_labels)

        self.attribute_index = {k: x for x, k in enumerate(dataset.columns)}
        self._mem_candidates = None
        self.n_objects, self.n_attributes = dataset.shape

        # self._mem_labels = cuda.mem_alloc(self._n_objects * np.float32(1).itemsize)

    def device_gain_ratio(self, subset, attribute, candidates):
        """

        :type attribute: unicode
        :param attribute:
        :type candidates: numpy.ndarray
        :param candidates:
        :return:
        """

        n_candidates = candidates.shape[0]

        _threads_per_block = ((n_candidates / CudaHandler._MIN_N_THREADS) + 1) * CudaHandler._MIN_N_THREADS
        if _threads_per_block > CudaHandler._MAX_N_THREADS:
            warnings.warn(
                'Warning: using more threads per GPU than allowed! Rolling back to ' + str(self._MAX_N_THREADS) + '.')
            _threads_per_block = CudaHandler._MAX_N_THREADS

        n_blocks = (n_candidates / _threads_per_block) + 1
        _grid_size = (
            int(np.sqrt(n_blocks)),
            int(np.sqrt(n_blocks))
        )

        _mem_candidates = cuda.mem_alloc(candidates.nbytes)
        cuda.memcpy_htod(_mem_candidates, candidates)  # send info to gpu memory
        ratios = np.empty(candidates.shape[0], dtype=np.float32)

        self._func_gain_ratio(
            self._mem_dataset,
            np.int32(self.n_objects),
            np.int32(self.n_attributes),
            cuda.In(subset),
            np.int32(self.attribute_index[attribute]),
            np.int32(n_candidates),
            _mem_candidates,
            self._mem_class_labels,
            np.int32(self.class_labels.shape[0]),
            block=(_threads_per_block, 1, 1),  # block size
            grid=_grid_size
        )

        cuda.memcpy_dtoh(ratios, _mem_candidates)  # send info to gpu memory
        return ratios

    def gain_ratio(self, subset, subset_left, subset_right, target_attr):
        warnings.filterwarnings('error')

        ig = self.host_information_gain(subset, subset_left, subset_right, target_attr)
        si = self.__split_info__(subset, subset_left, subset_right)

        try:
            gr = ig / si
        except RuntimeWarning as rw:
            if si == 0:
                gr = 0.
            else:
                raise rw

        warnings.filterwarnings('default')

        return gr

    @staticmethod
    def __split_info__(subset, subset_left, subset_right):
        sum_term = 0.
        for child_subset in [subset_left, subset_right]:
            temp = (child_subset.shape[0] / float(subset.shape[0]))
            try:
                sum_term += temp * np.log2(temp)
            except RuntimeWarning as rw:
                if temp == 0:
                    pass
                else:
                    raise rw

        return -sum_term

    def host_information_gain(self, subset, subset_left, subset_right, target_attr):
        sum_term = 0.
        for child_subset in [subset_left, subset_right]:
            sum_term += (child_subset.shape[0] / float(subset.shape[0])) * self.host_entropy(child_subset, target_attr)

        ig = self.host_entropy(subset, target_attr) - sum_term
        return ig

    @staticmethod
    def host_entropy(subset, target_attr):
        # the smaller, the better
        size = float(subset.shape[0])

        counter = Counter(subset[target_attr])

        _entropy = 0.
        for c, q in counter.iteritems():
            _entropy += (q / size) * np.log2(q / size)

        return -1. * _entropy


def main():
    from sklearn import datasets
    dt = datasets.load_iris()
    df = pd.DataFrame(
        data=np.hstack((dt.data.astype(np.float32), dt.target[:, np.newaxis].astype(np.float32))),
        columns=np.hstack((dt.feature_names, 'class'))
    )
    inst = CudaHandler(df)

    attr = df.columns[0]
    candidates = df[attr].unique()

    ratios = inst.device_gain_ratio(np.ones(df.shape[0], dtype=np.int32), attr, candidates)
    print ratios
    print candidates


if __name__ == '__main__':
    main()
