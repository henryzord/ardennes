import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from base import Handler

import os
import warnings

import numpy as np


class CudaHandler(Handler):
    _MIN_N_THREADS = 32
    _MAX_N_THREADS = 1024
    _N_OUTPUT = 3

    def __init__(self, dataset):
        """

        :type dataset: pandas.DataFrame
        :param dataset:
        """

        super(CudaHandler, self).__init__(dataset)

        sep = '\\' if os.name == 'nt' else '/'

        cur_path = os.path.abspath(__file__)
        split = '/'.join(cur_path.split(sep)[:-1])

        kernel = open(os.path.join(split, 'kernel.cu'), 'r').read()
        mod = SourceModule(source=kernel)

        self._func_gain_ratio = mod.get_function("gain_ratio")

        self._mem_dataset = cuda.mem_alloc(self.dataset.values.nbytes)
        self._mem_class_labels = cuda.mem_alloc(self.numerical_class_labels.nbytes)
        cuda.memcpy_htod(self._mem_dataset, self.dataset.values.ravel())
        cuda.memcpy_htod(self._mem_class_labels, self.numerical_class_labels)

        self._mem_candidates = None
        self.n_objects, self.n_attributes = dataset.shape

    def batch_gain_ratio(self, subset_index, attribute, candidates):
        """

        :type attribute: unicode
        :param attribute:
        :type candidates: numpy.ndarray
        :param candidates:
        :return:
        """

        n_candidates = candidates.shape[0]
        candidates = candidates.astype(np.float32)
        subset_index = subset_index.astype(np.int32)

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

        self._func_gain_ratio(
            self._mem_dataset,
            np.int32(self.n_objects),
            np.int32(self.n_attributes),
            cuda.In(subset_index),
            np.int32(self.attribute_index[attribute]),
            np.int32(n_candidates),
            _mem_candidates,
            np.int32(self.class_labels.shape[0]),
            self._mem_class_labels,
            block=(_threads_per_block, 1, 1),  # block size
            grid=_grid_size
        )

        cuda.memcpy_dtoh(candidates, _mem_candidates)  # send info to gpu memory
        return candidates
