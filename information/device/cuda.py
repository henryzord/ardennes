# noinspection PyUnresolvedReferences
import os

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np

from __base__ import Device


class CudaDevice(Device):
    MIN_N_THREADS = 32
    MAX_N_THREADS = 1024

    def __init__(self, dataset):
        super(CudaDevice, self).__init__(dataset)

        # noinspection PyUnresolvedReferences
        import pycuda.autoinit

        kernel = open(os.path.join(self._split, 'kernel.cu'), 'r').read()
        mod = SourceModule(source=kernel)

        self._func_gain_ratio = mod.get_function('gain_ratio')

        self.mem_dataset = cuda.mem_alloc(dataset.values.nbytes)
        self.mem_class_labels = cuda.mem_alloc(self.numerical_class_labels.nbytes)
        cuda.memcpy_htod(self.mem_dataset, dataset.values.ravel())
        cuda.memcpy_htod(self.mem_class_labels, self.numerical_class_labels)

    def device_gain_ratio(self, subset_index, attribute, candidates):
        n_candidates = candidates.shape[0]
        candidates = candidates.astype(np.float32)

        if n_candidates > CudaDevice.MAX_N_THREADS:
            raise NotImplementedError('Support for higher than %d threads per kernel launch not implemented!' % CudaDevice.MAX_N_THREADS)

        # _threads_per_block = ((n_candidates / CudaMaster.MIN_N_THREADS) + 1) * CudaMaster.MIN_N_THREADS
        # n_blocks = math.ceil(np.log10(n_candidates) / np.log10(CudaMaster.MIN_N_THREADS))
        n_blocks = 1
        n_threads_per_block = CudaDevice.MAX_N_THREADS
        grid_size = (1, 1)

        _mem_candidates = cuda.mem_alloc(candidates.nbytes)
        cuda.memcpy_htod(_mem_candidates, candidates)  # send info to gpu memory

        self._func_gain_ratio(
            self.mem_dataset,
            np.int32(self.n_objects),
            np.int32(self.n_attributes),
            cuda.In(subset_index.astype(np.int32)),
            np.int32(self.attribute_index[attribute]),
            np.int32(n_candidates),
            _mem_candidates,
            np.int32(self.class_labels.shape[0]),
            self.mem_class_labels,
            block=(n_threads_per_block, 1, 1),  # block size
            grid=grid_size
        )

        cuda.memcpy_dtoh(candidates, _mem_candidates)  # send info to gpu memory
        return candidates


