#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np
import os

from __base__ import Device


class CLDevice(Device):
    MIN_N_THREADS = 32
    MAX_N_THREADS = 1024

    def __init__(self, dataset, dataset_info):
        super(CLDevice, self).__init__(dataset, dataset_info)

        kernel = open(os.path.join(self._split, 'kernel.cl'), 'r').read()

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.flags = cl.mem_flags
        self.mem_dataset = cl.Buffer(
            self.ctx, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=self.dataset.values.ravel()
        )  # transfers dataset to device memory

        self.prg = cl.Program(self.ctx, kernel).build()  # builds program; registers functions

        self._func_gain_ratio = self.prg.gain_ratio
        self._func_predict = self.prg.predict

    def get_gain_ratios(self, subset_index, attribute, candidates):
        n_candidates = candidates.shape[0]
        candidates = candidates.astype(np.float32)

        if n_candidates > CLDevice.MAX_N_THREADS:
            raise NotImplementedError(
                'Support for higher than %d threads per kernel launch not implemented!' % CLDevice.MAX_N_THREADS
            )

        n_threads = n_candidates if (n_candidates % CLDevice.MIN_N_THREADS == 0) else \
            ((n_candidates / CLDevice.MIN_N_THREADS) + 1) * CLDevice.MIN_N_THREADS

        _mem_candidates = cl.Buffer(
            self.ctx, self.flags.READ_WRITE | self.flags.COPY_HOST_PTR, hostbuf=candidates
        )
        _mem_subset_index = cl.Buffer(
            self.ctx, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=subset_index.astype(np.int32)
        )

        global_size = (n_threads, )  # any size you want, but must be a multiple of 32
        local_size = (CLDevice.MIN_N_THREADS, )  # must be a multiple of 32

        self._func_gain_ratio(  # returns an event, for blocking
            self.queue,
            global_size,
            local_size,
            self.mem_dataset,
            np.int32(self.dataset_info.n_objects),
            np.int32(self.dataset_info.n_attributes),
            _mem_subset_index,
            np.int32(self.dataset_info.attribute_index[attribute]),
            np.int32(n_candidates),
            _mem_candidates,
            np.int32(self.dataset_info.class_labels.shape[0])
        )

        cl.enqueue_copy(self.queue, candidates, _mem_candidates)  # returns an event, for blocking

        return candidates

    def predict(self, data, dt, inner=False):
        if inner is False:
            return super(CLDevice, self).predict(data, dt, inner)
        else:
            n_predictions = data.shape[0]

            predictions = np.empty(n_predictions, dtype=np.int32)

            n_threads = n_predictions if (n_predictions % CLDevice.MIN_N_THREADS == 0) else \
                ((n_predictions / CLDevice.MIN_N_THREADS) + 1) * CLDevice.MIN_N_THREADS

            dt_matrix = dt.to_matrix()

            _mem_tree = cl.Buffer(
                self.ctx, self.flags.READ_ONLY | self.flags.COPY_HOST_PTR, hostbuf=dt_matrix.values.ravel()
            )

            _mem_predictions = cl.Buffer(
                self.ctx, self.flags.WRITE_ONLY | self.flags.COPY_HOST_PTR, hostbuf=predictions
            )

            global_size = (n_threads, )  # any size you want, but must be a multiple of 32
            local_size = (CLDevice.MIN_N_THREADS, )  # must be a multiple of 32

            self._func_predict(  # returns an event, for blocking
                self.queue,
                global_size,
                local_size,
                self.mem_dataset,
                np.int32(self.dataset_info.n_objects),
                np.int32(self.dataset_info.n_attributes),
                _mem_tree,
                np.int32(dt_matrix.shape[1]),
                np.int32(n_predictions),
                _mem_predictions,
                np.int32(dt.multi_tests),
            )

            cl.enqueue_copy(self.queue, predictions, _mem_predictions)  # returns an event, for blocking

            predictions = [self.dataset_info.inv_class_label_index[x] for x in predictions]
            return predictions

